"""
Tests for a real correction: "change" was computed against TODAY'S OPEN
price, but real trading platforms compute it against the PREVIOUS
TRADING DAY'S CLOSE — these differ whenever there's a gap between
sessions (common for gold, which trades near 24hrs internationally).
The user caught this comparing against a real platform's convention.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider


def _engine(candle_path=None):
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                               persistence_path=None, candle_persistence_path=candle_path,
                               open_position_path=None)


def test_prev_day_close_captured_on_live_day_transition():
    engine = _engine()
    day1_ts = 1735689600
    engine.on_tick(LiveTick(ts=day1_ts, open=154900, high=155010, low=154890,
                              close=154950, volume=100))
    engine.on_tick(LiveTick(ts=day1_ts + 300, open=154950, high=155050,
                              low=154940, close=155000, volume=100))

    day2_ts = day1_ts + 86400
    snap = engine.on_tick(LiveTick(ts=day2_ts, open=155200, high=155220,
                                     low=155180, close=155210, volume=100))

    assert snap["prev_day_close_price"] == 155000, \
        "Must capture the LAST close from the day that just ended"


def test_change_reflects_overnight_gap_not_just_intraday_move():
    """THE core bug: a gap between sessions must be captured in 'change' —
    the old day-open-based method would silently miss it entirely."""
    engine = _engine()
    day1_ts = 1735689600
    engine.on_tick(LiveTick(ts=day1_ts, open=154900, high=155010, low=154890,
                              close=155000, volume=100))

    day2_ts = day1_ts + 86400
    # day 2 opens with a 200-point gap UP from day 1's close
    snap = engine.on_tick(LiveTick(ts=day2_ts, open=155200, high=155220,
                                     low=155180, close=155210, volume=100))

    change_vs_prev_close = snap["ltp"] - snap["prev_day_close_price"]
    change_vs_today_open = snap["ltp"] - snap["day_open_price"]

    assert abs(change_vs_prev_close - 210) < 1, \
        "Change vs previous close should capture the full move including the gap"
    assert abs(change_vs_today_open - 10) < 1, \
        "Change vs today's open would (wrongly) show only the tiny intraday move, " \
        "missing the 200-point gap entirely — this is exactly the bug being fixed"


def test_prev_day_close_derived_from_persisted_history_after_restart():
    """A restart on a NEW day must still correctly find yesterday's close
    from persisted candle_history — not just from a live day-transition
    observed in the current session."""
    with tempfile.TemporaryDirectory() as d:
        candle_path = os.path.join(d, "candles.jsonl")

        e1 = _engine(candle_path=candle_path)
        day1_ts = 1735689600
        e1.on_tick(LiveTick(ts=day1_ts, open=154900, high=155010, low=154890,
                              close=154950, volume=100))
        e1.on_tick(LiveTick(ts=day1_ts + 300, open=154950, high=155050,
                              low=154940, close=155000, volume=100))

        e2 = _engine(candle_path=candle_path)
        day2_ts = day1_ts + 86400
        snap = e2.on_tick(LiveTick(ts=day2_ts, open=155150, high=155180,
                                     low=155140, close=155160, volume=100))

        assert snap["prev_day_close_price"] == 155000


def test_no_persisted_history_and_no_prior_day_gives_none_gracefully():
    """First-ever tick, no persisted history — must not crash, prev_day_close
    stays None until a genuine previous day's data exists."""
    engine = _engine()
    snap = engine.on_tick(LiveTick(ts=1735689600, open=155000, high=155010,
                                     low=154990, close=155000, volume=100))
    assert snap["prev_day_close_price"] is None


def test_api_snapshot_includes_prev_day_close_price():
    from fastapi.testclient import TestClient
    from api.main import app
    resp = TestClient(app).get("/api/snapshot")
    assert resp.status_code == 200
    assert "prev_day_close_price" in resp.json()


if __name__ == "__main__":
    tests = [
        test_prev_day_close_captured_on_live_day_transition,
        test_change_reflects_overnight_gap_not_just_intraday_move,
        test_prev_day_close_derived_from_persisted_history_after_restart,
        test_no_persisted_history_and_no_prior_day_gives_none_gracefully,
        test_api_snapshot_includes_prev_day_close_price,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__} -> {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
