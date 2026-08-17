"""
Regression tests for a real bug: the dashboard's "change" indicator compared
the live price against the close of the most recently loaded 5-minute
candle, which flips sign on ordinary tick noise regardless of the day's
actual trend — a user could see red/down while the session was genuinely
up hundreds of points. Fixed to compare against the day's open, matching
what real trading platforms mean by "change".
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider


def _engine():
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM", persistence_path=None, candle_persistence_path=None), broker


def test_day_open_price_set_on_first_tick():
    engine, _ = _engine()
    engine.on_tick(LiveTick(ts=1735707600, open=154500.0, high=154600.0,
                              low=154400.0, close=154550.0, volume=1000))
    assert engine.state.day_open_price == 154500.0


def test_day_open_price_stable_across_many_candles_same_day():
    """
    THE core scenario: price wiggles up and down across many 5-min candles
    within one day. day_open_price must stay fixed at the FIRST candle's
    open throughout — this is what makes the change indicator stable and
    meaningful instead of flip-flopping every few minutes.
    """
    engine, _ = _engine()
    base_ts = 1735707600  # 09:00 IST-equivalent epoch, arbitrary fixed reference
    prices = [154500, 154480, 154510, 154470, 154600, 154550, 155000, 154900, 155200]
    for i, p in enumerate(prices):
        engine.on_tick(LiveTick(ts=base_ts + i * 300, open=p, high=p + 20,
                                  low=p - 20, close=p + 5, volume=1000))
        assert engine.state.day_open_price == 154500.0, \
            f"day_open_price must not drift after tick {i}, stayed at the session's first open"


def test_day_open_price_resets_on_new_day():
    engine, _ = _engine()
    day1_ts = 1735707600
    engine.on_tick(LiveTick(ts=day1_ts, open=154500.0, high=154600, low=154400, close=154550, volume=1000))
    assert engine.state.day_open_price == 154500.0

    day2_ts = day1_ts + 86400
    engine.on_tick(LiveTick(ts=day2_ts, open=156000.0, high=156100, low=155900, close=156050, volume=1000))
    assert engine.state.day_open_price == 156000.0, \
        "A new trading day must reset the reference to the NEW day's open"


def test_snapshot_exposes_day_open_price():
    engine, _ = _engine()
    engine.on_tick(LiveTick(ts=1735707600, open=154500.0, high=154600, low=154400, close=154550, volume=1000))
    assert engine.state.last_snapshot.get("day_open_price") == 154500.0


def test_api_returns_day_open_price():
    from fastapi.testclient import TestClient
    import api.main as api_main
    original = api_main.LIVE_FEED_ACTIVE
    try:
        api_main.LIVE_FEED_ACTIVE = False  # exercise the simulated path, which also builds real snapshots
        client = TestClient(api_main.app)
        resp = client.get("/api/snapshot")
        assert resp.status_code == 200
        assert "day_open_price" in resp.json()
    finally:
        api_main.LIVE_FEED_ACTIVE = original


def test_dashboard_computes_change_from_day_open_not_rolling_candle():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert "snap.day_open_price" in html, \
        "The dashboard must compute change against day_open_price"
    assert "const prevPrice = lastClose" not in html, \
        "The old rolling-candle-close comparison must be fully removed"


if __name__ == "__main__":
    tests = [
        test_day_open_price_set_on_first_tick,
        test_day_open_price_stable_across_many_candles_same_day,
        test_day_open_price_resets_on_new_day,
        test_snapshot_exposes_day_open_price,
        test_api_returns_day_open_price,
        test_dashboard_computes_change_from_day_open_not_rolling_candle,
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
