"""
Tests for the Daily P&L dashboard panel — real, brokerage-adjusted results
grouped by trading day, so multi-week paper trading progress is visible
without manually totaling individual trades.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from tests.test_backtest_runner import make_synthetic_trending_candles


def _engine(persistence_path=None):
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                               persistence_path=persistence_path, candle_persistence_path=None)


def test_empty_history_returns_empty_list():
    engine = _engine()
    assert engine.get_daily_pnl_history() == []


def test_daily_history_groups_by_calendar_day():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trades.jsonl")
        engine = _engine(persistence_path=path)
        candles = make_synthetic_trending_candles(n=2000, drift=3.0, noise=10.0, seed=13)
        for c in candles:
            engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                      close=c.close, volume=c.volume))

        history = engine.get_daily_pnl_history()
        if history:
            # every trade in trade_log must be accounted for in exactly one day bucket
            total_trades_in_history = sum(d["trade_count"] for d in history)
            trades_with_pnl = [t for t in engine.state.trade_log if "net_pnl_inr" in t]
            assert total_trades_in_history == len(trades_with_pnl)


def test_daily_history_is_most_recent_first():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trades.jsonl")
        engine = _engine(persistence_path=path)
        candles = make_synthetic_trending_candles(n=3000, drift=3.0, noise=10.0, seed=21)
        for c in candles:
            engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                      close=c.close, volume=c.volume))

        history = engine.get_daily_pnl_history()
        if len(history) >= 2:
            dates = [h["date"] for h in history]
            assert dates == sorted(dates, reverse=True), \
                "Daily history must be ordered most-recent-day-first"


def test_daily_history_net_pnl_equals_gross_minus_charges():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trades.jsonl")
        engine = _engine(persistence_path=path)
        candles = make_synthetic_trending_candles(n=2000, drift=3.0, noise=10.0, seed=17)
        for c in candles:
            engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                      close=c.close, volume=c.volume))

        for day in engine.get_daily_pnl_history():
            expected_net = round(day["gross_pnl_inr"] - day["total_charges_inr"], 2)
            assert abs(day["net_pnl_inr"] - expected_net) < 0.02


def test_daily_history_wins_plus_losses_equals_trade_count():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trades.jsonl")
        engine = _engine(persistence_path=path)
        candles = make_synthetic_trending_candles(n=2000, drift=3.0, noise=10.0, seed=29)
        for c in candles:
            engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                      close=c.close, volume=c.volume))

        for day in engine.get_daily_pnl_history():
            assert day["wins"] + day["losses"] == day["trade_count"]


def test_daily_history_reads_full_persisted_file_not_just_bounded_memory():
    """
    THE core purpose: over a multi-week run, in-memory trade_log is bounded
    to 1000 entries, but the daily summary must still be accurate by
    reading the full persisted file — verify by manually shrinking the
    in-memory list while the file still has everything.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trades.jsonl")
        engine = _engine(persistence_path=path)
        candles = make_synthetic_trending_candles(n=2000, drift=3.0, noise=10.0, seed=37)
        for c in candles:
            engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                      close=c.close, volume=c.volume))

        if engine.state.trade_log:
            full_history_before = engine.get_daily_pnl_history()
            # simulate the in-memory bound kicking in
            engine.state.trade_log = []
            full_history_after = engine.get_daily_pnl_history()
            assert full_history_before == full_history_after, \
                "Daily history must be unaffected by in-memory trade_log being trimmed, " \
                "since it reads the persisted file"


def test_api_daily_pnl_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    resp = TestClient(app).get("/api/daily_pnl")
    assert resp.status_code == 200
    assert "days" in resp.json()


def test_dashboard_shows_daily_pnl_panel():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert "dailyPnlList" in html
    assert "loadDailyPnl" in html


if __name__ == "__main__":
    tests = [
        test_empty_history_returns_empty_list,
        test_daily_history_groups_by_calendar_day,
        test_daily_history_is_most_recent_first,
        test_daily_history_net_pnl_equals_gross_minus_charges,
        test_daily_history_wins_plus_losses_equals_trade_count,
        test_daily_history_reads_full_persisted_file_not_just_bounded_memory,
        test_api_daily_pnl_endpoint,
        test_dashboard_shows_daily_pnl_panel,
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