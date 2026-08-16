"""
Regression tests for the three issues found reviewing the modules built
after the previous code review (tick aggregator, live price surfacing,
volume-feed sanity).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_data.tick_aggregator import TickAggregator, Tick
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from backtesting.backtest_runner import run_backtest, OHLCV
from market_structure.structure_engine import TrendState
from tests.test_backtest_runner import make_synthetic_trending_candles


def _engine():
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM"), broker


# ---------- FINDING 1: a volume-less feed silently guts the strategy ----------

def test_zero_volume_feed_materially_changes_results():
    """
    Establishes WHY the subscription mode matters. Angel One's LTP mode (1)
    carries no volume field, so every tick arrives with volume=0. This must
    be treated as a broken feed, not a benign default: on real data it
    collapses the strategy rather than degrading it slightly.
    """
    settings = Settings()
    candles = make_synthetic_trending_candles(n=4000, drift=2.0, noise=12.0, seed=5)
    zeroed = [OHLCV(ts=c.ts, open=c.open, high=c.high, low=c.low, close=c.close, volume=0.0)
              for c in candles]

    normal = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)
    broken = run_backtest(zeroed, settings, htf_trend_override=TrendState.TRENDING_UP)

    print(f"with volume: {normal.metrics.total_trades} trades, "
          f"exp={normal.metrics.expectancy_r:+.3f}R | "
          f"zero volume: {broken.metrics.total_trades} trades, "
          f"exp={broken.metrics.expectancy_r:+.3f}R")

    assert normal.metrics.total_trades != broken.metrics.total_trades, \
        "A zero-volume feed must visibly change behaviour — if it doesn't, the " \
        "volume checks aren't actually wired into the decision path"


def test_aggregator_flags_a_volume_less_feed():
    agg = TickAggregator(interval_minutes=5)
    base = 1735689600
    for i in range(60):
        agg.add_tick(Tick(ts=base + i * 10, ltp=63000 + i, volume=0.0))
    assert agg.volume_feed_looks_broken is True, \
        "60 consecutive zero-volume ticks must raise the broken-feed flag"


def test_aggregator_does_not_flag_a_healthy_feed():
    agg = TickAggregator(interval_minutes=5)
    base = 1735689600
    for i in range(60):
        agg.add_tick(Tick(ts=base + i * 10, ltp=63000 + i, volume=1000 + i * 5))
    assert agg.volume_feed_looks_broken is False


def test_aggregator_does_not_flag_too_early():
    """A handful of zero-volume ticks at session open is normal — the flag
    must not fire before there's enough evidence."""
    agg = TickAggregator(interval_minutes=5)
    base = 1735689600
    for i in range(10):
        agg.add_tick(Tick(ts=base + i * 10, ltp=63000, volume=0.0))
    assert agg.volume_feed_looks_broken is False


# ---------- FINDING 2: live price was stale between candle closes ----------

def test_update_live_price_does_not_run_strategy_logic():
    """The lightweight price update must not advance the analysis pipeline —
    no analysis may ever run on an unclosed bar."""
    engine, _ = _engine()
    ticks_before = engine.state.tick_count
    candles_before = len(engine.state.candle_history)

    engine.update_live_price(ltp=63123.5, ts=1735689600)

    assert engine.state.tick_count == ticks_before, \
        "update_live_price must not advance tick_count — that's on_tick's job"
    assert len(engine.state.candle_history) == candles_before, \
        "update_live_price must not append to candle history"
    assert engine.state.last_tick_price == 63123.5


def test_live_price_is_fresher_than_last_completed_candle():
    """After a candle closes, subsequent ticks must still move the displayed
    price — this was the actual gap: the dashboard could sit on a bar close
    for up to a full interval."""
    engine, _ = _engine()
    engine.on_tick(LiveTick(ts=1735689600, open=63000, high=63010, low=62990,
                              close=63000, volume=1000))
    snapshot_price = engine.state.last_snapshot["ltp"]

    engine.update_live_price(ltp=63075.0, ts=1735689700)

    assert snapshot_price == 63000
    assert engine.state.last_tick_price == 63075.0, \
        "The latest tick price must be tracked separately from the last bar close"


def test_api_overlays_live_price_in_live_mode():
    """The snapshot endpoint must surface the fresh tick price, not the
    stale candle close, when a live feed is active."""
    import api.main as api_main
    from fastapi.testclient import TestClient

    engine = api_main.live_engine
    engine.on_tick(LiveTick(ts=1735689600, open=63000, high=63010, low=62990,
                              close=63000, volume=1000))
    engine.update_live_price(ltp=63456.0, ts=1735689900)

    original = api_main.LIVE_FEED_ACTIVE
    try:
        api_main.LIVE_FEED_ACTIVE = True
        snap = api_main._get_or_advance_snapshot()
        print(f"overlaid ltp: {snap['ltp']} (bar close was 63000)")
        assert snap["ltp"] == 63456.0, \
            "In live mode the snapshot must show the latest tick price"
        assert snap["regime_trend"] is not None, \
            "Analysis fields must still come from the last completed candle"
    finally:
        api_main.LIVE_FEED_ACTIVE = original


# ---------- FINDING 3: the broken-volume flag must actually be surfaced ----------

def test_health_endpoint_exposes_volume_feed_status():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "volume_feed_broken" in resp.json(), \
        "The broken-volume detector must be reachable by the dashboard, " \
        "not computed and then discarded"


def test_dashboard_renders_the_volume_warning_element():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    html = client.get("/").text
    assert "volumeWarning" in html
    assert "volume_feed_broken" in html, \
        "The dashboard must actually read the flag, not just define the banner"


if __name__ == "__main__":
    tests = [
        test_zero_volume_feed_materially_changes_results,
        test_aggregator_flags_a_volume_less_feed,
        test_aggregator_does_not_flag_a_healthy_feed,
        test_aggregator_does_not_flag_too_early,
        test_update_live_price_does_not_run_strategy_logic,
        test_live_price_is_fresher_than_last_completed_candle,
        test_api_overlays_live_price_in_live_mode,
        test_health_endpoint_exposes_volume_feed_status,
        test_dashboard_renders_the_volume_warning_element,
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
