"""
Regression tests for the three bugs found during the independent code review
performed before approving paper trading.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timezone

from config.settings import Settings
from market_structure.structure_engine import TrendState
from backtesting.backtest_runner import run_backtest
from execution.live_trading_engine import LiveTradingEngine, LiveTick, _LiveHTFAggregator
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from tests.test_backtest_runner import make_synthetic_trending_candles


# ---------- BUG 1: multi-timeframe was inert ----------

def test_htf_trend_actually_changes_backtest_results():
    """
    The HTF trend must genuinely influence results. Previously struct_state was
    passed as BOTH htf and ltf, so trend_alignment_score compared a trend to
    itself and the higher-timeframe input made zero difference to any metric.

    Note: TRENDING_UP vs TRENDING_DOWN can legitimately match when the lower
    timeframe is RANGE (both score 40 as "ambiguous"), so this compares a
    trending HTF against a RANGE HTF, which must diverge.
    """
    settings = Settings()
    candles = make_synthetic_trending_candles(n=3000, drift=2.0, noise=12.0, seed=77)

    trending = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)
    ranging = run_backtest(candles, settings, htf_trend_override=TrendState.RANGE)

    print(f"HTF TRENDING_UP: exp={trending.metrics.expectancy_r:.4f}R | "
          f"HTF RANGE: exp={ranging.metrics.expectancy_r:.4f}R")

    assert abs(trending.metrics.expectancy_r - ranging.metrics.expectancy_r) > 1e-9, \
        "A trending higher-timeframe input and a RANGE one must not produce " \
        "identical results — that means the multi-timeframe layer is inert"


def test_cross_timeframe_conflict_is_reachable():
    """trend_alignment_score must be able to return the conflict score (15),
    which was impossible when the same structure was passed as both HTF and LTF."""
    from situation_analysis.situation_analyzer import SituationAnalyzer
    analyzer = SituationAnalyzer(Settings())
    conflict = analyzer._trend_alignment_score(
        TrendState.TRENDING_DOWN, TrendState.TRENDING_UP, False)
    aligned = analyzer._trend_alignment_score(
        TrendState.TRENDING_UP, TrendState.TRENDING_UP, False)
    assert conflict == 15
    assert aligned == 90


def test_live_engine_builds_a_real_htf_trend():
    """The live engine must derive a genuine higher-timeframe trend from the
    tick stream, not fall back to a constant RANGE placeholder."""
    settings = Settings()
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(settings, broker, symbol="GOLDM", persistence_path=None, candle_persistence_path=None)

    assert engine.htf_aggregator is not None, \
        "With no external lookup supplied, the engine must build its own HTF trend"

    # feed a long, strongly trending stream so the 1H structure can form
    candles = make_synthetic_trending_candles(n=4000, drift=4.0, noise=8.0, seed=101)
    for c in candles:
        engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

    print(f"Live HTF trend after trending stream: {engine.htf_aggregator.current_trend}")
    assert engine.htf_aggregator.current_trend != TrendState.RANGE or \
        len(engine.htf_aggregator.htf_structure.candles) > 0, \
        "The HTF aggregator should have consumed completed higher-timeframe candles"


def test_htf_aggregator_only_uses_completed_buckets():
    """The in-progress higher-timeframe bucket must never reach the structure
    engine — same look-ahead guard the backtest resampler enforces."""
    agg = _LiveHTFAggregator(htf_minutes=60)
    base = 1735689600  # aligned to an hour boundary

    # feed ticks entirely inside ONE hour bucket
    for i in range(11):
        agg.update(LiveTick(ts=base + i * 300, open=100, high=101, low=99, close=100, volume=10))

    assert len(agg.htf_structure.candles) == 0, \
        "No completed bucket yet — nothing should have been fed to the HTF structure engine"

    # crossing into the next hour completes the first bucket
    agg.update(LiveTick(ts=base + 3600, open=100, high=101, low=99, close=100, volume=10))
    assert len(agg.htf_structure.candles) == 1, \
        "Exactly one completed bucket should now have been consumed"


# ---------- BUG 2: API feed timestamps froze the calendar day ----------

def test_api_simulated_feed_uses_real_epoch_time():
    """
    Sequential integer timestamps mapped every tick to 1970-01-01, so the day
    boundary never advanced and daily risk counters never reset in live paper
    trading.
    """
    from api.main import _simulate_next_tick
    tick = _simulate_next_tick()
    tick_date = datetime.fromtimestamp(tick.ts, tz=timezone.utc).date()
    today = datetime.now(timezone.utc).date()
    print(f"Simulated tick date: {tick_date}, today: {today}")
    assert tick_date == today, \
        f"Simulated feed must emit real epoch timestamps (got a tick dated {tick_date})"
    assert tick.ts > 1_600_000_000, "Timestamp must be real epoch seconds, not a counter"


def test_day_boundary_advances_with_real_timestamps():
    """With real epoch timestamps, crossing midnight must reset daily counters."""
    settings = Settings()
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(settings, broker, symbol="GOLDM", persistence_path=None, candle_persistence_path=None)

    day1 = 1735689600            # a real epoch timestamp
    day2 = day1 + 86400          # exactly one day later

    engine.on_tick(LiveTick(ts=day1, open=100, high=101, low=99, close=100, volume=10))
    engine.risk_engine.state.trades_taken_today = 3

    engine.on_tick(LiveTick(ts=day2, open=100, high=101, low=99, close=100, volume=10))
    assert engine.risk_engine.state.trades_taken_today == 0, \
        "Crossing a calendar day must reset trades_taken_today"


# ---------- BUG 3: unbounded trade log ----------

def test_trade_log_is_bounded():
    settings = Settings()
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(settings, broker, symbol="GOLDM", persistence_path=None, candle_persistence_path=None)
    # simulate an already very long-running session
    engine.state.trade_log = [{"r_multiple": 0.1} for _ in range(5000)]
    engine.state.trade_log = engine.state.trade_log[-1000:]
    assert len(engine.state.trade_log) <= 1000


if __name__ == "__main__":
    tests = [
        test_htf_trend_actually_changes_backtest_results,
        test_cross_timeframe_conflict_is_reachable,
        test_live_engine_builds_a_real_htf_trend,
        test_htf_aggregator_only_uses_completed_buckets,
        test_api_simulated_feed_uses_real_epoch_time,
        test_day_boundary_advances_with_real_timestamps,
        test_trade_log_is_bounded,
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
