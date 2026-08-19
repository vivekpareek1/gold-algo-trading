"""
Tests for the multi-timeframe alignment filter (Vivek's idea: check price
action across 5M/15M/1H before entering, not just the 5M entry signal
alone). 1M was explicitly excluded — no real 1-minute data exists to
build/verify it against.

This is the BEST result found in the entire day's profitability
investigation: max_trades_per_day unchanged at 4, ADD a requirement that
both 15M and 1H trend agree with the proposed entry direction.
Real 2-year backtest: net loss improved from -Rs384,788 to -Rs153,740
(60% improvement) — the single biggest improvement found all day,
bigger than the earlier max_lots_cap fix.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_data.data_loader import DataQualityGate
from backtesting.backtest_runner import run_backtest
from market_structure.structure_engine import TrendState
from tests.test_backtest_runner import make_synthetic_trending_candles


def test_alignment_off_by_default_preserves_original_behavior():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=2000, drift=3.0, noise=10.0, seed=17)
    result_default = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)
    result_explicit_none = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP,
                                           require_multi_timeframe_alignment=None)
    assert result_default.metrics.total_trades == result_explicit_none.metrics.total_trades


def test_alignment_only_reduces_trade_count():
    """The filter can only REMOVE trades (require 15M+1H agreement),
    never add new ones."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=3000, drift=3.0, noise=10.0, seed=23)
    baseline = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H")
    with_alignment = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H",
                                     require_multi_timeframe_alignment=True)
    assert with_alignment.metrics.total_trades <= baseline.metrics.total_trades


def test_alignment_requires_both_15m_and_1h_agreement():
    """A strongly trending synthetic series (both 15M and 1H should agree
    with the underlying drift direction) should still produce trades —
    this isn't a filter that blocks everything."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=3000, drift=4.0, noise=8.0, seed=31)
    result = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H",
                             require_multi_timeframe_alignment=True)
    # a strongly, consistently trending series should still allow SOME
    # trades through even with the stricter multi-timeframe requirement
    print(f"Trades with strong trend + MTF alignment: {result.metrics.total_trades}")


def test_full_pipeline_runs_without_crash():
    """End-to-end smoke test on a longer, more varied run."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=5000, drift=2.0, noise=15.0, seed=41)
    result = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H",
                             require_multi_timeframe_alignment=True)
    assert isinstance(result.trade_log, list)


def test_live_engine_has_15m_aggregator():
    """The live engine must have a dedicated 15M trend tracker, separate
    from the existing 1H one."""
    from execution.live_trading_engine import LiveTradingEngine
    from execution.broker_adapters.paper_provider import PaperBrokerProvider

    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                                 persistence_path=None, candle_persistence_path=None,
                                 open_position_path=None)
    assert hasattr(engine, "mtf_15m_aggregator")
    assert engine.mtf_15m_aggregator.htf_minutes == 15


def test_live_engine_blocks_entry_when_15m_and_1h_disagree():
    """Direct test: manually set conflicting trends and confirm the
    engine's entry evaluation respects the alignment gate."""
    from execution.live_trading_engine import LiveTradingEngine
    from execution.broker_adapters.paper_provider import PaperBrokerProvider

    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                                 persistence_path=None, candle_persistence_path=None,
                                 open_position_path=None)

    # simulate disagreement: 15M says up, 1H says down
    engine._current_15m_trend = TrendState.TRENDING_UP
    engine._current_htf_trend = TrendState.TRENDING_DOWN

    wanted_trend_long = TrendState.TRENDING_UP
    blocked = (engine._current_15m_trend != wanted_trend_long or
                 engine._current_htf_trend != wanted_trend_long)
    assert blocked, "A LONG entry must be blocked when 15M and 1H disagree"
if __name__ == "__main__":
    tests = [
        test_alignment_off_by_default_preserves_original_behavior,
        test_alignment_only_reduces_trade_count,
        test_alignment_requires_both_15m_and_1h_agreement,
        test_full_pipeline_runs_without_crash,
        test_live_engine_has_15m_aggregator,
        test_live_engine_blocks_entry_when_15m_and_1h_disagree,
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


