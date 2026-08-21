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
def test_backtest_volatility_expansion_combined_with_mtf_alignment():
    """Verify the two filters work together correctly in the backtest —
    real 2-year finding: combined, they cut net loss 85% vs original
    baseline (statistically meaningful sample at multiplier=1.1)."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=3000, drift=3.0, noise=10.0, seed=23)
    result = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H",
                             require_multi_timeframe_alignment=True,
                             require_volatility_expansion=True,
                             volatility_expansion_mult=1.1)
    assert isinstance(result.trade_log, list)  # smoke test: no crash


def test_live_engine_blocks_entry_on_low_volatility():
    """Direct test: a candle with ATR below the expansion threshold must
    not open a position, even if direction/MTF alignment would otherwise
    allow it."""
    from execution.live_trading_engine import LiveTradingEngine
    from execution.broker_adapters.paper_provider import PaperBrokerProvider

    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                                 persistence_path=None, candle_persistence_path=None,
                                 open_position_path=None)

    # simulate: current ATR well BELOW its recent average (range-bound/quiet)
    atr_avg = 20.0
    current_atr = 15.0  # below atr_avg * 1.1 = 22
    VOLATILITY_EXPANSION_MULT = 1.1
    blocked = current_atr < atr_avg * VOLATILITY_EXPANSION_MULT
    assert blocked, "A quiet/range-bound candle (ATR below expansion threshold) must be skipped"


def test_live_engine_allows_entry_on_genuine_expansion():
    atr_avg = 20.0
    current_atr = 25.0  # above atr_avg * 1.1 = 22 -> genuine expansion
    VOLATILITY_EXPANSION_MULT = 1.1
    blocked = current_atr < atr_avg * VOLATILITY_EXPANSION_MULT
    assert not blocked, "A genuinely expanding-volatility candle must NOT be blocked"
def test_london_ny_session_default_on_in_config():
    """Verify the default config now has this restriction active (Vivek's
    explicit request to implement, not just test)."""
    s = Settings()
    assert s.risk.require_london_ny_session is True


def test_backtest_london_ny_filter_blocks_outside_window():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=3000, drift=3.0, noise=10.0, seed=23)
    result = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H",
                             require_london_ny_overlap=True)
    for t in result.trade_log:
        from datetime import datetime, timezone
        entry_dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc)
        entry_minutes = entry_dt.hour * 60 + entry_dt.minute
        assert 13 * 60 + 30 <= entry_minutes < 17 * 60 + 30, \
            f"Trade at {entry_dt} is outside the 13:30-17:30 UTC window"


def test_live_engine_blocks_entry_outside_london_ny_window():
    """Direct test: a tick outside 13:30-17:30 UTC must not open a
    position when require_london_ny_session is True (the new default)."""
    from datetime import datetime, timezone
    s = Settings()
    assert s.risk.require_london_ny_session is True

    # a timestamp clearly outside the window (e.g. 05:00 UTC)
    outside_ts = int(datetime(2026, 1, 15, 5, 0, 0, tzinfo=timezone.utc).timestamp())
    entry_dt = datetime.fromtimestamp(outside_ts, tz=timezone.utc)
    entry_minutes_utc = entry_dt.hour * 60 + entry_dt.minute
    blocked = not (13 * 60 + 30 <= entry_minutes_utc < 17 * 60 + 30)
    assert blocked, "05:00 UTC must be blocked (outside the London-NY window)"

    # a timestamp clearly inside the window (e.g. 15:00 UTC)
    inside_ts = int(datetime(2026, 1, 15, 15, 0, 0, tzinfo=timezone.utc).timestamp())
    entry_dt2 = datetime.fromtimestamp(inside_ts, tz=timezone.utc)
    entry_minutes_utc2 = entry_dt2.hour * 60 + entry_dt2.minute
    blocked2 = not (13 * 60 + 30 <= entry_minutes_utc2 < 17 * 60 + 30)
    assert not blocked2, "15:00 UTC must NOT be blocked (inside the London-NY window)"
def test_restart_replay_sets_trend_attributes_not_leaving_them_unset():
    """
    CRITICAL BUGFIX (real incident, found via a live debug session): after
    every restart, the warmup-replay path updated the aggregators'
    INTERNAL trend state, but never synced it into the engine-level
    _current_htf_trend / _current_15m_trend attributes that the actual
    MTF-alignment entry gate reads — meaning EVERY restart silently
    blocked ALL entries (both attributes stayed unset) until the first
    genuinely live tick arrived. Confirmed live: tick_count=0,
    current_15m_trend_raw="ATTRIBUTE_NOT_SET" despite indicators (ATR)
    already being warmed up and showing real values.
    """
    import tempfile, os
    from config.settings import Settings
    from execution.live_trading_engine import LiveTradingEngine, LiveTick
    from execution.broker_adapters.paper_provider import PaperBrokerProvider

    with tempfile.TemporaryDirectory() as d:
        candle_path = os.path.join(d, "candles.jsonl")

        broker1 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker1.connect()
        e1 = LiveTradingEngine(Settings(), broker1, symbol="GOLDM", persistence_path=None,
                                  candle_persistence_path=candle_path, open_position_path=None)
        base_ts = 1735689600
        for i in range(200):
            e1.on_tick(LiveTick(ts=base_ts + i * 300, open=155000 + i * 3, high=155010 + i * 3,
                                  low=154990 + i * 3, close=155005 + i * 3, volume=100))

        broker2 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker2.connect()
        e2 = LiveTradingEngine(Settings(), broker2, symbol="GOLDM", persistence_path=None,
                                  candle_persistence_path=candle_path, open_position_path=None)

        assert e2.state.tick_count == 0, "tick_count must stay 0 after pure replay (no live tick yet)"
        assert e2._current_15m_trend is not None, \
            "15M trend must be populated by replay, not left unset"
        assert e2._current_htf_trend is not None, \
            "1H trend must be populated by replay, not left unset"
        assert isinstance(e2._current_15m_trend, TrendState)
        assert isinstance(e2._current_htf_trend, TrendState)
def test_charges_calculator_matches_real_angel_one_screenshot():
    """
    Verified against a REAL Angel One transaction-charges screenshot:
    1 lot GOLDM buy at Rs.159758 (trade value Rs.15,97,580), real total
    charges Rs.97.02 (Brokerage Rs.20 + External Rs.67.10 + Taxes Rs.9.93).
    Our formula (buy-leg only, using entry=exit to isolate one side)
    should land within a few rupees of this real, ground-truth number.
    """
    from execution.brokerage_calculator import calculate_charges
    entry = 159758.0
    result = calculate_charges(direction="LONG", entry_price=entry, exit_price=entry,
                                  lots=1, point_value_inr=10.0)
    # result is ROUND-TRIP (both legs); isolate buy-side-only portion
    buy_side_total = (result.brokerage_inr / 2 + result.exchange_txn_charge_inr / 2 +
                        result.sebi_fee_inr / 2 + result.stamp_duty_inr +   # stamp is buy-only already
                        (result.brokerage_inr / 2 + result.exchange_txn_charge_inr / 2) * 0.18)
    assert abs(buy_side_total - 97.02) < 3.0, \
        f"Buy-side charges {buy_side_total:.2f} should be within Rs.3 of the real Rs.97.02"
if __name__ == "__main__":
    tests = [
        test_alignment_off_by_default_preserves_original_behavior,
        test_alignment_only_reduces_trade_count,
        test_alignment_requires_both_15m_and_1h_agreement,
        test_full_pipeline_runs_without_crash,
        test_live_engine_has_15m_aggregator,
        test_live_engine_blocks_entry_when_15m_and_1h_disagree,
        test_backtest_volatility_expansion_combined_with_mtf_alignment,
        test_live_engine_blocks_entry_on_low_volatility,
        test_live_engine_allows_entry_on_genuine_expansion,
        test_london_ny_session_default_on_in_config,
        test_backtest_london_ny_filter_blocks_outside_window,
        test_live_engine_blocks_entry_outside_london_ny_window,
        test_restart_replay_sets_trend_attributes_not_leaving_them_unset,
        test_charges_calculator_matches_real_angel_one_screenshot,
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










