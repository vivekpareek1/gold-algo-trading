import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_structure.structure_engine import TrendState
from backtesting.backtest_runner import (
    run_backtest, build_htf_trend_lookup, _htf_trend_as_of, OHLCV
)
from tests.test_backtest_runner import make_synthetic_trending_candles


def test_htf_lookup_builds_without_crashing():
    candles = make_synthetic_trending_candles(n=300, drift=3.0, noise=10.0)
    lookup = build_htf_trend_lookup(candles, base_timeframe="5M", htf_timeframe="1H")
    print(f"HTF lookup has {len(lookup)} completed 1H buckets from 300 5M candles")
    assert isinstance(lookup, dict)
    assert len(lookup) > 0, "Should produce at least some completed HTF buckets from 300 base candles"


def test_htf_lookup_only_contains_completed_buckets():
    """The trailing/in-progress bucket at the end of the data must NOT appear."""
    candles = make_synthetic_trending_candles(n=50, drift=2.0, noise=8.0)  # not enough for many full 1H buckets
    lookup = build_htf_trend_lookup(candles, base_timeframe="5M", htf_timeframe="1H")

    from market_data.resampler import resample
    all_buckets = resample(candles, "5M", "1H")
    incomplete_ts = {rc.ohlcv.ts for rc in all_buckets if not rc.is_complete}

    for ts in lookup:
        assert ts not in incomplete_ts, \
            f"Lookup must not contain incomplete bucket ts={ts} — look-ahead bias risk"


def test_htf_trend_as_of_never_looks_forward():
    """Querying for a timestamp BEFORE any HTF data exists must return the
    conservative default, never a trend computed from later data."""
    lookup = {1000: TrendState.TRENDING_UP, 4600: TrendState.TRENDING_DOWN}
    result = _htf_trend_as_of(base_ts=500, htf_lookup=lookup)  # before any bucket
    assert result == TrendState.RANGE, \
        f"Querying before any HTF data exists must default to RANGE, got {result}"


def test_htf_trend_as_of_picks_most_recent_completed():
    lookup = {1000: TrendState.TRENDING_UP, 4600: TrendState.TRENDING_DOWN, 8200: TrendState.RANGE}
    # query a timestamp between the 2nd and 3rd bucket -> should get the 2nd bucket's trend
    result = _htf_trend_as_of(base_ts=6000, htf_lookup=lookup)
    assert result == TrendState.TRENDING_DOWN, \
        f"Should pick the most recent COMPLETED bucket at or before base_ts, got {result}"


def test_htf_trend_as_of_never_uses_future_bucket():
    """Query exactly at a base_ts just before a later bucket — must not use that later bucket."""
    lookup = {1000: TrendState.TRENDING_UP, 4600: TrendState.TRENDING_DOWN}
    result = _htf_trend_as_of(base_ts=4599, htf_lookup=lookup)  # 1 second before the 2nd bucket
    assert result == TrendState.TRENDING_UP, \
        f"Must use the EARLIER bucket only, not peek at the bucket starting 1 second later, got {result}"


def test_full_backtest_runs_with_real_multi_timeframe():
    """End-to-end: run_backtest with htf_timeframe='1H' instead of a constant override."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=400, drift=2.5, noise=12.0, seed=99)
    result = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H")
    print(f"Multi-TF backtest: trades={result.metrics.total_trades}, "
          f"signals={len(result.signal_log)}, expectancy={result.metrics.expectancy_r:.3f}R")
    assert result is not None
    assert len(result.signal_log) > 0


def test_multi_timeframe_vs_constant_produce_different_but_valid_results():
    """Sanity: real multi-TF mode and constant-override mode should both run
    cleanly, and don't have to produce identical results (that would suggest
    the HTF lookup isn't actually being used)."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=300, drift=2.0, noise=10.0, seed=55)

    result_constant = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)
    result_real = run_backtest(candles, settings, base_timeframe="5M", htf_timeframe="1H")

    print(f"Constant mode: {result_constant.metrics.total_trades} trades")
    print(f"Real multi-TF mode: {result_real.metrics.total_trades} trades")
    # not asserting they differ (could coincidentally match), just confirming both run cleanly
    assert result_constant is not None
    assert result_real is not None


if __name__ == "__main__":
    tests = [
        test_htf_lookup_builds_without_crashing,
        test_htf_lookup_only_contains_completed_buckets,
        test_htf_trend_as_of_never_looks_forward,
        test_htf_trend_as_of_picks_most_recent_completed,
        test_htf_trend_as_of_never_uses_future_bucket,
        test_full_backtest_runs_with_real_multi_timeframe,
        test_multi_timeframe_vs_constant_produce_different_but_valid_results,
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
