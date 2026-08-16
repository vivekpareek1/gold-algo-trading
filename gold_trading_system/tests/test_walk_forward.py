import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_structure.structure_engine import TrendState
from backtesting.walk_forward import run_walk_forward, _compute_decay_pct
from tests.test_backtest_runner import make_synthetic_trending_candles


def test_decay_calculation_no_decay():
    """Test performs as well as train -> 0% decay."""
    decay = _compute_decay_pct(train_expectancy=1.0, test_expectancy=1.2)
    assert decay == 0.0


def test_decay_calculation_partial_decay():
    """Train=1.0, test=0.5 -> 50% decay."""
    decay = _compute_decay_pct(train_expectancy=1.0, test_expectancy=0.5)
    assert abs(decay - 50.0) < 0.01


def test_decay_calculation_sign_flip_is_severe():
    """Train positive, test negative -> should be treated as heavy decay, not a small number."""
    decay = _compute_decay_pct(train_expectancy=1.0, test_expectancy=-0.5)
    print(f"Sign-flip decay: {decay}%")
    assert decay >= 100.0, \
        f"Going from +1.0R train to -0.5R test is a total loss of edge, should show >=100% decay, got {decay}"


def test_decay_calculation_train_already_negative():
    decay = _compute_decay_pct(train_expectancy=-0.5, test_expectancy=-1.0)
    assert decay == 100.0, "Train already negative and test worse -> treat as full decay flag"


def test_too_little_data_raises():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=50)  # well under the 100 minimum
    try:
        run_walk_forward(candles, settings, num_windows=4)
        assert False, "Expected ValueError for insufficient data"
    except ValueError as e:
        print(f"Correctly rejected: {e}")


def test_windows_too_small_raises():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=150)
    try:
        run_walk_forward(candles, settings, num_windows=20)  # 150/20 = 7.5 candles per window
        assert False, "Expected ValueError for windows too small to be meaningful"
    except ValueError as e:
        print(f"Correctly rejected: {e}")


def test_invalid_split_percentages_raise():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=500)
    try:
        run_walk_forward(candles, settings, train_pct=0.7, validation_pct=0.5)  # sums > 1.0
        assert False, "Expected ValueError when train_pct + validation_pct >= 1.0"
    except ValueError as e:
        print(f"Correctly rejected: {e}")


def test_full_walk_forward_runs_and_produces_correct_window_count():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=1000, drift=2.0, noise=12.0, seed=11)
    result = run_walk_forward(candles, settings, num_windows=3,
                                htf_trend_override=TrendState.TRENDING_UP)
    print(f"Walk-forward: {result.num_windows} windows, "
          f"avg_train={result.avg_train_expectancy}, avg_test={result.avg_test_expectancy}, "
          f"decay={result.train_to_test_decay_pct}%, overfit={result.is_likely_overfit}")

    assert result.num_windows == 3
    assert len(result.windows) == 3
    for w in result.windows:
        assert w.train_start_idx < w.train_end_idx <= w.validation_start_idx
        assert w.validation_start_idx < w.validation_end_idx <= w.test_start_idx
        assert w.test_start_idx < w.test_end_idx


def test_windows_are_chronologically_non_overlapping_within_a_window():
    """Train, validation, and test slices within ONE window must not overlap
    (that would leak future data into training) — check index ordering directly."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=800, drift=1.5, noise=10.0, seed=3)
    result = run_walk_forward(candles, settings, num_windows=2,
                                htf_trend_override=TrendState.TRENDING_UP)

    for w in result.windows:
        assert w.train_end_idx == w.validation_start_idx, \
            "Validation must start exactly where training ends — no gap, no overlap"
        assert w.validation_end_idx == w.test_start_idx, \
            "Test must start exactly where validation ends — no gap, no overlap"


def test_consistency_flag_present_and_readable():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=600, drift=2.0, noise=10.0, seed=21)
    result = run_walk_forward(candles, settings, num_windows=2,
                                htf_trend_override=TrendState.TRENDING_UP)
    print(f"Consistency flag: {result.consistency_flag}")
    assert isinstance(result.consistency_flag, str)
    assert len(result.consistency_flag) > 10


def test_windows_with_positive_test_expectancy_field_populated():
    """
    Regression test for a real finding on 2-year MCX data: an average-based
    decay% was dominated by one window with an unusually strong training
    expectancy, making 3-of-4 genuinely positive out-of-sample windows look
    like uniform overfitting. windows_with_positive_test_expectancy must be
    reported so this can't be misread from the aggregate % alone.
    """
    settings = Settings()
    candles = make_synthetic_trending_candles(n=1000, drift=2.0, noise=12.0, seed=44)
    result = run_walk_forward(candles, settings, num_windows=3,
                                htf_trend_override=TrendState.TRENDING_UP)

    # cross-check the reported count against the actual per-window test metrics
    actual_positive = sum(1 for w in result.windows if w.test_metrics.expectancy_r > 0)
    print(f"Reported positive windows: {result.windows_with_positive_test_expectancy}, "
          f"actual: {actual_positive}, pct: {result.windows_with_positive_test_expectancy_pct}")

    assert result.windows_with_positive_test_expectancy == actual_positive, \
        "windows_with_positive_test_expectancy must match the real per-window test data"
    assert 0 <= result.windows_with_positive_test_expectancy_pct <= 100
    expected_pct = round(actual_positive / result.num_windows * 100, 1)
    assert abs(result.windows_with_positive_test_expectancy_pct - expected_pct) < 0.01


if __name__ == "__main__":
    tests = [
        test_decay_calculation_no_decay,
        test_decay_calculation_partial_decay,
        test_decay_calculation_sign_flip_is_severe,
        test_decay_calculation_train_already_negative,
        test_too_little_data_raises,
        test_windows_too_small_raises,
        test_invalid_split_percentages_raise,
        test_full_walk_forward_runs_and_produces_correct_window_count,
        test_windows_are_chronologically_non_overlapping_within_a_window,
        test_consistency_flag_present_and_readable,
        test_windows_with_positive_test_expectancy_field_populated,
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
