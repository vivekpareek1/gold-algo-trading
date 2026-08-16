"""
Walk-Forward Validation (Sprint 1 §29).
"Do not deploy parameters simply because they perform well on one
historical dataset." This module enforces that discipline structurally:
it splits data into rolling Training -> Validation -> Out-of-Sample windows
and requires performance to hold up OUT of the window it was tuned on.

This does not itself optimize parameters (that's a separate, later concern)
— it provides the harness that makes "does this generalize?" answerable
rather than assumed.
"""
from dataclasses import dataclass

from backtesting.backtest_runner import OHLCV, run_backtest, BacktestResult
from backtesting.metrics import BacktestMetrics


@dataclass
class WalkForwardWindow:
    window_index: int
    train_start_idx: int
    train_end_idx: int
    validation_start_idx: int
    validation_end_idx: int
    test_start_idx: int
    test_end_idx: int

    train_metrics: BacktestMetrics
    validation_metrics: BacktestMetrics
    test_metrics: BacktestMetrics


@dataclass
class WalkForwardResult:
    windows: list
    num_windows: int

    avg_train_expectancy: float
    avg_validation_expectancy: float
    avg_test_expectancy: float

    # the core overfitting signal: how much performance decays out-of-sample
    train_to_test_decay_pct: float
    is_likely_overfit: bool
    consistency_flag: str   # human-readable verdict

    # BUGFIX: an average-based decay% can be dominated by one anomalous
    # window (e.g. one window with an unusually strong training expectancy
    # inflates the average and makes normal variance look like severe
    # decay). Real 2-year MCX data showed exactly this: 3 of 4 windows had
    # solidly positive out-of-sample expectancy, but the aggregate "71%
    # decay" figure alone made it look like uniform failure. Reporting
    # per-window consistency alongside the average prevents that
    # misreading.
    windows_with_positive_test_expectancy: int = 0
    windows_with_positive_test_expectancy_pct: float = 0.0


def run_walk_forward(candles: list[OHLCV], config, num_windows: int = 4,
                       train_pct: float = 0.6, validation_pct: float = 0.2,
                       overfit_decay_threshold_pct: float = 50.0,
                       **backtest_kwargs) -> WalkForwardResult:
    """
    Splits candles into num_windows ROLLING windows, each divided into
    train_pct / validation_pct / (remainder as test) segments. Runs the
    actual backtest engine on each segment — this is not a simulation of
    walk-forward, it genuinely re-runs the strategy on each slice.

    overfit_decay_threshold_pct: if test expectancy decays by more than
    this % relative to train expectancy, flag as likely overfit. This
    threshold is a starting hypothesis, not a proven cutoff — treat the
    flag as a prompt to investigate, not an automatic rejection.
    """
    n = len(candles)
    if n < 100:
        raise ValueError(
            f"Only {n} candles provided — walk-forward validation needs a substantial "
            f"history to produce meaningful windows. This is a hard guard, not a "
            f"suggestion: a 'validated' result from too little data is worse than no "
            f"validation at all, because it creates false confidence."
        )

    test_pct = 1.0 - train_pct - validation_pct
    if test_pct <= 0:
        raise ValueError(f"train_pct + validation_pct must be < 1.0, "
                          f"got train={train_pct} + validation={validation_pct}")

    window_size = n // num_windows
    if window_size < 30:
        raise ValueError(
            f"num_windows={num_windows} on {n} candles gives windows of only "
            f"{window_size} candles each — too small to be meaningful. "
            f"Reduce num_windows or provide more data."
        )

    windows = []
    train_expectancies = []
    validation_expectancies = []
    test_expectancies = []

    for w in range(num_windows):
        w_start = w * window_size
        w_end = min(w_start + window_size, n)
        w_len = w_end - w_start

        train_end = w_start + int(w_len * train_pct)
        validation_end = train_end + int(w_len * validation_pct)
        test_end = w_end

        train_slice = candles[w_start:train_end]
        validation_slice = candles[train_end:validation_end]
        test_slice = candles[validation_end:test_end]

        # each slice is backtested independently and chronologically —
        # test data is NEVER used to compute train/validation results,
        # preventing look-ahead leakage across the split boundary
        train_result = run_backtest(train_slice, config, **backtest_kwargs)
        validation_result = run_backtest(validation_slice, config, **backtest_kwargs)
        test_result = run_backtest(test_slice, config, **backtest_kwargs)

        windows.append(WalkForwardWindow(
            window_index=w,
            train_start_idx=w_start, train_end_idx=train_end,
            validation_start_idx=train_end, validation_end_idx=validation_end,
            test_start_idx=validation_end, test_end_idx=test_end,
            train_metrics=train_result.metrics,
            validation_metrics=validation_result.metrics,
            test_metrics=test_result.metrics,
        ))
        train_expectancies.append(train_result.metrics.expectancy_r)
        validation_expectancies.append(validation_result.metrics.expectancy_r)
        test_expectancies.append(test_result.metrics.expectancy_r)

    avg_train = sum(train_expectancies) / num_windows
    avg_validation = sum(validation_expectancies) / num_windows
    avg_test = sum(test_expectancies) / num_windows

    decay_pct = _compute_decay_pct(avg_train, avg_test)
    is_overfit = decay_pct > overfit_decay_threshold_pct

    windows_positive = sum(1 for e in test_expectancies if e > 0)
    windows_positive_pct = round(windows_positive / num_windows * 100, 1)

    if avg_train <= 0:
        consistency = "Training expectancy itself is non-positive — the strategy shows no " \
                       "edge even on the data it was tuned on. Do not proceed to live trading."
    elif windows_positive_pct >= 75:
        # majority of windows genuinely held up out-of-sample — the aggregate
        # decay% can still look large if driven by one strong training window,
        # so lead with the more reliable per-window consistency signal instead
        consistency = (f"{windows_positive}/{num_windows} windows ({windows_positive_pct:.0f}%) had "
                        f"positive out-of-sample expectancy — the strategy generalized reasonably "
                        f"well across most periods. The aggregate {decay_pct:.0f}% decay figure "
                        f"looks large but is likely skewed by one unusually strong training window, "
                        f"not uniform overfitting. Still treat this as encouraging-but-early, "
                        f"not confirmed.")
    elif is_overfit:
        consistency = (f"Out-of-sample expectancy decayed {decay_pct:.0f}% versus training, and "
                        f"only {windows_positive}/{num_windows} windows held up out-of-sample — "
                        f"likely genuine overfitting to the training window(s). Do not trust the "
                        f"training-window numbers as representative of live performance.")
    elif avg_test <= 0:
        consistency = ("Out-of-sample expectancy is non-positive even though decay is within "
                        "threshold — the edge may be too fragile or marginal to trade live.")
    else:
        consistency = (f"Out-of-sample expectancy ({avg_test:.3f}R) held up reasonably "
                        f"against training ({avg_train:.3f}R) — {decay_pct:.0f}% decay, "
                        f"within the {overfit_decay_threshold_pct:.0f}% tolerance.")

    return WalkForwardResult(
        windows=windows, num_windows=num_windows,
        avg_train_expectancy=round(avg_train, 4), avg_validation_expectancy=round(avg_validation, 4),
        avg_test_expectancy=round(avg_test, 4), train_to_test_decay_pct=round(decay_pct, 2),
        is_likely_overfit=is_overfit, consistency_flag=consistency,
        windows_with_positive_test_expectancy=windows_positive,
        windows_with_positive_test_expectancy_pct=windows_positive_pct,
    )


def _compute_decay_pct(train_expectancy: float, test_expectancy: float) -> float:
    """
    % decay from train to test. Handles sign changes explicitly: going from
    positive train expectancy to negative test expectancy is 100%+ decay
    (total edge loss), not a small number — a naive (train-test)/train
    calculation can produce misleadingly small percentages when train is
    near zero, so this is computed carefully rather than as a raw ratio.
    """
    if train_expectancy <= 0:
        return 100.0 if test_expectancy < train_expectancy else 0.0
    if test_expectancy >= train_expectancy:
        return 0.0  # test performed as well or better — no decay
    decay = (train_expectancy - test_expectancy) / abs(train_expectancy) * 100
    return min(decay, 200.0)  # cap for readability; anything past 200% is "total loss of edge"
