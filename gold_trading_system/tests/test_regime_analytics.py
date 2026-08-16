import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.regime_analytics import analyze_by_regime


def make_trades(regime, r_list):
    return [{"r_multiple": r, "entry_regime": regime} for r in r_list]


def test_basic_segmentation_by_regime():
    trades = (make_trades("TRENDING_UP", [1.0, 2.0, -1.0]) +
              make_trades("RANGE", [-1.0, -0.5]))
    breakdown = analyze_by_regime(trades, min_trades_for_confidence=1)
    assert "TRENDING_UP" in breakdown.per_regime_metrics
    assert "RANGE" in breakdown.per_regime_metrics
    assert breakdown.per_regime_metrics["TRENDING_UP"].total_trades == 3
    assert breakdown.per_regime_metrics["RANGE"].total_trades == 2


def test_best_and_worst_regime_identified():
    trades = (make_trades("TRENDING_UP", [2.0, 2.0, 2.0]) +
              make_trades("RANGE", [-1.0, -1.0, -1.0]))
    breakdown = analyze_by_regime(trades, min_trades_for_confidence=1)
    assert breakdown.best_regime == "TRENDING_UP"
    assert breakdown.worst_regime == "RANGE"


def test_low_confidence_regime_excluded_from_best_worst():
    """A regime with too few trades shouldn't be crowned 'best' just from luck."""
    trades = (make_trades("TRENDING_UP", [1.0, -0.5, 0.3, -0.2, 0.5] * 3) +  # 15 trades, modest edge
              make_trades("RANGE", [10.0]))  # 1 lucky trade, huge R but tiny sample
    breakdown = analyze_by_regime(trades, min_trades_for_confidence=10)
    assert breakdown.best_regime == "TRENDING_UP", \
        "A single lucky trade in a thin regime should not be crowned best"


def test_concentration_warning_fires_when_one_regime_carries_everything():
    trades = (make_trades("TRENDING_UP", [2.0] * 15) +          # strongly positive, real edge
              make_trades("RANGE", [-1.0] * 12) +                 # negative
              make_trades("HIGH_VOLATILITY", [-0.5] * 11))        # also negative
    breakdown = analyze_by_regime(trades, min_trades_for_confidence=10)
    print(f"Warning: {breakdown.regime_consistency_warning}")
    assert breakdown.regime_consistency_warning is not None
    assert "TRENDING_UP" in breakdown.regime_consistency_warning


def test_no_warning_when_multiple_regimes_positive():
    trades = (make_trades("TRENDING_UP", [1.0] * 15) +
              make_trades("RANGE", [0.5] * 12))
    breakdown = analyze_by_regime(trades, min_trades_for_confidence=10)
    assert breakdown.regime_consistency_warning is None


def test_missing_regime_tag_grouped_as_unknown():
    trades = [{"r_multiple": 1.0, "entry_regime": None},
              {"r_multiple": -0.5}]  # key entirely missing
    breakdown = analyze_by_regime(trades, min_trades_for_confidence=1)
    assert "UNKNOWN" in breakdown.per_regime_metrics
    assert breakdown.per_regime_metrics["UNKNOWN"].total_trades == 2


def test_empty_trade_log_no_crash():
    breakdown = analyze_by_regime([])
    assert breakdown.per_regime_metrics == {}
    assert breakdown.best_regime is None
    assert breakdown.worst_regime is None
    assert breakdown.regime_consistency_warning is None


def test_single_regime_no_comparison_possible():
    """With only one regime represented, there's nothing to compare — no warning, no crash."""
    trades = make_trades("TRENDING_UP", [1.0, -1.0, 2.0] * 5)
    breakdown = analyze_by_regime(trades, min_trades_for_confidence=10)
    assert breakdown.best_regime == "TRENDING_UP"
    assert breakdown.worst_regime == "TRENDING_UP"
    assert breakdown.regime_consistency_warning is None


if __name__ == "__main__":
    tests = [
        test_basic_segmentation_by_regime,
        test_best_and_worst_regime_identified,
        test_low_confidence_regime_excluded_from_best_worst,
        test_concentration_warning_fires_when_one_regime_carries_everything,
        test_no_warning_when_multiple_regimes_positive,
        test_missing_regime_tag_grouped_as_unknown,
        test_empty_trade_log_no_crash,
        test_single_regime_no_comparison_possible,
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
