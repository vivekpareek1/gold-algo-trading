import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.metrics import compute_metrics


def test_hand_calculated_simple_series():
    """3 wins of +2R, 2 losses of -1R. Hand-calculate every field."""
    trades = [2.0, 2.0, 2.0, -1.0, -1.0]
    m = compute_metrics(trades)
    print(f"Metrics: {m}")

    assert m.total_trades == 5
    assert m.win_rate == 60.0
    assert m.loss_rate == 40.0
    assert m.avg_win_r == 2.0
    assert m.avg_loss_r == -1.0
    # expectancy = (2+2+2-1-1)/5 = 4/5 = 0.8
    assert abs(m.expectancy_r - 0.8) < 0.0001
    # profit_factor = gross_profit(6) / gross_loss(2) = 3.0
    assert abs(m.profit_factor - 3.0) < 0.0001
    assert m.best_trade_r == 2.0
    assert m.worst_trade_r == -1.0


def test_max_drawdown_hand_calculated():
    """Equity curve: 0 -> 2 -> 4 -> 6 -> 5 -> 3 -> 1 -> 4. Peak=6, trough after=1. DD=5."""
    trades = [2, 2, 2, -1, -2, -2, 3]  # cumulative: 2,4,6,5,3,1,4
    m = compute_metrics(trades)
    print(f"Drawdown test - max_drawdown_r: {m.max_drawdown_r}")
    assert abs(m.max_drawdown_r - 5.0) < 0.0001, f"Expected max DD of 5.0, got {m.max_drawdown_r}"


def test_consecutive_streaks():
    trades = [1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1]
    m = compute_metrics(trades)
    print(f"Max win streak: {m.max_consecutive_wins}, max loss streak: {m.max_consecutive_losses}")
    assert m.max_consecutive_wins == 3
    assert m.max_consecutive_losses == 4


def test_zero_trades_no_crash():
    m = compute_metrics([])
    assert m.total_trades == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0


def test_all_wins_infinite_profit_factor():
    m = compute_metrics([1.0, 2.0, 1.5])
    assert m.profit_factor == float("inf")
    assert m.win_rate == 100.0


def test_all_losses():
    m = compute_metrics([-1.0, -0.5, -2.0])
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.expectancy_r < 0


def test_zero_r_trade_counts_as_loss_bucket():
    """A breakeven trade (r=0) should not be double-counted or crash the win/loss split."""
    trades = [1.0, 0.0, -1.0]
    m = compute_metrics(trades)
    assert m.total_trades == 3
    assert abs((m.win_rate + m.loss_rate) - 100.0) < 0.0001, \
        f"win_rate + loss_rate should sum to 100%, got {m.win_rate + m.loss_rate}"


if __name__ == "__main__":
    tests = [
        test_hand_calculated_simple_series,
        test_max_drawdown_hand_calculated,
        test_consecutive_streaks,
        test_zero_trades_no_crash,
        test_all_wins_infinite_profit_factor,
        test_all_losses,
        test_zero_r_trade_counts_as_loss_bucket,
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
