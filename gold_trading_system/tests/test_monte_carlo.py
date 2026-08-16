import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.monte_carlo import run_monte_carlo


def test_all_wins_zero_risk_of_ruin():
    """A strategy with 40 consistent +2R wins should NEVER show risk of ruin."""
    trades = [2.0] * 40
    result = run_monte_carlo(trades, num_simulations=500, seed=42)
    print(f"All-wins: risk_of_ruin={result.risk_of_ruin_pct}%, drawdown_p50={result.drawdown_p50}")
    assert result.risk_of_ruin_pct == 0.0
    assert result.drawdown_p50 == 0.0, "An always-winning sequence should never drawdown"


def test_all_losses_high_risk_of_ruin():
    """A strategy with 40 consistent -1R losses should show high/certain risk of ruin
    given a shallow ruin threshold."""
    trades = [-1.0] * 40
    result = run_monte_carlo(trades, num_simulations=500, ruin_threshold_r=-10.0, seed=42)
    print(f"All-losses: risk_of_ruin={result.risk_of_ruin_pct}%")
    assert result.risk_of_ruin_pct == 100.0, \
        "40 guaranteed -1R losses must certainly breach a -10R ruin threshold"


def test_low_confidence_flag_on_small_sample():
    trades = [1.0, -1.0, 2.0]  # only 3 trades
    result = run_monte_carlo(trades, num_simulations=200, min_trades_for_confidence=30, seed=1)
    assert result.is_low_confidence == True


def test_high_confidence_flag_on_large_sample():
    trades = [1.0, -1.0, 2.0, -0.5, 1.5] * 10  # 50 trades
    result = run_monte_carlo(trades, num_simulations=200, min_trades_for_confidence=30, seed=1)
    assert result.is_low_confidence == False


def test_empty_trades_no_crash():
    result = run_monte_carlo([], num_simulations=100)
    assert result.num_simulations == 0
    assert result.is_low_confidence == True
    assert result.risk_of_ruin_pct == 0.0


def test_drawdown_percentiles_ordered_correctly():
    """p50 <= p90 <= p99 <= worst_case must always hold, by definition of percentiles."""
    trades = [2.0, -1.0, 1.5, -2.0, 3.0, -1.5, 1.0, -3.0]
    result = run_monte_carlo(trades, num_simulations=1000, seed=7)
    print(f"Drawdown percentiles: p50={result.drawdown_p50}, p90={result.drawdown_p90}, "
          f"p99={result.drawdown_p99}, worst={result.max_drawdown_worst_case}")
    assert result.drawdown_p50 <= result.drawdown_p90
    assert result.drawdown_p90 <= result.drawdown_p99
    assert result.drawdown_p99 <= result.max_drawdown_worst_case


def test_final_equity_percentiles_ordered_correctly():
    trades = [2.0, -1.0, 1.5, -2.0, 3.0, -1.5, 1.0, -3.0]
    result = run_monte_carlo(trades, num_simulations=1000, seed=7)
    assert result.final_equity_r_p10 <= result.final_equity_r_p50
    assert result.final_equity_r_p50 <= result.final_equity_r_p90


def test_positive_expectancy_produces_positive_median_equity():
    """A genuinely positive-expectancy trade set should show a positive median final equity."""
    trades = [3.0, 3.0, -1.0, -1.0, -1.0]  # expectancy = (6-3)/5 = +0.6R per trade
    result = run_monte_carlo(trades, num_simulations=2000, seed=99)
    print(f"Positive expectancy median final equity: {result.final_equity_r_p50}")
    assert result.final_equity_r_p50 > 0, \
        "A positive-expectancy trade set should show positive median simulated equity"


def test_negative_expectancy_produces_negative_median_equity():
    trades = [1.0, -2.0, -2.0, 1.0, -2.0]  # expectancy = (1-2-2+1-2)/5 = -0.8R per trade
    result = run_monte_carlo(trades, num_simulations=2000, seed=99)
    print(f"Negative expectancy median final equity: {result.final_equity_r_p50}")
    assert result.final_equity_r_p50 < 0


def test_reproducible_with_same_seed():
    trades = [1.0, -1.0, 2.0, -0.5]
    result1 = run_monte_carlo(trades, num_simulations=500, seed=123)
    result2 = run_monte_carlo(trades, num_simulations=500, seed=123)
    assert result1.risk_of_ruin_pct == result2.risk_of_ruin_pct
    assert result1.drawdown_p50 == result2.drawdown_p50


def test_losing_streak_percentiles_are_integers_and_ordered():
    trades = [1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
    result = run_monte_carlo(trades, num_simulations=1000, seed=5)
    print(f"Losing streaks: p50={result.max_losing_streak_p50}, p90={result.max_losing_streak_p90}")
    assert isinstance(result.max_losing_streak_p50, int)
    assert result.max_losing_streak_p50 <= result.max_losing_streak_p90


if __name__ == "__main__":
    tests = [
        test_all_wins_zero_risk_of_ruin,
        test_all_losses_high_risk_of_ruin,
        test_low_confidence_flag_on_small_sample,
        test_high_confidence_flag_on_large_sample,
        test_empty_trades_no_crash,
        test_drawdown_percentiles_ordered_correctly,
        test_final_equity_percentiles_ordered_correctly,
        test_positive_expectancy_produces_positive_median_equity,
        test_negative_expectancy_produces_negative_median_equity,
        test_reproducible_with_same_seed,
        test_losing_streak_percentiles_are_integers_and_ordered,
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
