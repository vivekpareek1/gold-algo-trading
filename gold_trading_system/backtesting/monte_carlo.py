"""
Monte Carlo Analysis (Sprint 1 §30). Bootstrap-resamples the ACTUAL trade
R-multiple sequence (not a parametric assumption like "normal distribution")
thousands of times to estimate: drawdown distribution, risk of ruin,
expected equity variation, and losing-streak probability.

Bootstrap resampling (drawing WITH replacement from real trades) is used
instead of assuming a theoretical distribution because real trade returns
are rarely normally distributed — this respects the actual shape of
whatever edge (or lack of it) the strategy has shown so far.

Requires a real trade history to be meaningful — running this on too few
trades produces a wide, low-confidence distribution, which the result
explicitly flags rather than presenting false precision.
"""
import random
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    num_simulations: int
    trades_per_simulation: int
    starting_r_multiples_count: int
    is_low_confidence: bool   # True if input trade count is too small to trust

    drawdown_p50: float
    drawdown_p90: float
    drawdown_p99: float
    max_drawdown_worst_case: float

    risk_of_ruin_pct: float   # % of simulations where cumulative R dropped below ruin_threshold_r

    final_equity_r_p10: float
    final_equity_r_p50: float
    final_equity_r_p90: float

    max_losing_streak_p50: int
    max_losing_streak_p90: int


def run_monte_carlo(trade_r_multiples: list[float], num_simulations: int = 2000,
                      trades_per_simulation: int | None = None,
                      ruin_threshold_r: float = -20.0,
                      min_trades_for_confidence: int = 30,
                      seed: int | None = None) -> MonteCarloResult:
    """
    trade_r_multiples: the REAL historical R-multiples from closed trades
    (from backtest_runner or live paper trading). This is bootstrap-resampled
    with replacement — never fabricated or assumed.

    ruin_threshold_r: cumulative R-multiple drawdown considered "ruin" —
    e.g. -20R would mean losing 20x your per-trade risk unit cumulatively.
    This should be set relative to what the account could actually survive.
    """
    if seed is not None:
        random.seed(seed)

    n = len(trade_r_multiples)
    if n == 0:
        return MonteCarloResult(
            num_simulations=0, trades_per_simulation=0, starting_r_multiples_count=0,
            is_low_confidence=True, drawdown_p50=0, drawdown_p90=0, drawdown_p99=0,
            max_drawdown_worst_case=0, risk_of_ruin_pct=0.0,
            final_equity_r_p10=0, final_equity_r_p50=0, final_equity_r_p90=0,
            max_losing_streak_p50=0, max_losing_streak_p90=0,
        )

    trades_per_sim = trades_per_simulation or n
    is_low_confidence = n < min_trades_for_confidence

    drawdowns = []
    final_equities = []
    losing_streaks = []
    ruin_count = 0

    for _ in range(num_simulations):
        sample = [random.choice(trade_r_multiples) for _ in range(trades_per_sim)]

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        cur_loss_streak = 0
        max_loss_streak = 0
        hit_ruin = False

        for r in sample:
            cumulative += r
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_dd = max(max_dd, drawdown)

            if r <= 0:
                cur_loss_streak += 1
            else:
                cur_loss_streak = 0
            max_loss_streak = max(max_loss_streak, cur_loss_streak)

            if cumulative <= ruin_threshold_r:
                hit_ruin = True

        drawdowns.append(max_dd)
        final_equities.append(cumulative)
        losing_streaks.append(max_loss_streak)
        if hit_ruin:
            ruin_count += 1

    drawdowns.sort()
    final_equities.sort()
    losing_streaks.sort()

    return MonteCarloResult(
        num_simulations=num_simulations, trades_per_simulation=trades_per_sim,
        starting_r_multiples_count=n, is_low_confidence=is_low_confidence,
        drawdown_p50=_percentile(drawdowns, 50), drawdown_p90=_percentile(drawdowns, 90),
        drawdown_p99=_percentile(drawdowns, 99), max_drawdown_worst_case=drawdowns[-1],
        risk_of_ruin_pct=round(ruin_count / num_simulations * 100, 2),
        final_equity_r_p10=_percentile(final_equities, 10),
        final_equity_r_p50=_percentile(final_equities, 50),
        final_equity_r_p90=_percentile(final_equities, 90),
        max_losing_streak_p50=int(_percentile(losing_streaks, 50)),
        max_losing_streak_p90=int(_percentile(losing_streaks, 90)),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct / 100))
    return round(sorted_values[idx], 3)
