"""
Backtest metrics (Sprint 1 §27). Computed from a list of closed trade PnLs
(in R-multiples, for strategy-comparability across position sizes).
Never optimize for win rate alone — expectancy and profit factor matter more.
"""
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    total_trades: int
    win_rate: float
    loss_rate: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float
    profit_factor: float
    max_drawdown_r: float
    avg_r: float
    best_trade_r: float
    worst_trade_r: float
    max_consecutive_wins: int
    max_consecutive_losses: int


def compute_metrics(trade_r_multiples: list[float]) -> BacktestMetrics:
    """trade_r_multiples: list of realized R-multiples, one per closed trade
    (positive = win, negative = loss), in chronological order."""
    n = len(trade_r_multiples)
    if n == 0:
        return BacktestMetrics(
            total_trades=0, win_rate=0.0, loss_rate=0.0, avg_win_r=0.0, avg_loss_r=0.0,
            expectancy_r=0.0, profit_factor=0.0, max_drawdown_r=0.0, avg_r=0.0,
            best_trade_r=0.0, worst_trade_r=0.0,
            max_consecutive_wins=0, max_consecutive_losses=0,
        )

    wins = [r for r in trade_r_multiples if r > 0]
    losses = [r for r in trade_r_multiples if r <= 0]

    win_rate = len(wins) / n * 100
    loss_rate = len(losses) / n * 100
    avg_win_r = sum(wins) / len(wins) if wins else 0.0
    avg_loss_r = sum(losses) / len(losses) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    expectancy_r = sum(trade_r_multiples) / n
    avg_r = expectancy_r  # same thing, kept as separate field per schema naming

    max_drawdown_r = _max_drawdown(trade_r_multiples)

    max_cw, max_cl = _max_streaks(trade_r_multiples)

    return BacktestMetrics(
        total_trades=n, win_rate=win_rate, loss_rate=loss_rate,
        avg_win_r=avg_win_r, avg_loss_r=avg_loss_r,
        expectancy_r=expectancy_r, profit_factor=profit_factor,
        max_drawdown_r=max_drawdown_r, avg_r=avg_r,
        best_trade_r=max(trade_r_multiples), worst_trade_r=min(trade_r_multiples),
        max_consecutive_wins=max_cw, max_consecutive_losses=max_cl,
    )


def _max_drawdown(trade_r_multiples: list[float]) -> float:
    """Max peak-to-trough drawdown of the cumulative R equity curve."""
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in trade_r_multiples:
        cumulative += r
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_dd = max(max_dd, drawdown)
    return max_dd


def _max_streaks(trade_r_multiples: list[float]) -> tuple[int, int]:
    max_win_streak = cur_win_streak = 0
    max_loss_streak = cur_loss_streak = 0
    for r in trade_r_multiples:
        if r > 0:
            cur_win_streak += 1
            cur_loss_streak = 0
        else:
            cur_loss_streak += 1
            cur_win_streak = 0
        max_win_streak = max(max_win_streak, cur_win_streak)
        max_loss_streak = max(max_loss_streak, cur_loss_streak)
    return max_win_streak, max_loss_streak
