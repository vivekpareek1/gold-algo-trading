"""
Day-of-Week Situational Analysis (Tom Hougaard methodology).
Non-indicator-based: studies how price statistically behaves on specific
days of the week, using pattern recurrence rather than real-time technicals.

This is a MODIFIER layer feeding into Situation Analysis — same principle
as macro_bias: it adjusts conviction, it never triggers a trade on its own.
Patterns must be DISCOVERED from real historical data (Sprint 1 caveat:
insufficient sample size = explicitly flagged unreliable, never guessed).
"""
from dataclasses import dataclass
from enum import IntEnum


class Weekday(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class DailyCandle:
    ts_epoch_day: int    # days since epoch, used to derive weekday deterministically
    weekday: Weekday
    open: float
    high: float
    low: float
    close: float


@dataclass
class WeekdayStats:
    weekday: Weekday
    sample_size: int
    avg_return_pct: float          # mean (close-open)/open * 100
    win_rate_pct: float            # % of days that closed green
    avg_range_pct: float           # mean (high-low)/open * 100 — typical volatility for this day
    avg_gap_fill_rate_pct: float | None = None  # % of days that filled the prior day's gap
    is_reliable: bool = False       # False if sample_size below the configured minimum


@dataclass
class DayOfWeekBias:
    weekday: Weekday
    bias_score: float               # -100..+100, bullish positive; 0 if unreliable
    is_reliable: bool
    reasoning: str


class DayOfWeekAnalyzer:
    """
    Discovers statistically recurring weekday patterns from real historical
    candles. Never assumes a pattern — everything here is computed from
    the data actually passed in.
    """

    def __init__(self, min_sample_size: int = 20, min_win_rate_edge_pct: float = 8.0):
        # min_sample_size: below this many historical occurrences of a
        # weekday, treat any apparent edge as noise, not a real pattern.
        # min_win_rate_edge_pct: how far from 50% win rate counts as a
        # genuine statistical tilt worth using as a modifier.
        self.min_sample_size = min_sample_size
        self.min_win_rate_edge_pct = min_win_rate_edge_pct

    def compute_weekday_stats(self, daily_candles: list[DailyCandle]) -> dict[Weekday, WeekdayStats]:
        buckets: dict[Weekday, list[DailyCandle]] = {wd: [] for wd in Weekday}
        for c in daily_candles:
            buckets[c.weekday].append(c)

        results = {}
        for wd, candles in buckets.items():
            n = len(candles)
            if n == 0:
                results[wd] = WeekdayStats(
                    weekday=wd, sample_size=0, avg_return_pct=0.0,
                    win_rate_pct=0.0, avg_range_pct=0.0, is_reliable=False,
                )
                continue

            returns = [(c.close - c.open) / c.open * 100 for c in candles if c.open != 0]
            ranges = [(c.high - c.low) / c.open * 100 for c in candles if c.open != 0]
            wins = sum(1 for c in candles if c.close > c.open)

            avg_return = sum(returns) / len(returns) if returns else 0.0
            avg_range = sum(ranges) / len(ranges) if ranges else 0.0
            win_rate = (wins / n) * 100

            results[wd] = WeekdayStats(
                weekday=wd, sample_size=n, avg_return_pct=avg_return,
                win_rate_pct=win_rate, avg_range_pct=avg_range,
                is_reliable=(n >= self.min_sample_size),
            )
        return results

    def get_bias(self, weekday: Weekday, stats: dict[Weekday, WeekdayStats]) -> DayOfWeekBias:
        s = stats.get(weekday)
        if s is None or not s.is_reliable:
            return DayOfWeekBias(
                weekday=weekday, bias_score=0.0, is_reliable=False,
                reasoning=f"Insufficient sample size ({s.sample_size if s else 0} occurrences, "
                          f"need >= {self.min_sample_size}) — no statistically meaningful "
                          f"pattern established for this weekday.",
            )

        edge = s.win_rate_pct - 50.0
        if abs(edge) < self.min_win_rate_edge_pct:
            return DayOfWeekBias(
                weekday=weekday, bias_score=0.0, is_reliable=True,
                reasoning=f"Win rate {s.win_rate_pct:.1f}% is within noise range of 50% "
                          f"(n={s.sample_size}) — no meaningful directional tilt on this weekday.",
            )

        # scale the edge into a -100..+100 bias score, capped
        bias_score = max(-100.0, min(100.0, edge * 4))  # 12.5% edge -> +50 bias, etc.

        direction = "bullish" if bias_score > 0 else "bearish"
        return DayOfWeekBias(
            weekday=weekday, bias_score=bias_score, is_reliable=True,
            reasoning=f"{weekday.name.title()} has historically closed green {s.win_rate_pct:.1f}% "
                      f"of the time (n={s.sample_size}, avg return {s.avg_return_pct:+.3f}%) — "
                      f"a statistically meaningful {direction} tilt.",
        )
