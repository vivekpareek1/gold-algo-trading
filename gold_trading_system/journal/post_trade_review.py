"""
AI Post-Trade Review (Sprint 1 §36).
Classifies every closed trade as GOOD_WIN / BAD_WIN / GOOD_LOSS / BAD_LOSS
based on PROCESS quality — did the trade follow the rules — not just P&L.

"A profitable trade can still be a bad trade if it broke strategy rules.
A losing trade can still be a good trade if it followed a valid tested setup."

This module NEVER touches live strategy parameters. It only labels history.
Changes to strategy config require backtesting + explicit approval elsewhere
(StrategyVersion.is_approved) — this engine has no authority to trigger that.
"""
from dataclasses import dataclass, field
from enum import Enum


class TradeGrade(str, Enum):
    GOOD_WIN = "GOOD_WIN"
    BAD_WIN = "BAD_WIN"
    GOOD_LOSS = "GOOD_LOSS"
    BAD_LOSS = "BAD_LOSS"
    UNGRADED = "UNGRADED"   # e.g. breakeven trade with no clear P&L direction


@dataclass
class ProcessChecklist:
    """Each field is a fact about HOW the trade was taken/managed, not the outcome."""
    entry_confluence_score: int          # the long/short score AT entry
    min_required_score: int              # threshold that was in effect at entry
    position_size_within_risk_limit: bool
    stop_loss_within_max_distance: bool
    stop_was_ever_widened: bool          # should ALWAYS be False if trade_manager worked correctly
    exited_per_rules: bool               # True if exit was stop/target/momentum-decay/structure-break;
                                          # False if manually overridden outside the system's logic
    risk_reward_met_minimum: bool
    traded_during_blocked_event: bool = False  # True = violated news/event risk rules


@dataclass
class TradeReviewResult:
    grade: TradeGrade
    pnl_r: float
    process_violations: list
    process_score: int   # 0-100, how many checklist items passed
    explanation: str


class PostTradeReviewEngine:
    def review(self, pnl_r: float, checklist: ProcessChecklist) -> TradeReviewResult:
        violations = self._find_violations(checklist)
        process_score = self._process_score(checklist)
        is_win = pnl_r > 0
        is_clean_loss_or_breakeven = pnl_r <= 0

        if pnl_r == 0:
            grade = TradeGrade.UNGRADED
            explanation = "Breakeven trade — no clear win/loss outcome to grade against process."
        elif is_win and not violations:
            grade = TradeGrade.GOOD_WIN
            explanation = f"Profitable (+{pnl_r:.2f}R) AND followed all process rules — the target outcome."
        elif is_win and violations:
            grade = TradeGrade.BAD_WIN
            explanation = (
                f"Profitable (+{pnl_r:.2f}R) but violated process: {'; '.join(violations)}. "
                f"A good outcome from a bad process — do not let this reinforce the violation."
            )
        elif not is_win and not violations:
            grade = TradeGrade.GOOD_LOSS
            explanation = (
                f"Loss ({pnl_r:.2f}R) but followed a valid, rule-compliant setup — "
                f"this is an acceptable cost of a sound process, not a mistake to fix."
            )
        else:
            grade = TradeGrade.BAD_LOSS
            explanation = (
                f"Loss ({pnl_r:.2f}R) AND violated process: {'; '.join(violations)}. "
                f"Both the outcome and the process need correction."
            )

        return TradeReviewResult(
            grade=grade, pnl_r=pnl_r, process_violations=violations,
            process_score=process_score, explanation=explanation,
        )

    def _find_violations(self, c: ProcessChecklist) -> list:
        violations = []
        if c.entry_confluence_score < c.min_required_score:
            violations.append(
                f"Entered below the confluence threshold in effect "
                f"({c.entry_confluence_score} < {c.min_required_score})"
            )
        if not c.position_size_within_risk_limit:
            violations.append("Position size exceeded the configured risk limit")
        if not c.stop_loss_within_max_distance:
            violations.append("Stop-loss distance exceeded the configured maximum")
        if c.stop_was_ever_widened:
            violations.append("Stop-loss was widened at some point — violates the never-widen rule")
        if not c.exited_per_rules:
            violations.append("Exit was manually overridden rather than following system rules")
        if not c.risk_reward_met_minimum:
            violations.append("Risk:Reward was below the configured minimum at entry")
        if c.traded_during_blocked_event:
            violations.append("Trade was taken during a period that should have been blocked "
                               "by the news/event risk engine")
        return violations

    def _process_score(self, c: ProcessChecklist) -> int:
        checks = [
            c.entry_confluence_score >= c.min_required_score,
            c.position_size_within_risk_limit,
            c.stop_loss_within_max_distance,
            not c.stop_was_ever_widened,
            c.exited_per_rules,
            c.risk_reward_met_minimum,
            not c.traded_during_blocked_event,
        ]
        return round(sum(checks) / len(checks) * 100)


@dataclass
class JournalSummary:
    total_reviewed: int
    good_win_count: int
    bad_win_count: int
    good_loss_count: int
    bad_loss_count: int
    ungraded_count: int
    avg_process_score: float

    @property
    def process_discipline_pct(self) -> float:
        """% of trades with zero process violations, regardless of P&L outcome —
        the real measure of discipline, distinct from win rate."""
        graded = self.total_reviewed - self.ungraded_count
        if graded == 0:
            return 0.0
        disciplined = self.good_win_count + self.good_loss_count
        return round(disciplined / graded * 100, 1)


def summarize_reviews(results: list[TradeReviewResult]) -> JournalSummary:
    counts = {g: 0 for g in TradeGrade}
    for r in results:
        counts[r.grade] += 1
    n = len(results)
    avg_score = sum(r.process_score for r in results) / n if n > 0 else 0.0

    return JournalSummary(
        total_reviewed=n,
        good_win_count=counts[TradeGrade.GOOD_WIN],
        bad_win_count=counts[TradeGrade.BAD_WIN],
        good_loss_count=counts[TradeGrade.GOOD_LOSS],
        bad_loss_count=counts[TradeGrade.BAD_LOSS],
        ungraded_count=counts[TradeGrade.UNGRADED],
        avg_process_score=round(avg_score, 1),
    )
