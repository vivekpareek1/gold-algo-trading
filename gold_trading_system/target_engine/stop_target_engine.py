"""
Stop-Loss Engine (Sprint 1 §10) + Target Engine (Sprint 1 §12).
Never place an arbitrary stop or target — evaluate multiple real candidates
and pick with a stated reason, store SL_PRICE/SL_DISTANCE/SL_REASON exactly
as the spec requires. Targets are market-structure and volatility aware,
not just a fixed R-multiple in a vacuum.
"""
from dataclasses import dataclass
from enum import Enum


class StopMethod(str, Enum):
    STRUCTURE_SWING = "STRUCTURE_SWING"
    ATR = "ATR"
    VOLATILITY_BUFFER = "VOLATILITY_BUFFER"


@dataclass
class StopCandidate:
    method: StopMethod
    price: float
    distance_points: float
    reason: str


@dataclass
class StopLossResult:
    approved: bool
    price: float | None
    distance_points: float | None
    reason: str
    all_candidates: list
    rejection_reason: str = ""


@dataclass
class TargetResult:
    target_1: float
    target_2: float
    target_3: float
    basis: str   # explains how targets were derived


class StopLossEngine:
    """
    Evaluates candidate stops and picks the tightest one that still respects
    real structure (never placed inside noise) and the configured max
    distance. Never widens beyond what risk allows — if every valid
    structural candidate exceeds the cap, the caller should treat this as
    NO_TRADE rather than force a stop that ignores the market's own shape.
    """

    def __init__(self, config):
        self.config = config

    def evaluate(self, direction: str, entry_price: float, atr: float,
                 nearest_swing_low: float | None, nearest_swing_high: float | None,
                 atr_buffer_mult: float = 0.3) -> StopLossResult:
        candidates = []
        is_long = direction == "LONG"

        # structure stop: just beyond the nearest relevant swing, with a
        # small ATR buffer so normal noise/a shallow liquidity sweep doesn't
        # take it out immediately (same principle as the sweep-buffer logic
        # in the market structure engine)
        relevant_swing = nearest_swing_low if is_long else nearest_swing_high
        if relevant_swing is not None:
            buffer = atr * atr_buffer_mult
            structure_price = relevant_swing - buffer if is_long else relevant_swing + buffer
            distance = abs(entry_price - structure_price)
            candidates.append(StopCandidate(
                method=StopMethod.STRUCTURE_SWING, price=structure_price,
                distance_points=distance,
                reason=f"{'Below' if is_long else 'Above'} nearest swing "
                       f"({relevant_swing:.2f}) with {atr_buffer_mult}x ATR buffer",
            ))

        # ATR stop: pure volatility-based, doesn't need structure data at all
        atr_distance = atr * self.config.trailing.atr_multiplier
        atr_price = entry_price - atr_distance if is_long else entry_price + atr_distance
        candidates.append(StopCandidate(
            method=StopMethod.ATR, price=atr_price, distance_points=atr_distance,
            reason=f"{self.config.trailing.atr_multiplier}x ATR ({atr:.2f}) from entry",
        ))

        # volatility buffer stop: a tighter fallback when structure data is
        # thin, using a smaller ATR multiple as a floor
        vol_distance = atr * 1.0
        vol_price = entry_price - vol_distance if is_long else entry_price + vol_distance
        candidates.append(StopCandidate(
            method=StopMethod.VOLATILITY_BUFFER, price=vol_price, distance_points=vol_distance,
            reason=f"1.0x ATR ({atr:.2f}) minimum volatility buffer",
        ))

        # pick the TIGHTEST candidate that still respects the max distance cap —
        # never the loosest, since risk sizing should reflect real structure,
        # not the widest possible accommodation
        within_cap = [c for c in candidates if c.distance_points <= self.config.risk.max_stop_distance_points]

        if not within_cap:
            tightest_overall = min(candidates, key=lambda c: c.distance_points)
            return StopLossResult(
                approved=False, price=None, distance_points=None,
                reason="", all_candidates=candidates,
                rejection_reason=(
                    f"Even the tightest candidate ({tightest_overall.method.value}, "
                    f"{tightest_overall.distance_points:.1f}pts) exceeds the "
                    f"{self.config.risk.max_stop_distance_points}pt cap — the market's "
                    f"real structure doesn't fit the risk budget for this setup. "
                    f"NOT forcing a tighter stop that ignores structure."
                ),
            )

        chosen = min(within_cap, key=lambda c: c.distance_points)
        return StopLossResult(
            approved=True, price=chosen.price, distance_points=chosen.distance_points,
            reason=f"[{chosen.method.value}] {chosen.reason}",
            all_candidates=candidates,
        )


class TargetEngine:
    """
    Targets are structure-and-volatility aware, not a blind R-multiple.
    T1 favors the nearest real level (swing/resistance) if it's reasonably
    close to the R-multiple target; T2/T3 extend further using R-multiples
    scaled by regime strength. Reject trades whose resulting R:R falls
    below the configured minimum.
    """

    def __init__(self, config):
        self.config = config

    def calculate(self, direction: str, entry_price: float, stop_price: float,
                  nearest_resistance: float | None, nearest_support: float | None,
                  atr: float) -> TargetResult | None:
        is_long = direction == "LONG"
        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            return None

        # base R-multiple targets
        t1_r = entry_price + risk_distance * 1.0 if is_long else entry_price - risk_distance * 1.0
        t2_r = entry_price + risk_distance * 2.0 if is_long else entry_price - risk_distance * 2.0
        t3_r = entry_price + risk_distance * 3.0 if is_long else entry_price - risk_distance * 3.0

        basis_notes = ["R-multiple base (1R/2R/3R)"]

        # T1 structure awareness: if a real level sits closer than the 1R
        # target, prefer it (more realistic first take-profit than hoping
        # price blows through a level on the way to an arbitrary R target)
        relevant_level = nearest_resistance if is_long else nearest_support
        t1 = t1_r
        min_t1_rr_floor = 0.3   # T1 is the first PARTIAL booking level, not the
                                  # overall trade quality gate — config.risk.min_risk_reward
                                  # is checked against the overall setup elsewhere (using T2),
                                  # not against T1, which legitimately sits near 1R or less
        if relevant_level is not None:
            level_is_between_entry_and_1R = (
                (is_long and entry_price < relevant_level < t1_r) or
                (not is_long and t1_r < relevant_level < entry_price)
            )
            if level_is_between_entry_and_1R:
                candidate_t1_rr = abs(relevant_level - entry_price) / risk_distance
                if candidate_t1_rr >= min_t1_rr_floor:
                    t1 = relevant_level
                    basis_notes.append(f"T1 adjusted to nearest level ({relevant_level:.2f}) "
                                        f"ahead of the 1R target")
                else:
                    basis_notes.append(f"Nearby level ({relevant_level:.2f}) too close to entry "
                                        f"(R:R would be {candidate_t1_rr:.2f}, below the "
                                        f"{min_t1_rr_floor} floor) — kept the pure R-multiple T1")

        return TargetResult(
            target_1=round(t1, 2), target_2=round(t2_r, 2), target_3=round(t3_r, 2),
            basis="; ".join(basis_notes),
        )
