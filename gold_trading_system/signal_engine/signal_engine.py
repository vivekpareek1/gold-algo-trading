"""
Signal / Confluence Engine — ties structure, indicators, situation analysis,
and gold intelligence into ONE weighted score per direction, then a decision.

Design rules enforced here (from Sprint 1 + all discussion since):
- Weights are config, not hardcoded truths (config/settings.py).
- macro_bias and fair_value_deviation are MODIFIERS, never standalone triggers.
- A liquidity sweep + FVG fill + supporting volume is treated as stacked
  high-confluence, not just one factor among many.
- Risk engine has FINAL veto — this engine only recommends.
- NO_TRADE is a valid, expected, successful outcome.
"""
from dataclasses import dataclass
from enum import Enum

from market_structure.structure_engine import StructureState, StructureEvent, TrendState
from situation_analysis.situation_analyzer import SituationSnapshot, MarketRegime
from gold_intelligence.fair_value import FairValueResult, MacroBiasResult, MoveClassification


class Decision(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class TradeType(str, Enum):
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    MOMENTUM = "MOMENTUM"
    REVERSAL = "REVERSAL"
    RANGE = "RANGE"


@dataclass
class ConfluenceInputs:
    ltf_structure: StructureState
    situation: SituationSnapshot
    fair_value: FairValueResult
    macro: MacroBiasResult

    # indicator-derived booleans/values needed for scoring
    ema_aligned_bullish: bool
    ema_aligned_bearish: bool
    macd_bullish: bool
    macd_bearish: bool
    rsi: float
    price_above_vwap: bool
    volume_supportive: bool
    bb_squeeze: bool          # low volatility, potential breakout setup
    # BUGFIX: this used to be read from macro.components, where it never exists
    # (MacroBiasResult only carries real_yield/dxy/usdinr/crude). It lives on
    # MacroContext, so it must be passed in explicitly by the caller.
    session_quality_ok: bool = True


@dataclass
class ConfluenceResult:
    long_score: int
    short_score: int
    decision: Decision
    confidence: int
    trade_type: TradeType | None
    reasons_for: list
    reasons_against: list
    stacked_confluence: bool   # True if sweep+FVG+volume all align
    invalidation_condition: str


class SignalEngine:
    def __init__(self, config):
        self.config = config

    def evaluate(self, inputs: ConfluenceInputs) -> ConfluenceResult:
        long_components = self._score_direction(inputs, direction="LONG")
        short_components = self._score_direction(inputs, direction="SHORT")

        long_score = self._weighted_total(long_components)
        short_score = self._weighted_total(short_components)

        # gold-specific overlay — modifiers only, applied AFTER the base score,
        # never used to create a signal on their own
        long_score = self._apply_gold_overlay(long_score, inputs, direction="LONG")
        short_score = self._apply_gold_overlay(short_score, inputs, direction="SHORT")

        long_score = max(0, min(100, long_score))
        short_score = max(0, min(100, short_score))

        stacked = self._is_stacked_confluence(inputs)

        decision, confidence, trade_type = self._decide(
            long_score, short_score, inputs, stacked
        )

        reasons_for, reasons_against = self._build_reasons(
            inputs, decision, long_components if decision == Decision.BUY else short_components
        )

        invalidation = self._invalidation_condition(decision, inputs)

        return ConfluenceResult(
            long_score=long_score, short_score=short_score,
            decision=decision, confidence=confidence, trade_type=trade_type,
            reasons_for=reasons_for, reasons_against=reasons_against,
            stacked_confluence=stacked, invalidation_condition=invalidation,
        )

    # ---------- per-direction component scoring (each 0-100, weighted later) ----------
    def _score_direction(self, inp: ConfluenceInputs, direction: str) -> dict:
        is_long = direction == "LONG"
        c = {}

        # market_structure: trend direction match + recent bullish/bearish event
        trend_match = (
            inp.ltf_structure.trend == TrendState.TRENDING_UP if is_long
            else inp.ltf_structure.trend == TrendState.TRENDING_DOWN
        )
        event_supports = (
            inp.ltf_structure.last_event in (StructureEvent.CHOCH_BULLISH, StructureEvent.LIQUIDITY_SWEEP_LOW) if is_long
            else inp.ltf_structure.last_event in (StructureEvent.CHOCH_BEARISH, StructureEvent.LIQUIDITY_SWEEP_HIGH)
        )
        c["market_structure"] = 100 if (trend_match or event_supports) else (50 if inp.ltf_structure.trend == TrendState.RANGE else 0)

        # htf_trend_alignment: pulls from situation snapshot's alignment score directly
        c["htf_trend_alignment"] = inp.situation.trend_alignment_score

        # volume_oi
        c["volume_oi"] = 100 if inp.volume_supportive else 40

        # momentum
        momentum_map = {"STRONG": 100, "WEAKENING": 50, "DEAD": 10}
        c["momentum"] = momentum_map.get(inp.situation.momentum_health, 30)

        # ema_alignment
        c["ema_alignment"] = 100 if (inp.ema_aligned_bullish if is_long else inp.ema_aligned_bearish) else 20

        # vwap
        c["vwap"] = 100 if (inp.price_above_vwap == is_long) else 20

        # macd
        c["macd"] = 100 if (inp.macd_bullish if is_long else inp.macd_bearish) else 20

        # rsi — avoid the naive "RSI>70 short / RSI<30 long" trap per spec
        if is_long:
            c["rsi"] = 100 if 40 <= inp.rsi <= 75 else (60 if inp.rsi > 75 else 20)
        else:
            c["rsi"] = 100 if 25 <= inp.rsi <= 60 else (60 if inp.rsi < 25 else 20)

        # volatility_bb — squeeze is neutral-to-positive (pre-breakout), not penalized
        c["volatility_bb"] = 70 if inp.bb_squeeze else 50

        # risk_reward_quality — placeholder, real R:R computed downstream once
        # entry/stop/target are set by target_engine; scored neutral here
        c["risk_reward_quality"] = 60

        return c

    def _weighted_total(self, components: dict) -> int:
        w = self.config.confluence_weights
        total = 0.0
        for key, value in components.items():
            weight = getattr(w, key, 0)
            total += value * (weight / 100.0)
        return round(total)

    # ---------- gold-specific overlay: modifier only ----------
    def _apply_gold_overlay(self, score: int, inp: ConfluenceInputs, direction: str) -> int:
        overlay = self.config.gold_overlay
        adjustment = 0.0

        # macro_bias: positive = bullish gold. Modifies, doesn't trigger.
        macro_effect = inp.macro.macro_bias if direction == "LONG" else -inp.macro.macro_bias
        adjustment += (macro_effect / 100.0) * overlay.macro_bias_modifier_max

        # fair value deviation: only apply if reading is reliable
        if inp.fair_value.is_reliable and inp.fair_value.deviation_zscore is not None:
            # a LONG is slightly favoured if MCX is at a discount (zscore negative),
            # since reversion toward fair value would push price up (and vice versa)
            fv_effect = -inp.fair_value.deviation_zscore if direction == "LONG" else inp.fair_value.deviation_zscore
            fv_effect = max(-1.0, min(1.0, fv_effect / 2.0))  # normalize, cap influence
            adjustment += fv_effect * overlay.fair_value_deviation_max

        # session quality — read from the explicit field, not a key that
        # MacroBiasResult never contains
        if not inp.session_quality_ok:
            adjustment -= overlay.session_quality_max

        return round(score + adjustment)

    # ---------- stacked confluence detection ----------
    def _is_stacked_confluence(self, inp: ConfluenceInputs) -> bool:
        """Sweep + FVG fill + volume, per the discussion — high-quality entry pattern."""
        had_sweep = inp.ltf_structure.last_event in (
            StructureEvent.LIQUIDITY_SWEEP_HIGH, StructureEvent.LIQUIDITY_SWEEP_LOW
        )
        has_active_or_recent_fvg = len(inp.ltf_structure.active_fvgs) > 0
        return had_sweep and has_active_or_recent_fvg and inp.volume_supportive

    # ---------- final decision ----------
    def _decide(self, long_score: int, short_score: int, inp: ConfluenceInputs,
                stacked: bool) -> tuple[Decision, int, TradeType | None]:
        t = self.config.thresholds

        if not inp.situation.ready_for_ai_review:
            return Decision.NO_TRADE, 0, None

        best_score = max(long_score, short_score)
        if best_score <= t.no_trade_max:
            return Decision.NO_TRADE, 0, None

        # require meaningful separation so a genuinely ambiguous market
        # (both directions scoring similarly) doesn't produce a coin-flip trade
        if abs(long_score - short_score) < 10:
            return Decision.NO_TRADE, 0, None

        confidence = min(100, best_score + (10 if stacked else 0))

        if long_score > short_score:
            trade_type = self._infer_trade_type(inp, "LONG")
            return Decision.BUY, confidence, trade_type
        else:
            trade_type = self._infer_trade_type(inp, "SHORT")
            return Decision.SELL, confidence, trade_type

    def _infer_trade_type(self, inp: ConfluenceInputs, direction: str) -> TradeType:
        if inp.situation.is_pullback_opportunity:
            return TradeType.PULLBACK
        if inp.situation.regime == MarketRegime.REVERSAL_POSSIBLE:
            return TradeType.REVERSAL
        if inp.situation.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            return TradeType.MOMENTUM
        if inp.bb_squeeze:
            return TradeType.BREAKOUT
        return TradeType.RANGE

    def _build_reasons(self, inp: ConfluenceInputs, decision: Decision,
                        components: dict) -> tuple[list, list]:
        if decision == Decision.NO_TRADE:
            return [], [inp.situation.explanation, *inp.situation.warnings]

        reasons_for = [f"{k}: {v}/100" for k, v in components.items() if v >= 70]
        reasons_against = [f"{k}: {v}/100" for k, v in components.items() if v < 40]
        reasons_against.extend(inp.situation.warnings)
        return reasons_for, reasons_against

    def _invalidation_condition(self, decision: Decision, inp: ConfluenceInputs) -> str:
        if decision == Decision.BUY:
            return "Opposite CHOCH (bearish) confirmed, or price closes back below the entry structure level."
        if decision == Decision.SELL:
            return "Opposite CHOCH (bullish) confirmed, or price closes back above the entry structure level."
        return "N/A — no trade taken."
