"""
Situation Analysis Engine
Synthesizes structure, trend, momentum, macro, and risk state into ONE
coherent market-regime read with a plain-language explanation.

This is the "assembled context" layer — it's what makes the system
explainable (WHY BUY / WHY NO TRADE) and it's the gate that decides
whether the AI reasoning call is even worth making. Deterministic only;
no AI call happens inside this module.
"""
from dataclasses import dataclass, field
from enum import Enum

from market_structure.structure_engine import TrendState, StructureEvent, StructureState


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    REVERSAL_POSSIBLE = "REVERSAL_POSSIBLE"
    NOISY_MARKET = "NOISY_MARKET"


@dataclass
class IndicatorSnapshot:
    """Minimal fields Situation Analysis needs — full set lives in indicators module."""
    ema9: float = 0.0
    ema21: float = 0.0
    ema50: float = 0.0
    ema200: float = 0.0
    rsi: float = 50.0
    macd_hist: float = 0.0
    macd_hist_prev: float = 0.0     # for acceleration/deceleration check
    atr: float = 0.0
    atr_avg_20: float = 0.0         # to judge current ATR vs its own recent norm
    rel_volume: float = 1.0         # current volume / average volume


@dataclass
class MacroContext:
    macro_bias: float = 0.0          # -100..+100, from gold_intelligence module
    fair_value_deviation_pct: float = 0.0
    fair_value_zscore: float = 0.0
    session_quality_ok: bool = True
    move_classification: str = "UNKNOWN"  # METAL_DRIVEN / RUPEE_DRIVEN / AMPLIFIED / CONFLICTED


@dataclass
class SituationSnapshot:
    regime: MarketRegime
    htf_trend: TrendState
    ltf_trend: TrendState
    trend_alignment_score: int          # 0-100
    momentum_health: str                 # STRONG / WEAKENING / DEAD
    last_structure_event: StructureEvent
    is_pullback_opportunity: bool
    macro_alignment: str                 # SUPPORTIVE / NEUTRAL / OPPOSING
    ready_for_ai_review: bool            # gate: has this crossed the noise floor?
    explanation: str                     # plain-language summary for dashboard/journal
    warnings: list = field(default_factory=list)


class SituationAnalyzer:
    def __init__(self, config):
        self.config = config

    def analyze(self, htf_structure: StructureState, ltf_structure: StructureState,
                indicators: IndicatorSnapshot, macro: MacroContext) -> SituationSnapshot:

        warnings = []

        trend_alignment_score = self._trend_alignment_score(
            htf_structure.trend, ltf_structure.trend, ltf_structure.is_pullback_in_trend
        )

        momentum_health = self._momentum_health(indicators)

        regime = self._classify_regime(
            htf_structure, ltf_structure, indicators, momentum_health
        )

        macro_alignment = self._macro_alignment(macro, ltf_structure.trend)

        ready_for_ai = self._is_ready_for_ai(
            regime, trend_alignment_score, momentum_health, macro
        )

        if not macro.session_quality_ok:
            warnings.append("Session quality flag active (thin liquidity or market mismatch) — "
                             "fair-value deviation reading may be unreliable.")
        if indicators.atr > 0 and indicators.atr_avg_20 > 0 and indicators.atr > indicators.atr_avg_20 * 2:
            warnings.append("ATR is more than 2x its 20-period average — abnormal volatility, "
                             "consider reduced size or standing aside.")
        if ltf_structure.last_event in (StructureEvent.LIQUIDITY_SWEEP_HIGH,
                                          StructureEvent.LIQUIDITY_SWEEP_LOW):
            warnings.append(f"Recent {ltf_structure.last_event.value} detected — "
                             f"await confirmation candle before treating this as a reversal.")

        explanation = self._build_explanation(
            regime, htf_structure, ltf_structure, momentum_health,
            macro_alignment, trend_alignment_score
        )

        return SituationSnapshot(
            regime=regime,
            htf_trend=htf_structure.trend,
            ltf_trend=ltf_structure.trend,
            trend_alignment_score=trend_alignment_score,
            momentum_health=momentum_health,
            last_structure_event=ltf_structure.last_event,
            is_pullback_opportunity=ltf_structure.is_pullback_in_trend,
            macro_alignment=macro_alignment,
            ready_for_ai_review=ready_for_ai,
            explanation=explanation,
            warnings=warnings,
        )

    # ---------- sub-analyses ----------
    def _trend_alignment_score(self, htf_trend: TrendState, ltf_trend: TrendState,
                                 is_pullback: bool) -> int:
        if htf_trend == ltf_trend and htf_trend != TrendState.RANGE:
            return 90  # fully aligned
        if is_pullback:
            return 65  # against-HTF move classified as a valid pullback, not a conflict
        if htf_trend == TrendState.RANGE or ltf_trend == TrendState.RANGE:
            return 40  # ambiguous, not a genuine conflict either
        return 15  # genuine cross-timeframe conflict

    def _momentum_health(self, ind: IndicatorSnapshot) -> str:
        macd_accelerating = abs(ind.macd_hist) > abs(ind.macd_hist_prev)
        # ATR-relative threshold avoids the degenerate case where flat EMAs
        # (0 separation) trivially satisfy a purely relative comparison
        min_meaningful_separation = max(ind.atr * 0.1, 1e-6)
        ema_stack_separating = abs(ind.ema9 - ind.ema21) > min_meaningful_separation
        volume_supportive = ind.rel_volume >= 1.0

        score = sum([macd_accelerating, ema_stack_separating, volume_supportive])
        if score >= 2:
            return "STRONG"
        elif score == 1:
            return "WEAKENING"
        return "DEAD"

    def _classify_regime(self, htf: StructureState, ltf: StructureState,
                          ind: IndicatorSnapshot, momentum_health: str) -> MarketRegime:
        # reversal signal takes priority — sweep + CHOCH combo
        if ltf.last_event in (StructureEvent.CHOCH_BULLISH, StructureEvent.CHOCH_BEARISH):
            return MarketRegime.REVERSAL_POSSIBLE

        if ind.atr > 0 and ind.atr_avg_20 > 0:
            if ind.atr > ind.atr_avg_20 * 1.5:
                return MarketRegime.HIGH_VOLATILITY
            if ind.atr < ind.atr_avg_20 * 0.6:
                return MarketRegime.LOW_VOLATILITY

        if ltf.trend == TrendState.TRENDING_UP and momentum_health == "STRONG":
            return MarketRegime.TRENDING_UP
        if ltf.trend == TrendState.TRENDING_DOWN and momentum_health == "STRONG":
            return MarketRegime.TRENDING_DOWN

        # check the more specific NOISY_MARKET condition before the generic RANGE
        # catch-all, otherwise dead-momentum sideways markets never get flagged
        if momentum_health == "DEAD" and ltf.trend == TrendState.RANGE:
            return MarketRegime.NOISY_MARKET

        if ltf.trend == TrendState.RANGE and htf.trend == TrendState.RANGE:
            return MarketRegime.RANGE

        return MarketRegime.RANGE  # conservative default

    def _macro_alignment(self, macro: MacroContext, ltf_trend: TrendState) -> str:
        # macro_bias > 0 = bullish for gold; align with an uptrend, oppose a downtrend
        if ltf_trend == TrendState.TRENDING_UP:
            if macro.macro_bias > 20:
                return "SUPPORTIVE"
            if macro.macro_bias < -20:
                return "OPPOSING"
        elif ltf_trend == TrendState.TRENDING_DOWN:
            if macro.macro_bias < -20:
                return "SUPPORTIVE"
            if macro.macro_bias > 20:
                return "OPPOSING"
        return "NEUTRAL"

    def _is_ready_for_ai(self, regime: MarketRegime, trend_alignment: int,
                           momentum_health: str, macro: MacroContext) -> bool:
        """
        Gate that keeps the AI call OUT of the hot path (per Sprint 1 latency design).
        Only call AI when the deterministic picture is already interesting enough
        to be worth reasoning over — this is checked again by the full confluence
        score downstream; this is a coarse pre-filter, not the final gate.
        """
        if regime == MarketRegime.NOISY_MARKET:
            return False
        if trend_alignment < 40:
            return False
        if momentum_health == "DEAD" and regime not in (MarketRegime.REVERSAL_POSSIBLE,):
            return False
        return True

    def _build_explanation(self, regime, htf, ltf, momentum_health,
                            macro_alignment, trend_alignment_score) -> str:
        parts = [f"Regime: {regime.value}."]

        if htf.trend == ltf.trend:
            parts.append(f"Higher and lower timeframes both {htf.trend.value.lower().replace('_', ' ')} "
                         f"— aligned (score {trend_alignment_score}/100).")
        elif ltf.is_pullback_in_trend:
            parts.append(f"Higher timeframe {htf.trend.value.lower().replace('_', ' ')}, "
                         f"lower timeframe pulling back — treated as a pullback opportunity, "
                         f"not a conflict (score {trend_alignment_score}/100).")
        else:
            parts.append(f"Higher timeframe {htf.trend.value.lower().replace('_', ' ')} vs "
                         f"lower timeframe {ltf.trend.value.lower().replace('_', ' ')} — "
                         f"genuine conflict (score {trend_alignment_score}/100).")

        parts.append(f"Momentum is {momentum_health.lower()}.")

        if ltf.last_event != StructureEvent.NONE:
            parts.append(f"Last structure event: {ltf.last_event.value}.")

        parts.append(f"Macro backdrop is {macro_alignment.lower()} of the current move.")

        return " ".join(parts)
