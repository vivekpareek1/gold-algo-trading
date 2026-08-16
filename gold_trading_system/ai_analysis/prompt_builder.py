"""
Prompt Builder — assembles the structured context block the AI reasons over
(Sprint 1 §9's exact field list). AI receives ONLY this structured summary,
never raw candle series — the deterministic engines have already done the
calculation work; AI does interpretation on top of it.
"""
from market_structure.structure_engine import StructureState, TrendState, StructureEvent
from situation_analysis.situation_analyzer import SituationSnapshot, IndicatorSnapshot
from gold_intelligence.fair_value import FairValueResult, MacroBiasResult
from signal_engine.signal_engine import ConfluenceResult


def build_context(
    instrument: str,
    current_price: float,
    htf_structure: StructureState,
    ltf_structure: StructureState,
    situation: SituationSnapshot,
    indicators: IndicatorSnapshot,
    fair_value: FairValueResult,
    macro: MacroBiasResult,
    confluence: ConfluenceResult,
    news_risk: str = "NORMAL",
    spread_points: float = 0.0,
    session: str = "UNKNOWN",
) -> str:
    """Returns the structured text block to send to the AI reasoning call."""

    nearest_swing_high = (
        ltf_structure.swing_highs[-1].price if ltf_structure.swing_highs else None
    )
    nearest_swing_low = (
        ltf_structure.swing_lows[-1].price if ltf_structure.swing_lows else None
    )

    lines = [
        f"Market: MCX Gold",
        f"Instrument: {instrument}",
        f"Current Price: {current_price}",
        "",
        f"Market Regime: {situation.regime.value}",
        f"Higher-Timeframe Trend: {htf_structure.trend.value}",
        f"Lower-Timeframe Trend: {ltf_structure.trend.value}",
        f"Trend Alignment Score: {situation.trend_alignment_score}/100",
        f"Is Pullback Opportunity: {situation.is_pullback_opportunity}",
        "",
        f"EMA9: {indicators.ema9:.2f}",
        f"EMA21: {indicators.ema21:.2f}",
        f"EMA50: {indicators.ema50:.2f}",
        f"EMA200: {indicators.ema200:.2f}",
        f"RSI: {indicators.rsi:.1f}",
        f"MACD Histogram: {indicators.macd_hist:.4f} (prev: {indicators.macd_hist_prev:.4f})",
        f"ATR: {indicators.atr:.2f} (20-period avg: {indicators.atr_avg_20:.2f})",
        "",
        f"Relative Volume: {indicators.rel_volume:.2f}x",
        f"Momentum Health: {situation.momentum_health}",
        "",
        f"Last Structure Event: {ltf_structure.last_event.value}",
        f"Nearest Swing High: {nearest_swing_high}",
        f"Nearest Swing Low: {nearest_swing_low}",
        f"Active FVGs: {len(ltf_structure.active_fvgs)}",
        "",
        f"MCX Fair Value Deviation: {fair_value.deviation_pct:.3f}% "
        f"(reliable: {fair_value.is_reliable}, z-score: {fair_value.deviation_zscore})",
        f"Macro Bias (gold): {macro.macro_bias:.1f} (-100 bearish to +100 bullish)",
        "",
        f"Confluence Long Score: {confluence.long_score}/100",
        f"Confluence Short Score: {confluence.short_score}/100",
        f"Stacked Confluence (sweep+FVG+volume): {confluence.stacked_confluence}",
        "",
        f"News Risk: {news_risk}",
        f"Spread: {spread_points} points",
        f"Session: {session}",
        "",
        f"Situation Summary: {situation.explanation}",
        f"Warnings: {'; '.join(situation.warnings) if situation.warnings else 'None'}",
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a disciplined institutional gold trading analyst reviewing \
already-computed market structure, indicators, and risk context for MCX GOLDM. \
You do NOT calculate indicators yourself — they are provided. Your job is contextual \
interpretation: does this setup genuinely warrant a trade, and why or why not.

Rules:
- NO_TRADE is a valid, expected, successful outcome. Prefer it over a marginal setup.
- Never recommend a position size — that is the risk engine's job.
- Never invent news/economic events — only use what's given in News Risk.
- If confluence scores conflict with the situation summary, say so explicitly \
  rather than picking a side silently.
- Respond ONLY with a single JSON object matching the required schema. No prose, \
  no markdown fences, nothing outside the JSON object.
"""
