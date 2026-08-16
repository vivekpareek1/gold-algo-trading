"""
AI Analysis Layer — strict schema (Sprint 1 §9, and the AI-vs-algorithm
boundary from throughout the build).

AI does: contextual interpretation, setup ranking, conflict analysis,
explanation. AI NEVER does: calculate indicators, decide position size,
override risk limits, invent economic events, modify live parameters.

A malformed or failed AI response is treated as NO_TRADE — fail-safe,
never fail-open. The AI is advisory; the risk engine has final veto.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class AISignal(BaseModel):
    decision: Literal["BUY", "SELL", "NO_TRADE"]
    confidence: int = Field(ge=0, le=100)

    entry_zone_low: float | None = None
    entry_zone_high: float | None = None
    stop_loss: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    target_3: float | None = None
    risk_reward: float | None = None

    market_regime: str = ""
    trade_type: Literal["BREAKOUT", "PULLBACK", "MOMENTUM", "REVERSAL", "RANGE"] | None = None

    reasons_for_entry: list[str] = Field(default_factory=list)
    reasons_against_entry: list[str] = Field(default_factory=list)
    invalidation_condition: str = ""
    trailing_method: Literal["ATR", "EMA9", "EMA21", "EMA50", "STRUCTURE", "HYBRID"] | None = None
    news_risk: Literal["NORMAL", "EVENT_APPROACHING", "HIGH_IMPACT", "POST_EVENT"] = "NORMAL"
    final_explanation: str = ""

    @field_validator("decision")
    @classmethod
    def decision_consistency(cls, v, info):
        return v

    def model_post_init(self, __context) -> None:
        # A BUY/SELL decision must carry the essential trade parameters.
        # A response claiming BUY with no stop-loss is not usable — treat
        # as malformed so the caller can fall back to NO_TRADE.
        if self.decision in ("BUY", "SELL"):
            if self.stop_loss is None or self.entry_zone_low is None:
                raise ValueError(
                    f"Decision={self.decision} but missing stop_loss/entry_zone — "
                    f"incomplete AI response, must not be trusted as-is."
                )

    @classmethod
    def no_trade_fallback(cls, reason: str) -> "AISignal":
        """The fail-safe default used whenever the AI call fails or returns
        something unparseable. Never fail-open into a trade."""
        return cls(
            decision="NO_TRADE",
            confidence=0,
            final_explanation=f"AI call failed or returned invalid output — defaulting to "
                               f"NO_TRADE (fail-safe). Reason: {reason}",
        )
