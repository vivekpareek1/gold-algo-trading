"""
Gold Intelligence Module — the project's differentiator.
1. Synthetic fair value: MCX GOLDM vs international XAU/USD, adjusted for
   USD/INR and import duty, to detect premium/discount deviation.
2. Move classification: is a price move metal-driven, rupee-driven,
   amplified, or conflicted.
3. Macro bias: DXY, US real yields, USD/INR, crude -> a filter/conviction
   modifier, NEVER a standalone trade trigger (per Sprint 1 design).
"""
from dataclasses import dataclass
from enum import Enum
from collections import deque


class MoveClassification(str, Enum):
    METAL_DRIVEN = "METAL_DRIVEN"
    RUPEE_DRIVEN = "RUPEE_DRIVEN"
    AMPLIFIED = "AMPLIFIED"       # both XAU/USD and USD/INR move same direction
    CONFLICTED = "CONFLICTED"     # moving opposite directions
    FLAT = "FLAT"


@dataclass
class FairValueResult:
    mcx_price: float
    theoretical_price: float
    deviation: float
    deviation_pct: float
    deviation_zscore: float | None
    is_reliable: bool             # False if session mismatch / stale data
    unreliable_reason: str = ""


@dataclass
class MacroInputs:
    dxy: float
    dxy_prev: float
    us10y_real_yield: float       # FRED DFII10
    us10y_real_yield_prev: float
    usdinr: float
    usdinr_prev: float
    crude: float
    crude_prev: float


@dataclass
class MacroBiasResult:
    macro_bias: float   # -100..+100, bullish-for-gold positive
    components: dict


class FairValueEngine:
    """
    landed_cost_per_10g = (XAUUSD / 31.1035) * 10 * USDINR * (1 + import_duty)
    theoretical_futures_price = landed_cost * (1 + carry_cost * days_to_expiry/365)
    """

    def __init__(self, config, zscore_lookback: int = 100):
        self.config = config
        self._history: deque = deque(maxlen=zscore_lookback)

    def calculate(self, mcx_price: float, xauusd: float, usdinr: float,
                  days_to_expiry: int, both_sessions_live: bool,
                  data_stale: bool = False) -> FairValueResult:

        if data_stale:
            return FairValueResult(
                mcx_price=mcx_price, theoretical_price=0, deviation=0,
                deviation_pct=0, deviation_zscore=None,
                is_reliable=False, unreliable_reason="Input data is stale"
            )

        if self.config.gold_specific.session_overlap_required_for_deviation and not both_sessions_live:
            return FairValueResult(
                mcx_price=mcx_price, theoretical_price=0, deviation=0,
                deviation_pct=0, deviation_zscore=None,
                is_reliable=False,
                unreliable_reason="MCX and international gold sessions not both live — "
                                   "deviation reading would be unreliable"
            )

        troy_oz = self.config.gold_specific.troy_oz_to_grams
        duty = self.config.gold_specific.import_duty_rate
        carry = self.config.gold_specific.carry_cost_rate_annual

        landed_cost_per_10g = (xauusd / troy_oz) * 10 * usdinr * (1 + duty)
        theoretical_price = landed_cost_per_10g * (1 + carry * days_to_expiry / 365)

        deviation = mcx_price - theoretical_price
        deviation_pct = (deviation / theoretical_price * 100) if theoretical_price != 0 else 0.0

        self._history.append(deviation_pct)
        zscore = self._compute_zscore(deviation_pct)

        return FairValueResult(
            mcx_price=mcx_price,
            theoretical_price=theoretical_price,
            deviation=deviation,
            deviation_pct=deviation_pct,
            deviation_zscore=zscore,
            is_reliable=True,
        )

    def _compute_zscore(self, current_value: float) -> float | None:
        if len(self._history) < 10:  # not enough history for a meaningful zscore
            return None
        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (current_value - mean) / std

    def classify_move(self, xauusd_change_pct: float, usdinr_change_pct: float,
                       flat_threshold: float = 0.02) -> MoveClassification:
        """
        Decompose an MCX gold move into what actually drove it.
        flat_threshold: below this magnitude (%), treat the leg as flat/noise.
        """
        xau_moving = abs(xauusd_change_pct) > flat_threshold
        inr_moving = abs(usdinr_change_pct) > flat_threshold

        if not xau_moving and not inr_moving:
            return MoveClassification.FLAT
        if xau_moving and not inr_moving:
            return MoveClassification.METAL_DRIVEN
        if inr_moving and not xau_moving:
            return MoveClassification.RUPEE_DRIVEN

        # both moving — same sign = amplified, opposite sign = conflicted
        same_direction = (xauusd_change_pct > 0) == (usdinr_change_pct > 0)
        return MoveClassification.AMPLIFIED if same_direction else MoveClassification.CONFLICTED


class MacroContextEngine:
    """
    Produces MACRO_BIAS (-100..+100). This is a FILTER/CONVICTION MODIFIER
    ONLY — per Sprint 1 design, it must never generate a trade on its own.
    Weights are configurable hypotheses, not fixed truths.
    """

    # starting weights — must be validated via backtesting, same principle
    # as confluence_weights in config/settings.py
    WEIGHTS = {
        "real_yield": 0.30,   # falling real yields = bullish gold
        "dxy": 0.25,          # falling DXY = bullish gold
        "usdinr": 0.15,       # rising USDINR = bullish for MCX gold specifically
        "crude": 0.10,        # rising crude = mild bullish (inflation proxy)
    }

    def compute(self, inputs: MacroInputs) -> MacroBiasResult:
        components = {}

        # each component scored -100..+100 based on direction, then weighted
        components["real_yield"] = self._direction_score(
            inputs.us10y_real_yield_prev, inputs.us10y_real_yield, invert=True
        )
        components["dxy"] = self._direction_score(
            inputs.dxy_prev, inputs.dxy, invert=True
        )
        components["usdinr"] = self._direction_score(
            inputs.usdinr_prev, inputs.usdinr, invert=False
        )
        components["crude"] = self._direction_score(
            inputs.crude_prev, inputs.crude, invert=False
        )

        macro_bias = sum(components[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        macro_bias = max(-100.0, min(100.0, macro_bias))

        return MacroBiasResult(macro_bias=macro_bias, components=components)

    def _direction_score(self, prev: float, current: float, invert: bool) -> float:
        """
        Simple direction score, not magnitude-weighted (v1). Falling value
        scores positive if invert=True (e.g. falling yields = bullish gold).
        """
        if prev == 0:
            return 0.0
        pct_change = (current - prev) / abs(prev) * 100
        score = -pct_change if invert else pct_change
        # clip a single indicator's contribution so no one input dominates
        return max(-100.0, min(100.0, score * 20))  # scaled for sensitivity
