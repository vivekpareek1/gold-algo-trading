"""
Market Structure Engine
Detects: swing points, trend, BOS/CHOCH, liquidity sweeps, FVG, pullback-vs-conflict.

Design rule from the spec: never trade on a single signal. This module produces
STATE, not decisions — signal_engine consumes this state alongside indicators.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TrendState(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE = "RANGE"


class StructureEvent(str, Enum):
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"
    LIQUIDITY_SWEEP_HIGH = "LIQUIDITY_SWEEP_HIGH"   # swept a high, potential bearish reversal
    LIQUIDITY_SWEEP_LOW = "LIQUIDITY_SWEEP_LOW"     # swept a low, potential bullish reversal
    NONE = "NONE"


@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def is_bullish(self) -> bool:
        return self.close > self.open

    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["HIGH", "LOW"]
    swept: bool = False


@dataclass
class FVG:
    start_index: int          # index of candle 1 (the gap-leaving candle's neighbour)
    end_index: int             # index of candle 3
    top: float
    bottom: float
    direction: Literal["BULLISH", "BEARISH"]
    filled: bool = False
    fill_pct: float = 0.0


@dataclass
class StructureState:
    trend: TrendState = TrendState.RANGE
    last_event: StructureEvent = StructureEvent.NONE
    swing_highs: list = field(default_factory=list)
    swing_lows: list = field(default_factory=list)
    active_fvgs: list = field(default_factory=list)
    is_pullback_in_trend: bool = False   # True = against-trend move that is a VALID
                                          # pullback entry opportunity, not a conflict signal


class MarketStructureEngine:
    """
    Stateful engine — call update() with each new closed candle.
    NEVER acts on an unclosed/forming candle (avoids false sweep/breakout signals).
    """

    def __init__(self, swing_lookback: int = 3, sweep_buffer_atr_mult: float = 0.3):
        self.candles: list[Candle] = []
        self.swing_lookback = swing_lookback
        self.sweep_buffer_atr_mult = sweep_buffer_atr_mult
        self.state = StructureState()

    def update(self, candle: Candle, current_atr: float, higher_tf_trend: TrendState | None = None):
        """
        candle: must be a CLOSED candle only. Passing a forming candle will
        corrupt swing detection and produce false sweep signals.
        higher_tf_trend: pass the trend from the higher timeframe so this
        engine can classify a counter-move as pullback vs genuine conflict.
        """
        self.candles.append(candle)
        self._detect_swings()
        self._detect_liquidity_sweep(current_atr)
        self._detect_trend()
        self._detect_fvg()
        self._classify_pullback_vs_conflict(higher_tf_trend)
        return self.state

    # ---------- swing detection ----------
    def _detect_swings(self):
        n = len(self.candles)
        i = n - 1 - self.swing_lookback
        if i < self.swing_lookback:
            return  # not enough candles yet

        window = self.candles[i - self.swing_lookback: i + self.swing_lookback + 1]
        mid = self.candles[i]

        if all(mid.high >= c.high for c in window):
            self.state.swing_highs.append(SwingPoint(index=i, price=mid.high, kind="HIGH"))
        if all(mid.low <= c.low for c in window):
            self.state.swing_lows.append(SwingPoint(index=i, price=mid.low, kind="LOW"))

        # keep memory bounded
        self.state.swing_highs = self.state.swing_highs[-50:]
        self.state.swing_lows = self.state.swing_lows[-50:]

    # ---------- liquidity sweep + CHOCH ----------
    def _detect_liquidity_sweep(self, current_atr: float):
        """
        A sweep = wick breaks a prior swing level, but candle CLOSES back inside.
        This is the core rule from the spec: never enter on the breakout candle
        itself — classify wick-vs-close first.
        """
        if len(self.candles) < 2 or not self.state.swing_highs and not self.state.swing_lows:
            self.state.last_event = StructureEvent.NONE
            return

        current = self.candles[-1]
        buffer = current_atr * self.sweep_buffer_atr_mult

        # check sweep of most recent unswept swing high
        unswept_highs = [s for s in self.state.swing_highs if not s.swept]
        if unswept_highs:
            nearest_high = unswept_highs[-1]
            wick_beyond = current.high > nearest_high.price
            close_inside = current.close < nearest_high.price
            if wick_beyond and close_inside:
                nearest_high.swept = True
                self.state.last_event = StructureEvent.LIQUIDITY_SWEEP_HIGH
                return  # a sweep event takes priority this candle; don't also flag BOS

        unswept_lows = [s for s in self.state.swing_lows if not s.swept]
        if unswept_lows:
            nearest_low = unswept_lows[-1]
            wick_beyond = current.low < nearest_low.price
            close_inside = current.close > nearest_low.price
            if wick_beyond and close_inside:
                nearest_low.swept = True
                self.state.last_event = StructureEvent.LIQUIDITY_SWEEP_LOW
                return

        self.state.last_event = StructureEvent.NONE

    # ---------- trend detection ----------
    def _detect_trend(self):
        highs = self.state.swing_highs[-4:]
        lows = self.state.swing_lows[-4:]

        if len(highs) < 2 or len(lows) < 2:
            self.state.trend = TrendState.RANGE
            return

        higher_highs = highs[-1].price > highs[-2].price
        higher_lows = lows[-1].price > lows[-2].price
        lower_highs = highs[-1].price < highs[-2].price
        lower_lows = lows[-1].price < lows[-2].price

        prev_trend = self.state.trend

        if higher_highs and higher_lows:
            new_trend = TrendState.TRENDING_UP
        elif lower_highs and lower_lows:
            new_trend = TrendState.TRENDING_DOWN
        else:
            new_trend = TrendState.RANGE

        # CHOCH = trend was one direction, and structure just flipped
        if prev_trend == TrendState.TRENDING_UP and new_trend == TrendState.TRENDING_DOWN:
            self.state.last_event = StructureEvent.CHOCH_BEARISH
        elif prev_trend == TrendState.TRENDING_DOWN and new_trend == TrendState.TRENDING_UP:
            self.state.last_event = StructureEvent.CHOCH_BULLISH

        self.state.trend = new_trend

    # ---------- FVG detection ----------
    def _detect_fvg(self):
        """3-candle imbalance: candle[-3], candle[-2], candle[-1]."""
        if len(self.candles) < 3:
            return
        c1, c2, c3 = self.candles[-3], self.candles[-2], self.candles[-1]

        # bullish FVG: c1.high < c3.low  (gap left below c3, above c1)
        if c1.high < c3.low:
            self.state.active_fvgs.append(FVG(
                start_index=len(self.candles) - 3,
                end_index=len(self.candles) - 1,
                top=c3.low,
                bottom=c1.high,
                direction="BULLISH",
            ))

        # bearish FVG: c1.low > c3.high
        if c1.low > c3.high:
            self.state.active_fvgs.append(FVG(
                start_index=len(self.candles) - 3,
                end_index=len(self.candles) - 1,
                top=c1.low,
                bottom=c3.high,
                direction="BEARISH",
            ))

        # check fill status of existing FVGs against latest candle
        current = self.candles[-1]
        for fvg in self.state.active_fvgs:
            if fvg.filled:
                continue
            overlap_low = max(fvg.bottom, current.low)
            overlap_high = min(fvg.top, current.high)
            if overlap_high > overlap_low:
                gap_size = fvg.top - fvg.bottom
                filled_amount = overlap_high - overlap_low
                fvg.fill_pct = min(1.0, fvg.fill_pct + filled_amount / gap_size) if gap_size > 0 else 1.0
                if fvg.fill_pct >= 0.95:
                    fvg.filled = True

        self.state.active_fvgs = [f for f in self.state.active_fvgs if not f.filled][-20:]

    # ---------- pullback vs genuine conflict ----------
    def _classify_pullback_vs_conflict(self, higher_tf_trend: TrendState | None):
        """
        Core "follow the trend" rule: a lower-TF move against the higher-TF
        trend is NOT automatically bad. If higher TF is still intact, this is
        classified as a pullback (a valid entry opportunity), not a conflict.
        """
        if higher_tf_trend is None:
            self.state.is_pullback_in_trend = False
            return

        lower_tf_against_higher = (
            (higher_tf_trend == TrendState.TRENDING_UP and self.state.trend == TrendState.TRENDING_DOWN) or
            (higher_tf_trend == TrendState.TRENDING_DOWN and self.state.trend == TrendState.TRENDING_UP)
        )
        self.state.is_pullback_in_trend = lower_tf_against_higher
