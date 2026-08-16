"""
Incremental Indicators Engine.
Each indicator updates in O(1) per new candle — NOT a full recalculation
over the whole series. This is what keeps the hot-path decision loop
(Sprint 1 latency design) under ~30ms per tick.
"""
from dataclasses import dataclass, field
from collections import deque


@dataclass
class EMAState:
    period: int
    value: float | None = None
    alpha: float = field(init=False)

    def __post_init__(self):
        self.alpha = 2.0 / (self.period + 1)

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price  # seed with first price
        else:
            self.value = (price - self.value) * self.alpha + self.value
        return self.value


@dataclass
class RSIState:
    period: int = 14
    avg_gain: float | None = None
    avg_loss: float | None = None
    prev_price: float | None = None
    value: float = 50.0

    def update(self, price: float) -> float:
        if self.prev_price is None:
            self.prev_price = price
            return self.value

        change = price - self.prev_price
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self.avg_gain is None:
            self.avg_gain = gain
            self.avg_loss = loss
        else:
            # Wilder's smoothing
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        self.prev_price = price

        if self.avg_loss == 0:
            self.value = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.value = 100.0 - (100.0 / (1.0 + rs))
        return self.value


@dataclass
class MACDState:
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    fast_ema: EMAState = field(init=False)
    slow_ema: EMAState = field(init=False)
    signal_ema: EMAState = field(init=False)
    macd_line: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    prev_histogram: float = 0.0

    def __post_init__(self):
        self.fast_ema = EMAState(period=self.fast_period)
        self.slow_ema = EMAState(period=self.slow_period)
        self.signal_ema = EMAState(period=self.signal_period)

    def update(self, price: float) -> tuple[float, float, float]:
        fast = self.fast_ema.update(price)
        slow = self.slow_ema.update(price)
        self.macd_line = fast - slow
        self.signal_line = self.signal_ema.update(self.macd_line)
        self.prev_histogram = self.histogram
        self.histogram = self.macd_line - self.signal_line
        return self.macd_line, self.signal_line, self.histogram


@dataclass
class ATRState:
    period: int = 14
    prev_close: float | None = None
    value: float | None = None
    _tr_buffer: deque = field(default_factory=lambda: deque(maxlen=14))

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close),
            )
        self.prev_close = close

        if self.value is None:
            self._tr_buffer.append(tr)
            if len(self._tr_buffer) == self.period:
                self.value = sum(self._tr_buffer) / self.period
            else:
                self.value = tr  # provisional until warmed up
        else:
            # Wilder's smoothing
            self.value = (self.value * (self.period - 1) + tr) / self.period

        return self.value


@dataclass
class VWAPState:
    """Session VWAP — resets at the start of each trading session."""
    cumulative_pv: float = 0.0
    cumulative_volume: float = 0.0
    value: float | None = None

    def update(self, high: float, low: float, close: float, volume: float) -> float:
        typical_price = (high + low + close) / 3.0
        self.cumulative_pv += typical_price * volume
        self.cumulative_volume += volume
        if self.cumulative_volume > 0:
            self.value = self.cumulative_pv / self.cumulative_volume
        return self.value if self.value is not None else close

    def reset_session(self):
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        self.value = None


@dataclass
class BollingerBandsState:
    period: int = 20
    std_mult: float = 2.0
    _prices: deque = field(default_factory=lambda: deque(maxlen=20))
    upper: float | None = None
    mid: float | None = None
    lower: float | None = None

    def update(self, price: float) -> tuple[float, float, float]:
        self._prices.append(price)
        n = len(self._prices)
        mean = sum(self._prices) / n
        if n > 1:
            variance = sum((p - mean) ** 2 for p in self._prices) / n
            std = variance ** 0.5
        else:
            std = 0.0
        self.mid = mean
        self.upper = mean + self.std_mult * std
        self.lower = mean - self.std_mult * std
        return self.upper, self.mid, self.lower


class IndicatorEngine:
    """
    One instance per (instrument, timeframe) pair.
    Call update() once per CLOSED candle only.
    """

    def __init__(self, volume_avg_period: int = 20, atr_avg_period: int = 20):
        self.ema9 = EMAState(period=9)
        self.ema21 = EMAState(period=21)
        self.ema50 = EMAState(period=50)
        self.ema200 = EMAState(period=200)
        self.rsi = RSIState(period=14)
        self.macd = MACDState()
        self.atr = ATRState(period=14)
        self.vwap = VWAPState()
        self.bb = BollingerBandsState()

        self._volume_buffer: deque = deque(maxlen=volume_avg_period)
        self._atr_buffer: deque = deque(maxlen=atr_avg_period)

    def update(self, high: float, low: float, close: float, volume: float) -> dict:
        self.ema9.update(close)
        self.ema21.update(close)
        self.ema50.update(close)
        self.ema200.update(close)
        self.rsi.update(close)
        self.macd.update(close)
        atr_val = self.atr.update(high, low, close)
        self.vwap.update(high, low, close, volume)
        self.bb.update(close)

        self._volume_buffer.append(volume)
        self._atr_buffer.append(atr_val)

        rel_volume = (
            volume / (sum(self._volume_buffer) / len(self._volume_buffer))
            if self._volume_buffer and sum(self._volume_buffer) > 0
            else 1.0
        )
        atr_avg_20 = (
            sum(self._atr_buffer) / len(self._atr_buffer)
            if self._atr_buffer else atr_val
        )

        return {
            "ema9": self.ema9.value,
            "ema21": self.ema21.value,
            "ema50": self.ema50.value,
            "ema200": self.ema200.value,
            "rsi": self.rsi.value,
            "macd_line": self.macd.macd_line,
            "macd_signal": self.macd.signal_line,
            "macd_hist": self.macd.histogram,
            "macd_hist_prev": self.macd.prev_histogram,
            "atr": atr_val,
            "atr_avg_20": atr_avg_20,
            "vwap": self.vwap.value,
            "bb_upper": self.bb.upper,
            "bb_mid": self.bb.mid,
            "bb_lower": self.bb.lower,
            "rel_volume": rel_volume,
        }
