"""
Tick-to-Candle Aggregator.
Angel One's WebSocket streams raw tick-by-tick LTP (last traded price)
data, not pre-built candles. This module buckets ticks into 5-minute
OHLCV candles and emits a completed candle exactly once, the moment its
time window has elapsed — never early, never twice.

This is deliberately separate from and reusable by any tick source, so
the same aggregator works whether the feed is Angel One's WebSocket, a
different broker, or a test harness.
"""
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Tick:
    ts: int          # epoch seconds
    ltp: float        # last traded price
    volume: float = 0.0   # cumulative or incremental volume, depending on feed


@dataclass
class CompletedCandle:
    ts: int    # bucket start, epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


class TickAggregator:
    def __init__(self, interval_minutes: int = 5):
        self.interval_minutes = interval_minutes
        self._bucket_start: int | None = None
        self._o = self._h = self._l = self._c = None
        self._v = 0.0
        self._last_cumulative_volume: float | None = None
        # Volume-sanity tracking. A feed subscribed in the wrong mode (Angel
        # One's LTP mode 1 carries NO volume field at all) silently yields
        # volume=0 forever, which turns every volume-confirmation check in
        # the strategy into a no-op that always passes. On real 2-year MCX
        # data that flipped results from +0.262R / PF 1.69 to -0.378R / PF
        # 0.37 — so this must be loud, not silent.
        self.ticks_seen = 0
        self.nonzero_volume_ticks = 0

    @property
    def volume_feed_looks_broken(self) -> bool:
        """True once enough ticks have arrived with zero volume throughout —
        the signature of subscribing in a mode that carries no volume."""
        return self.ticks_seen >= 50 and self.nonzero_volume_ticks == 0

    def _bucket_of(self, ts: int) -> int:
        span = self.interval_minutes * 60
        return (ts // span) * span

    def add_tick(self, tick: Tick) -> CompletedCandle | None:
        """
        Feed one tick. Returns a CompletedCandle if this tick's timestamp
        crossed into a new bucket (meaning the PREVIOUS bucket just closed),
        else None. The in-progress bucket is never returned — only fully
        elapsed windows, same look-ahead guard used everywhere else in this
        system.
        """
        b = self._bucket_of(tick.ts)
        self.ticks_seen += 1
        if tick.volume:
            self.nonzero_volume_ticks += 1

        if self._bucket_start is None:
            self._start_bucket(b, tick)
            return None

        if b == self._bucket_start:
            self._h = max(self._h, tick.ltp)
            self._l = min(self._l, tick.ltp)
            self._c = tick.ltp
            self._v += self._incremental_volume(tick)
            return None

        # tick belongs to a NEW bucket — the old one is now complete
        completed = CompletedCandle(ts=self._bucket_start, open=self._o, high=self._h,
                                      low=self._l, close=self._c, volume=self._v)
        self._start_bucket(b, tick)
        return completed

    def _start_bucket(self, bucket_start: int, tick: Tick):
        self._bucket_start = bucket_start
        self._o = self._h = self._l = self._c = tick.ltp
        self._v = self._incremental_volume(tick)

    def _incremental_volume(self, tick: Tick) -> float:
        """
        Angel One's feed typically reports CUMULATIVE traded volume for the
        day, not per-tick volume. If it looks cumulative (monotonically
        non-decreasing, large jumps), derive the incremental amount; if it
        looks like a small per-tick value already, use it as-is. This
        heuristic errs toward treating volume as cumulative, since that is
        Angel One's documented behavior for LTP feeds — adjust if your feed
        differs.
        """
        if self._last_cumulative_volume is None:
            self._last_cumulative_volume = tick.volume
            return 0.0
        delta = tick.volume - self._last_cumulative_volume
        self._last_cumulative_volume = tick.volume
        return max(delta, 0.0)   # a volume counter resetting (new session) must not go negative
