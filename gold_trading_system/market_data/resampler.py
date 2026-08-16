"""
Candle Resampler — aggregates a base timeframe series (e.g. 5M) into higher
timeframes (15M/1H/4H) using real timestamp-boundary grouping, not just
"every N candles" (which breaks if the feed has gaps or missing candles —
exactly the scenario the Data Quality Engine flags but doesn't reject).

Standard OHLCV aggregation rules:
  open   = first candle's open in the bucket
  high   = max high across the bucket
  low    = min low across the bucket
  close  = last candle's close in the bucket
  volume = sum of volumes in the bucket

CRITICAL for backtesting correctness: a higher-timeframe candle is only
"complete" once its time window has fully elapsed. Using an in-progress
bucket's aggregate as if it were a closed candle is a look-ahead bias bug —
this resampler marks incomplete trailing buckets accordingly.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

from backtesting.backtest_runner import OHLCV


TIMEFRAME_MINUTES = {
    "1M": 1, "3M": 3, "5M": 5, "15M": 15, "30M": 30,
    "1H": 60, "4H": 240, "1D": 1440,
}


@dataclass
class ResampledCandle:
    ohlcv: OHLCV
    is_complete: bool   # False for a trailing in-progress bucket — never trade off this


def resample(base_candles: list[OHLCV], base_timeframe: str, target_timeframe: str) -> list[ResampledCandle]:
    """
    base_candles: chronologically ordered, ts as epoch seconds.
    Groups candles into target_timeframe buckets aligned to timestamp
    boundaries (e.g. 15M buckets start at :00, :15, :30, :45).
    """
    if target_timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown target timeframe: {target_timeframe}")
    if base_timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown base timeframe: {base_timeframe}")

    base_min = TIMEFRAME_MINUTES[base_timeframe]
    target_min = TIMEFRAME_MINUTES[target_timeframe]
    if target_min < base_min:
        raise ValueError(f"Cannot resample UP from {base_timeframe} to a smaller {target_timeframe}")
    if target_min % base_min != 0:
        raise ValueError(f"{target_timeframe} is not an even multiple of base {base_timeframe} — "
                          f"aggregation boundaries would be inconsistent")

    if not base_candles:
        return []

    buckets: dict[int, list[OHLCV]] = {}
    for c in base_candles:
        bucket_key = _bucket_start(c.ts, target_min)
        buckets.setdefault(bucket_key, []).append(c)

    sorted_keys = sorted(buckets.keys())
    result = []
    last_base_ts = base_candles[-1].ts

    for idx, key in enumerate(sorted_keys):
        bucket_candles = buckets[key]
        agg = OHLCV(
            ts=key,
            open=bucket_candles[0].open,
            high=max(c.high for c in bucket_candles),
            low=min(c.low for c in bucket_candles),
            close=bucket_candles[-1].close,
            volume=sum(c.volume for c in bucket_candles),
        )
        # a bucket is complete only if the NEXT bucket's start time has been
        # reached by the base data — i.e. this window has fully elapsed
        bucket_end = key + target_min * 60
        is_complete = last_base_ts >= bucket_end or idx < len(sorted_keys) - 1
        result.append(ResampledCandle(ohlcv=agg, is_complete=is_complete))

    return result


def _bucket_start(ts_epoch: int, target_minutes: int) -> int:
    """Aligns a timestamp down to the start of its target-timeframe bucket."""
    bucket_seconds = target_minutes * 60
    return (ts_epoch // bucket_seconds) * bucket_seconds
