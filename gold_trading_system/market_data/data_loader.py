"""
Historical Data Loader + Quality Gate (Sprint 1 §22).
Ingests OHLCV candles from a CSV (Angel One's getCandleData() format, or any
similarly-shaped export) and runs them through the same quality checks the
live feed would face: timestamps, missing candles, duplicates, outliers,
bid/ask sanity is N/A for historical closes but OHLC sanity is checked.

If data quality fails on any row -> that row is REJECTED and logged, not
silently included. A backtest run on unvalidated data produces false
confidence — this gate exists specifically to prevent that.
"""
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta

from backtesting.backtest_runner import OHLCV
from situation_analysis.day_of_week_situational import DailyCandle, Weekday


@dataclass
class QualityIssue:
    row_index: int
    ts: str
    issue_type: str
    detail: str


@dataclass
class LoadResult:
    candles: list[OHLCV]
    rejected_count: int
    issues: list[QualityIssue]
    total_rows_seen: int


class DataQualityGate:
    def __init__(self, config, expected_interval_minutes: int | None = None):
        self.config = config
        self.expected_interval_minutes = expected_interval_minutes

    def load_csv(self, filepath: str, ts_format: str = "%Y-%m-%d %H:%M:%S") -> LoadResult:
        """
        Expected CSV columns: timestamp, open, high, low, close, volume
        (matches Angel One SmartAPI's getCandleData() array order, named).
        """
        raw_rows = []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(row)
        return self.load_rows(raw_rows, ts_format=ts_format)

    def load_rows(self, raw_rows: list[dict], ts_format: str = "%Y-%m-%d %H:%M:%S") -> LoadResult:
        issues: list[QualityIssue] = []
        accepted: list[OHLCV] = []
        prev_dt: datetime | None = None
        prev_close: float | None = None
        recent_closes: list[float] = []
        recent_pct_changes: list[float] = []

        for i, row in enumerate(raw_rows):
            try:
                dt = datetime.strptime(row["timestamp"], ts_format)
                o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                v = float(row["volume"])
            except (KeyError, ValueError) as e:
                issues.append(QualityIssue(i, row.get("timestamp", "?"), "PARSE_ERROR", str(e)))
                continue  # no valid dt/close from this row — nothing to advance references to

            # OHLC sanity: high must be the max, low must be the min
            if not (l <= o <= h and l <= c <= h):
                issues.append(QualityIssue(i, str(dt), "OHLC_SANITY_FAIL",
                                             f"O={o} H={h} L={l} C={c} violates L<=O,C<=H"))
                # BUGFIX: still advance prev_dt/prev_close to this row's real
                # values before moving on. A rejected row is still a REAL
                # observed data point in time — the NEXT row's gap/change
                # must be measured against what actually happened most
                # recently, not against a stale reference from several
                # candles ago. Freezing these references on rejection was
                # the root cause of a cascading false-rejection bug that
                # discarded most of a real, clean 6-month MCX dataset.
                prev_dt, prev_close = dt, c
                continue

            if v < 0:
                issues.append(QualityIssue(i, str(dt), "NEGATIVE_VOLUME", f"volume={v}"))
                prev_dt, prev_close = dt, c
                continue

            # duplicate timestamp check
            if prev_dt is not None and dt == prev_dt:
                issues.append(QualityIssue(i, str(dt), "DUPLICATE_TIMESTAMP",
                                             f"Same timestamp as previous row"))
                continue  # do NOT advance prev_dt here — it's already this exact timestamp

            # ordering check — must be strictly increasing (no look-ahead risk from unsorted input)
            if prev_dt is not None and dt < prev_dt:
                issues.append(QualityIssue(i, str(dt), "OUT_OF_ORDER",
                                             f"Timestamp {dt} is before previous {prev_dt}"))
                continue  # do NOT advance — this row is out of sequence, not a new reference point

            # missing candle gap check, if an expected interval is configured
            if self.expected_interval_minutes and prev_dt is not None:
                expected_gap = timedelta(minutes=self.expected_interval_minutes)
                actual_gap = dt - prev_dt
                # allow up to 3x the expected interval before flagging (accounts for
                # session breaks/weekends) — anything beyond that is a real gap
                if actual_gap > expected_gap * 3:
                    issues.append(QualityIssue(i, str(dt), "MISSING_CANDLES",
                                                 f"Gap of {actual_gap} exceeds tolerance"))
                    # NOT rejecting this row outright — the candle itself may be valid,
                    # just flagging that prior candles are missing. Caller decides.

            # outlier check — compares the candle-to-candle % CHANGE against
            # recent volatility, not the raw price LEVEL against a rolling mean.
            # BUGFIX (round 1): the original level-based check compared each
            # close to the mean of the last 10-50 ACCEPTED closes. During a
            # genuine sustained trend (e.g. gold moving from 130k to 167k over
            # 6 months — a real, normal price move), the rolling mean lagged
            # behind price, so trending candles kept reading as outliers.
            # BUGFIX (round 2): even after switching to % change, prev_close
            # was only updated on ACCEPTED rows — so once a row was rejected,
            # the next row's % change was measured against an increasingly
            # stale reference, making it look MORE anomalous, causing the same
            # kind of runaway rejection cascade through a different path.
            # Fixed by always advancing prev_dt/prev_close to the true most
            # recent observed candle regardless of accept/reject, while still
            # keeping the recent_pct_changes BASELINE window free of flagged
            # anomalies so genuine bad ticks don't get "normalized" into it.
            is_outlier = False
            if prev_close is not None and prev_close != 0:
                pct_change = abs((c - prev_close) / prev_close) * 100

                if len(recent_pct_changes) >= 10:
                    mean_pct = sum(recent_pct_changes) / len(recent_pct_changes)
                    variance_pct = sum((x - mean_pct) ** 2 for x in recent_pct_changes) / len(recent_pct_changes)
                    std_pct = variance_pct ** 0.5
                    if std_pct > 0 and pct_change > mean_pct + self.config.data_quality.outlier_std_threshold * std_pct:
                        issues.append(QualityIssue(i, str(dt), "OUTLIER",
                                                     f"{pct_change:.2f}% change is "
                                                     f"{(pct_change-mean_pct)/std_pct:.1f} std devs above recent norm"))
                        is_outlier = True

                if not is_outlier:
                    recent_pct_changes.append(pct_change)
                    if len(recent_pct_changes) > 50:
                        recent_pct_changes.pop(0)

            # always advance the reference point to this row — accepted or not
            prev_dt, prev_close = dt, c

            if is_outlier:
                continue

            accepted.append(OHLCV(ts=int(dt.timestamp()), open=o, high=h, low=l, close=c, volume=v))
            recent_closes.append(c)
            if len(recent_closes) > 50:
                recent_closes.pop(0)

        return LoadResult(
            candles=accepted,
            # BUGFIX: was len(issues), which double-counts rows that get an
            # informational MISSING_CANDLES flag (non-rejecting) AND then also
            # fail the OUTLIER check (rejecting) — inflating the reported
            # rejection count above the total row count. True rejected rows
            # is simply what didn't make it into `accepted`.
            rejected_count=len(raw_rows) - len(accepted),
            issues=issues,
            total_rows_seen=len(raw_rows),
        )

    def to_daily_candles(self, ohlcv_list: list[OHLCV]) -> list[DailyCandle]:
        """Converts OHLCV (epoch ts) into DailyCandle with weekday attached —
        feeds directly into DayOfWeekAnalyzer."""
        result = []
        for c in ohlcv_list:
            dt = datetime.fromtimestamp(c.ts)
            weekday = Weekday(dt.weekday())  # Python: Monday=0 ... matches our Weekday enum
            result.append(DailyCandle(
                ts_epoch_day=dt.toordinal(), weekday=weekday,
                open=c.open, high=c.high, low=c.low, close=c.close,
            ))
        return result
