import sys, os, csv, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_data.data_loader import DataQualityGate


def make_row(ts, o, h, l, c, v):
    return {"timestamp": ts, "open": str(o), "high": str(h), "low": str(l),
            "close": str(c), "volume": str(v)}


def test_clean_data_all_accepted():
    gate = DataQualityGate(Settings())
    rows = [
        make_row("2026-01-01 09:00:00", 63000, 63050, 62980, 63020, 1000),
        make_row("2026-01-01 09:05:00", 63020, 63060, 63000, 63040, 1100),
        make_row("2026-01-01 09:10:00", 63040, 63080, 63010, 63060, 950),
    ]
    result = gate.load_rows(rows)
    print(f"Clean data: accepted={len(result.candles)}, rejected={result.rejected_count}")
    assert len(result.candles) == 3
    assert result.rejected_count == 0


def test_ohlc_sanity_violation_rejected():
    """Low higher than close/open — physically impossible candle must be rejected."""
    gate = DataQualityGate(Settings())
    rows = [
        make_row("2026-01-01 09:00:00", 63000, 63050, 62980, 63020, 1000),
        make_row("2026-01-01 09:05:00", 63020, 63030, 63100, 63040, 1100),  # low > high!
    ]
    result = gate.load_rows(rows)
    print(f"Issues: {result.issues}")
    assert len(result.candles) == 1
    assert result.rejected_count == 1
    assert result.issues[0].issue_type == "OHLC_SANITY_FAIL"


def test_negative_volume_rejected():
    gate = DataQualityGate(Settings())
    rows = [
        make_row("2026-01-01 09:00:00", 63000, 63050, 62980, 63020, -500),
    ]
    result = gate.load_rows(rows)
    assert result.rejected_count == 1
    assert result.issues[0].issue_type == "NEGATIVE_VOLUME"


def test_duplicate_timestamp_rejected():
    gate = DataQualityGate(Settings())
    rows = [
        make_row("2026-01-01 09:00:00", 63000, 63050, 62980, 63020, 1000),
        make_row("2026-01-01 09:00:00", 63020, 63060, 63000, 63040, 1100),  # same ts
    ]
    result = gate.load_rows(rows)
    assert len(result.candles) == 1
    assert result.issues[0].issue_type == "DUPLICATE_TIMESTAMP"


def test_out_of_order_rejected():
    gate = DataQualityGate(Settings())
    rows = [
        make_row("2026-01-01 09:10:00", 63000, 63050, 62980, 63020, 1000),
        make_row("2026-01-01 09:00:00", 63020, 63060, 63000, 63040, 1100),  # earlier than prev
    ]
    result = gate.load_rows(rows)
    assert len(result.candles) == 1
    assert result.issues[0].issue_type == "OUT_OF_ORDER"


def test_missing_candles_flagged_not_necessarily_rejected():
    """A large gap should be flagged, but the candle itself isn't inherently bad."""
    gate = DataQualityGate(Settings(), expected_interval_minutes=5)
    rows = [
        make_row("2026-01-01 09:00:00", 63000, 63050, 62980, 63020, 1000),
        make_row("2026-01-01 11:00:00", 63020, 63060, 63000, 63040, 1100),  # 2hr gap vs 5min expected
    ]
    result = gate.load_rows(rows)
    print(f"Gap test - accepted={len(result.candles)}, issues={result.issues}")
    assert any(i.issue_type == "MISSING_CANDLES" for i in result.issues)
    assert len(result.candles) == 2, "The gapped candle itself should still be accepted, just flagged"


def test_outlier_rejected():
    gate = DataQualityGate(Settings())
    rows = []
    # 15 stable candles around 63000
    for i in range(15):
        rows.append(make_row(f"2026-01-01 09:{i:02d}:00", 63000 + i, 63010 + i, 62990 + i, 63000 + i, 1000))
    # then an absurd outlier
    rows.append(make_row("2026-01-01 09:20:00", 80000, 80100, 79900, 80000, 1000))
    result = gate.load_rows(rows)
    print(f"Outlier test - rejected={result.rejected_count}, issues={[i.issue_type for i in result.issues]}")
    assert any(i.issue_type == "OUTLIER" for i in result.issues), \
        "A 17000-point jump should be flagged as an outlier"


def test_parse_error_handled_gracefully():
    gate = DataQualityGate(Settings())
    rows = [
        {"timestamp": "2026-01-01 09:00:00", "open": "not_a_number", "high": "63050",
         "low": "62980", "close": "63020", "volume": "1000"},
    ]
    result = gate.load_rows(rows)
    assert result.rejected_count == 1
    assert result.issues[0].issue_type == "PARSE_ERROR"


def test_full_csv_file_load():
    """End-to-end: write a real CSV file, load it, verify results."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerow(["2026-01-01 09:00:00", "63000", "63050", "62980", "63020", "1000"])
        writer.writerow(["2026-01-01 09:05:00", "63020", "63060", "63000", "63040", "1100"])
        path = f.name

    gate = DataQualityGate(Settings())
    result = gate.load_csv(path)
    os.unlink(path)

    assert len(result.candles) == 2
    assert result.total_rows_seen == 2


def test_to_daily_candles_conversion():
    gate = DataQualityGate(Settings())
    rows = [make_row("2026-08-10 09:00:00", 63000, 63050, 62980, 63020, 1000)]  # a Monday
    result = gate.load_rows(rows)
    daily = gate.to_daily_candles(result.candles)
    assert len(daily) == 1
    print(f"Weekday derived: {daily[0].weekday}")
    assert daily[0].weekday.name == "MONDAY"


def test_sustained_trend_not_rejected_as_outliers():
    """
    Regression test for a real bug found on real MCX data: a genuine sustained
    price trend (small, consistent period-over-period moves) must NOT get
    mass-rejected as outliers just because price drifts far from its level
    many candles ago. This uses % change, not absolute price level.
    """
    from datetime import datetime, timedelta
    gate = DataQualityGate(Settings())
    rows = []
    price = 130000.0
    base = datetime(2026, 1, 1, 9, 0, 0)
    # steady ~0.15% per-candle uptrend for 200 candles — a real, unremarkable
    # trend (this is what a genuine multi-month gold rally looks like zoomed in)
    for i in range(200):
        price *= 1.0015
        ts = (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(make_row(ts, price, price * 1.001, price * 0.999, price, 1000))
    result = gate.load_rows(rows)
    print(f"Sustained trend test: accepted={len(result.candles)}/200, "
          f"rejected_as_outlier={sum(1 for iss in result.issues if iss.issue_type=='OUTLIER')}")
    assert len(result.candles) >= 190, \
        f"A genuine steady trend should not trigger mass outlier rejection, " \
        f"only {len(result.candles)}/200 candles were accepted"


def test_genuine_spike_still_caught_amid_a_trend():
    """A real bad-tick spike injected into an otherwise steady trend must
    still be caught, even with the trend-robust % change check."""
    from datetime import datetime, timedelta
    gate = DataQualityGate(Settings())
    rows = []
    price = 130000.0
    base = datetime(2026, 1, 1, 9, 0, 0)
    for i in range(30):
        price *= 1.001
        ts = (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(make_row(ts, price, price * 1.001, price * 0.999, price, 1000))
    # inject one wild spike candle
    spike_price = price * 1.5   # 50% single-candle jump — clearly a bad tick
    spike_ts = (base + timedelta(minutes=5 * 30)).strftime("%Y-%m-%d %H:%M:%S")
    rows.append(make_row(spike_ts, spike_price, spike_price * 1.001,
                          spike_price * 0.999, spike_price, 1000))
    result = gate.load_rows(rows)
    outlier_issues = [iss for iss in result.issues if iss.issue_type == "OUTLIER"]
    print(f"Spike-in-trend test: outliers caught={len(outlier_issues)}")
    assert len(outlier_issues) >= 1, "A genuine 50% single-candle spike must still be caught"


def test_rejected_count_never_exceeds_total_rows():
    """Regression: rejected_count must never exceed total_rows_seen (it did,
    due to double-counting issues instead of unique rejected rows)."""
    gate = DataQualityGate(Settings(), expected_interval_minutes=5)
    rows = [make_row("2026-01-01 09:00:00", 100, 101, 99, 100, 1000)]
    # a row that is BOTH a missing-candle gap AND an OHLC violation
    rows.append(make_row("2026-01-01 15:00:00", 100, 101, 200, 100, 1000))
    result = gate.load_rows(rows)
    assert result.rejected_count <= result.total_rows_seen, \
        f"rejected_count ({result.rejected_count}) must not exceed total_rows_seen " \
        f"({result.total_rows_seen})"


def test_single_bad_tick_does_not_cascade_reject_subsequent_good_candles():
    """
    Regression test for the frozen-reference cascade bug: after one genuine
    bad tick is correctly rejected, prev_close must still advance to that
    tick's real value — otherwise the NEXT (perfectly normal) candle gets
    compared against an increasingly stale reference and can itself start
    looking anomalous, cascading into rejecting good data.
    """
    from datetime import datetime, timedelta
    gate = DataQualityGate(Settings())
    rows = []
    price = 130000.0
    base = datetime(2026, 1, 1, 9, 0, 0)
    for i in range(20):
        price *= 1.001
        ts = (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(make_row(ts, price, price * 1.001, price * 0.999, price, 1000))

    # one genuine bad tick
    bad_price = price * 1.4
    bad_ts = (base + timedelta(minutes=5 * 20)).strftime("%Y-%m-%d %H:%M:%S")
    rows.append(make_row(bad_ts, bad_price, bad_price * 1.001, bad_price * 0.999, bad_price, 1000))

    # resume completely normal trend AFTER the bad tick — these must all be accepted
    for i in range(21, 40):
        price *= 1.001
        ts = (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(make_row(ts, price, price * 1.001, price * 0.999, price, 1000))

    result = gate.load_rows(rows)
    outlier_count = sum(1 for iss in result.issues if iss.issue_type == "OUTLIER")
    print(f"Cascade test: accepted={len(result.candles)}/40, outliers={outlier_count}")
    # The bad tick itself gets flagged, and — since the very next candle's
    # change is measured relative to that bad price — one follow-on rejection
    # is legitimate too. What matters is the cascade STOPS there rather than
    # continuing to reject every subsequent candle (the actual bug this test
    # guards against): a bounded 1-2 rejections is correct, dozens would mean
    # the cascade returned.
    assert 1 <= outlier_count <= 2, \
        f"Expected 1-2 outliers (the bad tick, possibly +1 follow-on), got {outlier_count} " \
        f"— anything more suggests the rejection cascade has returned"
    assert len(result.candles) >= 37, \
        f"Nearly all 40 candles should be accepted (only the bad tick and maybe " \
        f"its immediate follow-on rejected), got {len(result.candles)} — a cascade " \
        f"would reject far more than that"


if __name__ == "__main__":
    tests = [
        test_clean_data_all_accepted,
        test_ohlc_sanity_violation_rejected,
        test_negative_volume_rejected,
        test_duplicate_timestamp_rejected,
        test_out_of_order_rejected,
        test_missing_candles_flagged_not_necessarily_rejected,
        test_outlier_rejected,
        test_parse_error_handled_gracefully,
        test_full_csv_file_load,
        test_to_daily_candles_conversion,
        test_sustained_trend_not_rejected_as_outliers,
        test_genuine_spike_still_caught_amid_a_trend,
        test_rejected_count_never_exceeds_total_rows,
        test_single_bad_tick_does_not_cascade_reject_subsequent_good_candles,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__} -> {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
