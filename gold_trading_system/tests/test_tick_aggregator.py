import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_data.tick_aggregator import TickAggregator, Tick


def epoch(minute_offset, second_offset=0):
    base = 1735689600  # aligned to a clean hour boundary
    return base + minute_offset * 60 + second_offset


def test_first_tick_never_emits_a_candle():
    agg = TickAggregator(interval_minutes=5)
    result = agg.add_tick(Tick(ts=epoch(0), ltp=100.0, volume=1000))
    assert result is None


def test_ticks_within_same_bucket_never_emit():
    agg = TickAggregator(interval_minutes=5)
    agg.add_tick(Tick(ts=epoch(0), ltp=100.0, volume=1000))
    r1 = agg.add_tick(Tick(ts=epoch(1), ltp=101.0, volume=1050))
    r2 = agg.add_tick(Tick(ts=epoch(4, 59), ltp=99.0, volume=1200))
    assert r1 is None
    assert r2 is None


def test_crossing_bucket_boundary_emits_correct_ohlc():
    agg = TickAggregator(interval_minutes=5)
    agg.add_tick(Tick(ts=epoch(0), ltp=100.0, volume=1000))     # open
    agg.add_tick(Tick(ts=epoch(1), ltp=105.0, volume=1100))     # high
    agg.add_tick(Tick(ts=epoch(2), ltp=98.0, volume=1150))      # low
    agg.add_tick(Tick(ts=epoch(4), ltp=102.0, volume=1300))     # close
    completed = agg.add_tick(Tick(ts=epoch(5), ltp=103.0, volume=1350))  # crosses into next bucket

    print(f"Completed candle: {completed}")
    assert completed is not None
    assert completed.open == 100.0
    assert completed.high == 105.0
    assert completed.low == 98.0
    assert completed.close == 102.0
    assert completed.ts == epoch(0)


def test_volume_derived_from_cumulative_correctly():
    """Angel One reports cumulative day volume — verify incremental extraction."""
    agg = TickAggregator(interval_minutes=5)
    agg.add_tick(Tick(ts=epoch(0), ltp=100.0, volume=5000))   # baseline, delta=0
    agg.add_tick(Tick(ts=epoch(1), ltp=101.0, volume=5100))   # +100
    agg.add_tick(Tick(ts=epoch(2), ltp=100.0, volume=5250))   # +150
    completed = agg.add_tick(Tick(ts=epoch(5), ltp=100.0, volume=5300))  # crosses boundary

    print(f"Aggregated volume: {completed.volume}")
    assert completed.volume == 250.0, f"Expected 100+150=250 incremental volume, got {completed.volume}"


def test_volume_reset_does_not_go_negative():
    """A new trading session resets cumulative volume to a small number —
    must not produce a huge negative delta."""
    agg = TickAggregator(interval_minutes=5)
    agg.add_tick(Tick(ts=epoch(0), ltp=100.0, volume=50000))   # end of a session, high cumulative
    agg.add_tick(Tick(ts=epoch(1), ltp=100.0, volume=10))      # new session started, volume reset
    completed = agg.add_tick(Tick(ts=epoch(5), ltp=100.0, volume=20))

    assert completed.volume >= 0, f"Volume must never be negative, got {completed.volume}"


def test_multiple_consecutive_candles():
    agg = TickAggregator(interval_minutes=5)
    completed_candles = []
    for i in range(20):
        result = agg.add_tick(Tick(ts=epoch(i), ltp=100.0 + i, volume=1000 + i * 10))
        if result:
            completed_candles.append(result)

    print(f"Completed candles: {len(completed_candles)}")
    # 20 minutes of 1-min-spaced ticks with a 5-min bucket -> ~3-4 completed candles
    assert len(completed_candles) >= 3
    # verify each completed candle's ts is a clean 5-min boundary
    for c in completed_candles:
        assert c.ts % 300 == 0, f"Candle timestamp {c.ts} is not on a 5-minute boundary"


def test_never_emits_the_in_progress_bucket():
    """Sanity: with only ticks inside one bucket, nothing should ever be emitted,
    matching the look-ahead guard used elsewhere in the system."""
    agg = TickAggregator(interval_minutes=5)
    results = []
    for i in range(4):
        results.append(agg.add_tick(Tick(ts=epoch(0, i * 10), ltp=100.0, volume=1000 + i)))
    assert all(r is None for r in results)


if __name__ == "__main__":
    tests = [
        test_first_tick_never_emits_a_candle,
        test_ticks_within_same_bucket_never_emit,
        test_crossing_bucket_boundary_emits_correct_ohlc,
        test_volume_derived_from_cumulative_correctly,
        test_volume_reset_does_not_go_negative,
        test_multiple_consecutive_candles,
        test_never_emits_the_in_progress_bucket,
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
