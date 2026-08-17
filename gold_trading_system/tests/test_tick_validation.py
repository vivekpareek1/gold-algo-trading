"""
Tests for tick sanity validation — a real gap found in final pre-money
review: nothing validated incoming candle data before it reached
structure/indicator/signal logic. A single malformed feed message (zero
price, negative price, garbled OHLC, a decimal-place error) could
silently corrupt ATR/EMA for every subsequent candle, or even let a
position open at a nonsensical price. Rejected, not silently trusted.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from tests.test_backtest_runner import make_synthetic_trending_candles


def _engine():
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                               persistence_path=None, candle_persistence_path=None, open_position_path=None)


def _warm_up(engine, n=100, seed=1):
    candles = make_synthetic_trending_candles(n=n, drift=1.0, noise=10.0, seed=seed)
    for c in candles:
        engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))
    return candles[-1].ts


def test_zero_price_tick_rejected():
    engine = _engine()
    last_ts = _warm_up(engine)
    good_price = engine.state.last_snapshot["ltp"]
    tick_count_before = engine.state.tick_count

    snap = engine.on_tick(LiveTick(ts=last_ts + 300, open=0, high=0, low=0, close=0, volume=100))

    assert engine.state.tick_count == tick_count_before, \
        "A rejected tick must NOT advance tick_count — it never entered the pipeline"
    assert snap["ltp"] == good_price, \
        "The snapshot must show the last GOOD price, not the garbage 0"


def test_negative_price_tick_rejected():
    engine = _engine()
    last_ts = _warm_up(engine)
    tick_count_before = engine.state.tick_count
    engine.on_tick(LiveTick(ts=last_ts + 300, open=-100, high=-90, low=-110, close=-95, volume=100))
    assert engine.state.tick_count == tick_count_before


def test_high_less_than_low_rejected():
    """Malformed OHLC where high < low is physically impossible and must
    never reach the structure engine."""
    engine = _engine()
    last_ts = _warm_up(engine)
    tick_count_before = engine.state.tick_count
    engine.on_tick(LiveTick(ts=last_ts + 300, open=63000, high=62900, low=63100,
                              close=63000, volume=100))
    assert engine.state.tick_count == tick_count_before


def test_open_close_outside_high_low_rejected():
    engine = _engine()
    last_ts = _warm_up(engine)
    tick_count_before = engine.state.tick_count
    # close way above the stated high — internally inconsistent candle
    engine.on_tick(LiveTick(ts=last_ts + 300, open=63000, high=63050, low=62950,
                              close=64000, volume=100))
    assert engine.state.tick_count == tick_count_before


def test_negative_volume_rejected():
    engine = _engine()
    last_ts = _warm_up(engine)
    tick_count_before = engine.state.tick_count
    engine.on_tick(LiveTick(ts=last_ts + 300, open=63000, high=63050, low=62950,
                              close=63000, volume=-50))
    assert engine.state.tick_count == tick_count_before


def test_extreme_price_jump_rejected_as_likely_feed_corruption():
    """
    Gold does not move 15%+ in a single 5-minute bar in any real market
    condition. A jump that size is far more likely a feed/decimal error
    (e.g. a stray extra digit) than a genuine move, and must be rejected
    rather than treated as real and used to corrupt ATR/EMA.
    """
    engine = _engine()
    last_ts = _warm_up(engine)
    good_price = engine.state.last_snapshot["ltp"]
    tick_count_before = engine.state.tick_count

    # a 10x price spike, classic symptom of a decimal-place bug
    spike_price = good_price * 10
    engine.on_tick(LiveTick(ts=last_ts + 300, open=spike_price, high=spike_price * 1.001,
                              low=spike_price * 0.999, close=spike_price, volume=100))

    assert engine.state.tick_count == tick_count_before
    assert engine.state.last_snapshot["ltp"] == good_price


def test_normal_price_movement_never_falsely_rejected():
    """The validation must not be so strict that ordinary volatile trading
    gets rejected — only genuinely impossible/absurd data."""
    engine = _engine()
    last_ts = _warm_up(engine)
    tick_count_before = engine.state.tick_count

    # a realistic, if volatile, 5-minute move: ~1% (gold CAN move this much
    # on a genuine news event, well under the 15% corruption threshold)
    good_price = engine.state.last_snapshot["ltp"]
    moved_price = good_price * 1.01
    engine.on_tick(LiveTick(ts=last_ts + 300, open=good_price, high=moved_price * 1.001,
                              low=good_price * 0.999, close=moved_price, volume=1000))

    assert engine.state.tick_count == tick_count_before + 1, \
        "A realistic, if volatile, price move must NOT be rejected"


def test_first_ever_tick_has_no_prior_price_to_compare_against():
    """The very first tick an engine ever sees can't be jump-checked against
    a prior price (there isn't one) — it must still be accepted if otherwise valid."""
    engine = _engine()
    snap = engine.on_tick(LiveTick(ts=1735689600, open=63000, high=63050,
                                     low=62950, close=63010, volume=1000))
    assert engine.state.tick_count == 1
    assert snap["ltp"] == 63010


def test_rejected_tick_does_not_crash_the_engine():
    """A rejected tick must degrade gracefully — never raise, since that
    would kill the feed thread and stop all trading."""
    engine = _engine()
    _warm_up(engine)
    try:
        engine.on_tick(LiveTick(ts=999999999, open=float('nan'), high=float('nan'),
                                  low=float('nan'), close=float('nan'), volume=0))
    except Exception as e:
        assert False, f"A malformed tick must never crash the engine, got {type(e).__name__}: {e}"


def test_candle_gap_detected_blocks_new_entry_evaluation():
    """
    Regression test for a real gap: risk_engine's stale-data protection
    existed but was hardcoded to never engage. If the feed drops for an
    extended period and reconnects, the candle immediately after the gap
    must not open a fresh position — its indicators were built on an
    interrupted stream. (Signal evaluation/logging still happens regardless
    of this veto — by design, for visibility into what the confluence
    engine would have recommended — only actual trade EXECUTION is gated.)
    """
    engine = _engine()
    last_ts = _warm_up(engine, n=100)
    assert engine.state.open_trade_manager is None

    gap_ts = last_ts + 1800
    last_close = engine.state.last_snapshot["ltp"]
    engine.on_tick(LiveTick(ts=gap_ts, open=last_close + 5, high=last_close + 15,
                              low=last_close - 5, close=last_close + 10, volume=1000))

    assert engine.state.open_trade_manager is None, \
        "No position should open on the candle immediately following a detected data gap"


def test_normal_cadence_does_not_falsely_trigger_gap_detection():
    """Ordinary 5-minute-spaced candles must never be treated as a gap."""
    engine = _engine()
    last_ts = _warm_up(engine, n=100)
    signals_before = len(engine.state.signal_log)

    last_close = engine.state.last_snapshot["ltp"]
    engine.on_tick(LiveTick(ts=last_ts + 300, open=last_close, high=last_close + 20,
                              low=last_close - 20, close=last_close + 5, volume=1000))

    signals_after = len(engine.state.signal_log)
    assert signals_after == signals_before + 1, \
        "Normal 5-minute cadence must NOT be treated as a data gap"


if __name__ == "__main__":
    tests = [
        test_zero_price_tick_rejected,
        test_negative_price_tick_rejected,
        test_high_less_than_low_rejected,
        test_open_close_outside_high_low_rejected,
        test_negative_volume_rejected,
        test_extreme_price_jump_rejected_as_likely_feed_corruption,
        test_normal_price_movement_never_falsely_rejected,
        test_first_ever_tick_has_no_prior_price_to_compare_against,
        test_rejected_tick_does_not_crash_the_engine,
        test_candle_gap_detected_blocks_new_entry_evaluation,
        test_normal_cadence_does_not_falsely_trigger_gap_detection,
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
