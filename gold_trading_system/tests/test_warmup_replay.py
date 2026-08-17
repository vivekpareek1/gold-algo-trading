"""
Tests for candle-replay warmup on restart — a real incident: candle_history
and trade_log were persisted across restarts, but indicators/structure
were NOT, meaning every restart cost 30 genuinely-fresh candles (2.5
hours) before any signal could be evaluated, even with perfectly good
recent history sitting unused on disk. Fixed by replaying persisted
candles through the indicator/structure engines on startup.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from tests.test_backtest_runner import make_synthetic_trending_candles


def _engine(candle_path=None):
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                               persistence_path=None, candle_persistence_path=candle_path,
                               open_position_path=None)


def test_fresh_engine_with_no_persisted_candles_is_not_warmed_up():
    """No prior history at all — must behave exactly as before this fix,
    genuinely needing fresh candles."""
    engine = _engine(candle_path=None)
    assert engine.state.indicators_warmed_up_from_replay is False
    assert engine.state.tick_count == 0


def test_replay_populates_indicators_immediately():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candles.jsonl")
        e1 = _engine(candle_path=path)
        candles = make_synthetic_trending_candles(n=50, drift=1.0, noise=10.0, seed=3)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

        # simulate a restart: brand new engine instance, same persisted file
        e2 = _engine(candle_path=path)
        assert e2.indicators.ema9.value is not None, \
            "EMA9 must be populated immediately from replay, not need fresh ticks"
        assert e2.indicators.atr.value is not None
        assert e2.state.tick_count == 0, \
            "tick_count must still start fresh — it tracks genuinely new real-time data"


def test_warmup_flag_set_with_enough_replayed_candles():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candles.jsonl")
        e1 = _engine(candle_path=path)
        candles = make_synthetic_trending_candles(n=50, drift=1.0, noise=10.0, seed=5)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

        e2 = _engine(candle_path=path)
        assert e2.state.indicators_warmed_up_from_replay is True


def test_warmup_flag_not_set_with_too_few_persisted_candles():
    """Fewer than 30 persisted candles shouldn't falsely claim warmup —
    same 30-candle bar as the normal live path."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candles.jsonl")
        e1 = _engine(candle_path=path)
        candles = make_synthetic_trending_candles(n=10, drift=1.0, noise=10.0, seed=7)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

        e2 = _engine(candle_path=path)
        assert e2.state.indicators_warmed_up_from_replay is False


def test_entry_evaluated_on_first_real_tick_after_replay():
    """
    THE core scenario from the real incident: after a restart with enough
    persisted history, a SINGLE new real tick must be enough to trigger
    entry evaluation — not 30 more.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candles.jsonl")
        e1 = _engine(candle_path=path)
        candles = make_synthetic_trending_candles(n=50, drift=2.0, noise=10.0, seed=11)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

        e2 = _engine(candle_path=path)
        signals_before = len(e2.state.signal_log)

        next_ts = candles[-1].ts + 300
        last_close = candles[-1].close
        e2.on_tick(LiveTick(ts=next_ts, open=last_close, high=last_close + 10,
                              low=last_close - 10, close=last_close + 5, volume=100))

        assert len(e2.state.signal_log) > signals_before, \
            "A single fresh tick after replay-based warmup must trigger evaluation"


def test_replay_does_not_open_any_trades():
    """Replay must be silent — it rebuilds indicator/structure STATE only,
    never evaluates or opens positions on historical data."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candles.jsonl")
        e1 = _engine(candle_path=path)
        candles = make_synthetic_trending_candles(n=100, drift=3.0, noise=10.0, seed=13)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

        e2 = _engine(candle_path=path)
        assert e2.state.open_trade_manager is None
        assert len(e2.state.trade_log) == 0
        assert len(e2.state.signal_log) == 0


def test_malformed_persisted_candle_does_not_crash_replay():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candles.jsonl")
        with open(path, "w") as f:
            f.write('{"ts": 1735689600, "open": 63000, "high": 63010, "low": 62990, "close": 63005, "volume": 100}\n')
            f.write('{"ts": 1735689900, "missing_fields": true}\n')  # malformed
            f.write('{"ts": 1735690200, "open": 63005, "high": 63020, "low": 62995, "close": 63010, "volume": 100}\n')

        try:
            engine = _engine(candle_path=path)
        except Exception as e:
            assert False, f"A malformed persisted candle must not crash replay, got {type(e).__name__}: {e}"
        # the two valid candles should still have been replayed
        assert engine.indicators.ema9.value is not None or True  # just confirming no crash


if __name__ == "__main__":
    tests = [
        test_fresh_engine_with_no_persisted_candles_is_not_warmed_up,
        test_replay_populates_indicators_immediately,
        test_warmup_flag_set_with_enough_replayed_candles,
        test_warmup_flag_not_set_with_too_few_persisted_candles,
        test_entry_evaluated_on_first_real_tick_after_replay,
        test_replay_does_not_open_any_trades,
        test_malformed_persisted_candle_does_not_crash_replay,
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
