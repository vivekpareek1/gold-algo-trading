"""
Tests for trade history persistence — the gap found in the final pre-launch
review: a service restart (crash, redeploy, reboot) silently wiped the
entire in-memory trade history. Not a financial risk in paper mode, but a
data-completeness one for a multi-week live paper run.
"""
import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from tests.test_backtest_runner import make_synthetic_trending_candles


def _engine_with_persistence(path):
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM", persistence_path=path,
                               candle_persistence_path=None, open_position_path=None)


def test_no_persistence_file_on_first_run():
    """A fresh path with no existing file must not error and starts empty."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "history.jsonl")
        engine = _engine_with_persistence(path)
        assert engine.state.trade_log == []


def test_closed_trade_is_written_to_disk():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "history.jsonl")
        engine = _engine_with_persistence(path)
        candles = make_synthetic_trending_candles(n=500, drift=3.0, noise=10.0, seed=7)
        for c in candles:
            engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                      close=c.close, volume=c.volume))

        if len(engine.state.trade_log) > 0:
            assert os.path.exists(path), "A file must be created once at least one trade closes"
            with open(path) as f:
                lines = [l for l in f if l.strip()]
            assert len(lines) == len(engine.state.trade_log), \
                "Every closed trade in memory must have a matching persisted line"
            # each line must be valid, parseable JSON
            for line in lines:
                json.loads(line)
        else:
            print("No trades closed in this synthetic run — nothing to verify persistence against")


def test_history_survives_a_simulated_restart():
    """
    THE core scenario this exists for: create an engine, close some trades,
    then create a BRAND NEW engine instance pointed at the same file
    (simulating a process restart) — the new engine must recover the prior
    trade history rather than starting from zero.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "history.jsonl")

        engine1 = _engine_with_persistence(path)
        candles = make_synthetic_trending_candles(n=800, drift=3.0, noise=10.0, seed=11)
        for c in candles:
            engine1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                       close=c.close, volume=c.volume))
        trades_before_restart = len(engine1.state.trade_log)
        print(f"Trades closed before simulated restart: {trades_before_restart}")

        # simulate the process dying and a fresh one starting — a NEW engine
        # instance, not a continuation, pointed at the SAME persistence file
        engine2 = _engine_with_persistence(path)

        assert len(engine2.state.trade_log) == trades_before_restart, \
            "A freshly constructed engine must recover the full trade history " \
            "from disk, matching what existed before the simulated restart"

        if trades_before_restart > 0:
            assert engine2.state.trade_log[0] == engine1.state.trade_log[0], \
                "Recovered trade records must match the originals exactly"


def test_persistence_disabled_when_path_is_none():
    """Tests and short-lived sessions must be able to opt out of touching disk."""
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(Settings(), broker, symbol="GOLDM", persistence_path=None, candle_persistence_path=None, open_position_path=None)
    candles = make_synthetic_trending_candles(n=500, drift=3.0, noise=10.0, seed=7)
    for c in candles:
        engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))
    # must not raise, and must simply not persist anything — nothing to
    # assert on disk since no path was given


def test_corrupted_persistence_file_does_not_crash_startup():
    """A malformed file (partial write, disk issue) must degrade gracefully,
    not prevent the engine from starting at all."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "history.jsonl")
        with open(path, "w") as f:
            f.write('{"valid": "json", "r_multiple": 0.5}\n')
            f.write('not valid json at all {{{\n')

        try:
            engine = _engine_with_persistence(path)
        except Exception as e:
            assert False, f"A corrupted persistence file must not crash startup, got {type(e).__name__}: {e}"
        assert len(engine.state.trade_log) == 1, \
            "The valid line before the corrupted one must still be recovered, " \
            "not discarded just because a LATER line was bad"


def test_persisted_trade_fields_match_in_memory_fields():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "history.jsonl")
        engine = _engine_with_persistence(path)
        candles = make_synthetic_trending_candles(n=800, drift=3.0, noise=10.0, seed=23)
        for c in candles:
            engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                      close=c.close, volume=c.volume))

        if engine.state.trade_log:
            with open(path) as f:
                first_persisted = json.loads(f.readline())
            first_in_memory = engine.state.trade_log[0]
            required_keys = {"entry_price", "exit_price", "direction", "r_multiple",
                              "exit_reason", "ts", "entry_regime"}
            assert required_keys.issubset(first_persisted.keys())
            assert first_persisted == first_in_memory


if __name__ == "__main__":
    tests = [
        test_no_persistence_file_on_first_run,
        test_closed_trade_is_written_to_disk,
        test_history_survives_a_simulated_restart,
        test_persistence_disabled_when_path_is_none,
        test_corrupted_persistence_file_does_not_crash_startup,
        test_persisted_trade_fields_match_in_memory_fields,
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
