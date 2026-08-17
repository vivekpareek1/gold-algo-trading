"""
Tests for open-position persistence — a real incident: an open live trade
vanished entirely on a service restart (not just the P&L outcome, but
risk-engine's position tracking and the broker's margin/equity bookkeeping
went out of sync too). Fixed by persisting the full TradeManagerState plus
broker position/equity, restored on startup WITHOUT re-charging commission.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from trade_manager.trade_manager import TradeManager, TradeManagerState
from tests.test_backtest_runner import make_synthetic_trending_candles


def _paths(d):
    return (os.path.join(d, "trades.jsonl"), os.path.join(d, "candles.jsonl"),
            os.path.join(d, "open_position.json"))


def _engine(broker, trade_path=None, candle_path=None, open_pos_path=None):
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                               persistence_path=trade_path, candle_persistence_path=candle_path,
                               open_position_path=open_pos_path)


def test_no_open_position_file_on_first_run():
    with tempfile.TemporaryDirectory() as d:
        tp, cp, op = _paths(d)
        broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker.connect()
        engine = _engine(broker, tp, cp, op)
        assert engine.state.open_trade_manager is None


def test_manually_opened_trade_gets_persisted_to_disk():
    with tempfile.TemporaryDirectory() as d:
        tp, cp, op = _paths(d)
        broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker.connect()
        engine = _engine(broker, tp, cp, op)

        tm_state = TradeManagerState(direction="LONG", entry_price=155000.0,
                                        original_stop=154800.0, current_stop=154800.0,
                                        original_risk_points=200.0,
                                        target_1=155200.0, target_2=155400.0, target_3=155600.0)
        engine.state.open_trade_manager = TradeManager(engine.config, tm_state)
        engine.state.open_trade_lots = 2
        engine.state.open_trade_entry_regime = "TRENDING_UP"
        engine._persist_open_position()

        assert os.path.exists(op)


def test_full_restore_after_simulated_restart():
    """THE core scenario: open a trade, simulate a full restart (new
    engine, new broker instance), verify the position is fully restored —
    entry, stop, lots, trade state, AND broker equity/position bookkeeping."""
    with tempfile.TemporaryDirectory() as d:
        tp, cp, op = _paths(d)

        broker1 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker1.connect()
        e1 = _engine(broker1, tp, cp, op)
        candles = make_synthetic_trending_candles(n=300, drift=4.0, noise=8.0, seed=1)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

        assert e1.state.open_trade_manager is not None, \
            "Test fixture assumption: seed=1 must leave a trade open at the end"
        original_direction = e1.state.open_trade_manager.state.direction
        original_entry = e1.state.open_trade_manager.state.entry_price
        original_stop = e1.state.open_trade_manager.state.current_stop
        original_lots = e1.state.open_trade_lots
        original_equity = broker1.get_balance().equity_inr

        # simulate a full restart
        broker2 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker2.connect()
        e2 = _engine(broker2, tp, cp, op)

        assert e2.state.open_trade_manager is not None, "Position must be restored"
        assert e2.state.open_trade_manager.state.direction == original_direction
        assert e2.state.open_trade_manager.state.entry_price == original_entry
        assert e2.state.open_trade_manager.state.current_stop == original_stop
        assert e2.state.open_trade_lots == original_lots
        assert broker2.get_balance().equity_inr == original_equity, \
            "Broker equity must match exactly — a fresh broker would show " \
            "the starting equity, silently erasing the committed position"

        positions = broker2.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == original_lots * (1 if original_direction == "LONG" else -1)


def test_restored_trade_can_be_closed_normally():
    """A restored trade must be fully manageable afterward — trailing,
    exit detection, and real brokerage-adjusted P&L recording must all
    work exactly as if the restart never happened."""
    with tempfile.TemporaryDirectory() as d:
        tp, cp, op = _paths(d)

        broker1 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker1.connect()
        e1 = _engine(broker1, tp, cp, op)
        candles = make_synthetic_trending_candles(n=300, drift=4.0, noise=8.0, seed=1)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

        broker2 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker2.connect()
        e2 = _engine(broker2, tp, cp, op)
        assert e2.state.open_trade_manager is not None

        last_ts = candles[-1].ts
        entry_price = e2.state.open_trade_manager.state.entry_price
        for i in range(1, 100):
            price = entry_price - i * 3
            e2.on_tick(LiveTick(ts=last_ts + 300 + i * 300, open=price, high=price + 10,
                                  low=price - 15, close=price + 2, volume=100))
            if e2.state.open_trade_manager is None:
                break

        assert e2.state.open_trade_manager is None, "Trade must close normally after restore"
        assert len(e2.state.trade_log) >= 1
        closed = e2.state.trade_log[-1]
        assert "net_pnl_inr" in closed, "Closed trade must include real brokerage-adjusted P&L"
        assert not os.path.exists(op), \
            "The open-position file must be cleared once the trade closes"


def test_no_double_commission_on_restore():
    """Restoring must NOT re-charge entry commission/slippage — that
    already happened before the restart."""
    with tempfile.TemporaryDirectory() as d:
        tp, cp, op = _paths(d)
        broker1 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker1.connect()
        e1 = _engine(broker1, tp, cp, op)
        candles = make_synthetic_trending_candles(n=300, drift=4.0, noise=8.0, seed=1)
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))
        equity_before_restart = broker1.get_balance().equity_inr

        broker2 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker2.connect()
        e2 = _engine(broker2, tp, cp, op)

        assert broker2.get_balance().equity_inr == equity_before_restart, \
            "Equity must be EXACTLY restored, not reduced further by a second commission charge"


def test_stale_open_position_file_cleared_after_close_not_resurrected():
    """Once a trade closes normally, a LATER restart must not somehow
    resurrect it from a stale file."""
    with tempfile.TemporaryDirectory() as d:
        tp, cp, op = _paths(d)
        broker1 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker1.connect()
        e1 = _engine(broker1, tp, cp, op)
        candles = make_synthetic_trending_candles(n=300, drift=4.0, noise=8.0, seed=3)  # closes fully
        for c in candles:
            e1.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))
        assert e1.state.open_trade_manager is None, "Fixture assumption: seed=3 has no open trade at the end"
        assert not os.path.exists(op)

        broker2 = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker2.connect()
        e2 = _engine(broker2, tp, cp, op)
        assert e2.state.open_trade_manager is None, \
            "No open-position file exists — nothing should be resurrected"


def test_corrupted_open_position_file_does_not_crash_startup():
    with tempfile.TemporaryDirectory() as d:
        tp, cp, op = _paths(d)
        with open(op, "w") as f:
            f.write("not valid json {{{")

        broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker.connect()
        try:
            engine = _engine(broker, tp, cp, op)
        except Exception as e:
            assert False, f"A corrupted open-position file must not crash startup, got {type(e).__name__}: {e}"
        assert engine.state.open_trade_manager is None, \
            "A corrupted file should degrade to 'no open position', not crash or fabricate one"


if __name__ == "__main__":
    tests = [
        test_no_open_position_file_on_first_run,
        test_manually_opened_trade_gets_persisted_to_disk,
        test_full_restore_after_simulated_restart,
        test_restored_trade_can_be_closed_normally,
        test_no_double_commission_on_restore,
        test_stale_open_position_file_cleared_after_close_not_resurrected,
        test_corrupted_open_position_file_does_not_crash_startup,
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
