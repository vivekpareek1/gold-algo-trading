"""
Tests for a real, urgent bug found while reviewing with a live trade open:
a genuine large price move (rare, but real gaps happen) would be rejected
FOREVER by tick validation, since the reference price only updates on
accepted ticks. Worse: on_tick() returns early on rejection, meaning an
OPEN POSITION's stop-loss/trailing check would never run again either —
a permanent cascade here would silently freeze risk management on a live
trade. Same failure shape as an earlier data_loader cascade found during
backtesting.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from trade_manager.trade_manager import TradeManager, TradeManagerState


def _engine():
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                               persistence_path=None, candle_persistence_path=None)


def test_genuine_persistent_move_eventually_accepted():
    """A real gap that PERSISTS across multiple readings must eventually
    be accepted, not rejected forever."""
    engine = _engine()
    base_ts = 1735689600
    engine.on_tick(LiveTick(ts=base_ts, open=155000, high=155010, low=154990,
                              close=155000, volume=100))

    gap_price = 155000 * 1.20
    accepted = False
    for i in range(5):
        tick_price = gap_price + i
        engine.on_tick(LiveTick(ts=base_ts + 300 + i * 300, open=tick_price,
                                  high=tick_price + 5, low=tick_price - 5,
                                  close=tick_price, volume=100))
        if abs(engine.state.last_snapshot["ltp"] - tick_price) < 1:
            accepted = True
            break
    assert accepted, "A persistent genuine price move must eventually be accepted, " \
        "not rejected forever"


def test_recovery_happens_within_three_ticks():
    """Specifically verify the fix's threshold — must recover quickly
    (within a few ticks), not need dozens of readings."""
    engine = _engine()
    base_ts = 1735689600
    engine.on_tick(LiveTick(ts=base_ts, open=155000, high=155010, low=154990,
                              close=155000, volume=100))

    gap_price = 155000 * 1.20
    for i in range(3):
        engine.on_tick(LiveTick(ts=base_ts + 300 + i * 300, open=gap_price + i,
                                  high=gap_price + i + 5, low=gap_price + i - 5,
                                  close=gap_price + i, volume=100))

    assert abs(engine.state.last_snapshot["ltp"] - (gap_price + 2)) < 1, \
        "Must recover within exactly 3 consecutive readings"


def test_single_glitch_tick_still_correctly_rejected():
    """A ONE-OFF glitch (not persisting) must still be rejected — the fix
    must not make the system gullible to any single bad tick."""
    engine = _engine()
    base_ts = 1735689600
    engine.on_tick(LiveTick(ts=base_ts, open=155000, high=155010, low=154990,
                              close=155000, volume=100))

    # one wild glitch
    engine.on_tick(LiveTick(ts=base_ts + 300, open=999999, high=999999,
                              low=999999, close=999999, volume=100))
    assert engine.state.last_snapshot["ltp"] == 155000, \
        "A single glitch must be rejected, reference must stay at the last good price"

    # followed by NORMAL ticks (not persisting at the glitch level) —
    # confirms the glitch didn't leave the system in some bad state
    engine.on_tick(LiveTick(ts=base_ts + 600, open=155010, high=155020,
                              low=155000, close=155010, volume=100))
    assert engine.state.last_snapshot["ltp"] == 155010


def test_rejection_counter_resets_after_a_normal_tick():
    """If a jump-rejection streak is broken by a normal tick, the counter
    must reset — two SEPARATE one-off glitches (with a normal tick between
    them) must each independently be rejected, not accumulate toward the
    3-strike threshold."""
    engine = _engine()
    base_ts = 1735689600
    engine.on_tick(LiveTick(ts=base_ts, open=155000, high=155010, low=154990,
                              close=155000, volume=100))

    # glitch 1
    engine.on_tick(LiveTick(ts=base_ts + 300, open=999999, high=999999,
                              low=999999, close=999999, volume=100))
    assert engine.state.consecutive_price_jump_rejections == 1

    # normal tick breaks the streak
    engine.on_tick(LiveTick(ts=base_ts + 600, open=155010, high=155020,
                              low=155000, close=155010, volume=100))
    assert engine.state.consecutive_price_jump_rejections == 0

    # glitch 2 — must be treated as a FRESH first rejection, not a 2nd strike
    engine.on_tick(LiveTick(ts=base_ts + 900, open=999999, high=999999,
                              low=999999, close=999999, volume=100))
    assert engine.state.consecutive_price_jump_rejections == 1
    assert engine.state.last_snapshot["ltp"] == 155010, \
        "The second glitch must still be rejected, not force-accepted"


def test_open_position_stop_loss_still_checked_after_rejected_tick():
    """
    THE critical safety scenario: with a position OPEN, a rejected tick
    must not prevent the NEXT valid tick from correctly checking/updating
    the stop-loss. Manually sets up an open position to guarantee the
    scenario (rather than hoping synthetic data happens to open one).
    """
    engine = _engine()
    base_ts = 1735689600
    engine.on_tick(LiveTick(ts=base_ts, open=155000, high=155010, low=154990,
                              close=155000, volume=100))

    # manually place an open trade, as if a real entry had just happened
    tm_state = TradeManagerState(direction="LONG", entry_price=155000.0,
                                    original_stop=154800.0, current_stop=154800.0,
                                    original_risk_points=200.0,
                                    target_1=155200.0, target_2=155400.0, target_3=155600.0)
    engine.state.open_trade_manager = TradeManager(engine.config, tm_state)
    engine.state.open_trade_lots = 1
    engine.risk_engine.register_position_opened()

    # a glitch tick arrives — must be rejected, position must remain untouched
    engine.on_tick(LiveTick(ts=base_ts + 300, open=999999, high=999999,
                              low=999999, close=999999, volume=100))
    assert engine.state.open_trade_manager is not None, \
        "Position must still be open after a rejected glitch tick"

    # NOW a real tick arrives that should hit the stop-loss (low breaches 154800)
    engine.on_tick(LiveTick(ts=base_ts + 600, open=154850, high=154860,
                              low=154750, close=154800, volume=100))

    assert engine.state.open_trade_manager is None, \
        "The stop-loss must still be checked and the position closed on the " \
        "next VALID tick — a prior rejection must not have permanently frozen " \
        "trade management"
    assert len(engine.state.trade_log) == 1, \
        "The closed trade must be recorded"


if __name__ == "__main__":
    tests = [
        test_genuine_persistent_move_eventually_accepted,
        test_recovery_happens_within_three_ticks,
        test_single_glitch_tick_still_correctly_rejected,
        test_rejection_counter_resets_after_a_normal_tick,
        test_open_position_stop_loss_still_checked_after_rejected_tick,
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
