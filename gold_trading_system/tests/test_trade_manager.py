import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from trade_manager.trade_manager import (
    TradeManager, TradeManagerState, TradeState, ExitReason
)


def make_long_trade(entry=63000, stop=62900, t1=63200, t2=63400, t3=63600):
    state = TradeManagerState(
        direction="LONG", entry_price=entry, original_stop=stop, current_stop=stop,
        original_risk_points=entry - stop, target_1=t1, target_2=t2, target_3=t3,
    )
    return TradeManager(Settings(), state)


def test_partial_booking_at_1R():
    tm = make_long_trade(entry=63000, stop=62900)  # risk = 100pts
    actions = tm.check_partial_booking(current_price=63100)  # +1R
    print(f"Actions at 1R: {actions}, state: {tm.state.trade_state}")
    assert tm.state.booked_at_1R == True
    assert tm.state.quantity_remaining_pct == 75.0  # 100 - 25 (default config)
    assert tm.state.trade_state == TradeState.PROFITABLE


def test_partial_booking_at_target1_moves_to_breakeven():
    tm = make_long_trade(entry=63000, stop=62900, t1=63200)
    tm.check_partial_booking(current_price=63100)   # 1R first
    actions = tm.check_partial_booking(current_price=63200)  # target 1
    print(f"Actions at T1: {actions}, stop now: {tm.state.current_stop}")
    assert tm.state.booked_at_t1 == True
    assert tm.state.current_stop == 63000, \
        f"Stop should move to breakeven (entry=63000), got {tm.state.current_stop}"
    assert tm.state.trade_state == TradeState.BREAKEVEN_PROTECTED


def test_never_books_same_level_twice():
    tm = make_long_trade(entry=63000, stop=62900)
    tm.check_partial_booking(current_price=63100)
    qty_after_first = tm.state.quantity_remaining_pct
    tm.check_partial_booking(current_price=63150)  # still above 1R, should NOT re-book
    assert tm.state.quantity_remaining_pct == qty_after_first, \
        "Should not book the same 1R level twice"


def test_stop_never_widens_long():
    """THE critical safety rule: stop must never move backward for a LONG."""
    tm = make_long_trade(entry=63000, stop=62900)
    tm.state.current_stop = 63000  # simulate already moved to breakeven

    # attempt to apply a WORSE (lower) stop via trailing update
    result = tm.update_trailing_stop(
        current_price=63050, ema9=62950, ema21=62900, ema50=62800,  # all below current stop
        atr=10, momentum_health="STRONG", structure_broke_against=False,
    )
    print(f"Attempted widen result: {result}, actual current_stop: {tm.state.current_stop}")
    assert tm.state.current_stop == 63000, \
        f"Stop must NOT have widened from 63000, got {tm.state.current_stop}"


def test_stop_never_widens_short():
    state = TradeManagerState(
        direction="SHORT", entry_price=63000, original_stop=63100, current_stop=63100,
        original_risk_points=100, target_1=62800, target_2=62600, target_3=62400,
    )
    tm = TradeManager(Settings(), state)
    tm.state.current_stop = 63000  # already tightened

    result = tm.update_trailing_stop(
        current_price=62950, ema9=63050, ema21=63080, ema50=63150,  # all above current stop (worse)
        atr=10, momentum_health="STRONG", structure_broke_against=False,
    )
    assert tm.state.current_stop == 63000, \
        f"SHORT stop must NOT have widened from 63000, got {tm.state.current_stop}"


def test_trailing_tightens_on_strong_momentum():
    tm = make_long_trade(entry=63000, stop=62900)
    tm.state.current_stop = 62950
    result = tm.update_trailing_stop(
        current_price=63100, ema9=63020, ema21=62980, ema50=62900,  # ema9 improves the stop
        atr=10, momentum_health="STRONG", structure_broke_against=False,
    )
    print(f"Trail result: {result}")
    assert result is not None
    assert tm.state.current_stop == 63020
    assert result.method_used == "EMA9"


def test_exits_immediately_on_structure_break():
    tm = make_long_trade(entry=63000, stop=62900)
    result = tm.update_trailing_stop(
        current_price=63000, ema9=63000, ema21=63000, ema50=63000,
        atr=10, momentum_health="STRONG", structure_broke_against=True,
    )
    assert result is None
    assert tm.state.trade_state == TradeState.EXITED
    assert tm.state.exit_reason == ExitReason.STRUCTURE_BREAK


def test_exits_on_dead_momentum():
    tm = make_long_trade(entry=63000, stop=62900)
    tm.update_trailing_stop(
        current_price=63100, ema9=63000, ema21=63000, ema50=63000,
        atr=10, momentum_health="DEAD", structure_broke_against=False,
    )
    assert tm.state.trade_state == TradeState.EXITED
    assert tm.state.exit_reason == ExitReason.MOMENTUM_DECAY


def test_stop_loss_hit_closes_trade():
    tm = make_long_trade(entry=63000, stop=62900)
    hit = tm.check_stop_hit(current_price=62890)
    assert hit == True
    assert tm.state.trade_state == TradeState.EXITED
    assert tm.state.exit_reason == ExitReason.STOP_LOSS_HIT


def test_full_lifecycle_long_trade():
    """End-to-end: entry -> 1R -> T1 -> breakeven -> T2 -> runner -> momentum decay exit."""
    tm = make_long_trade(entry=63000, stop=62900, t1=63200, t2=63400)
    history = []

    tm.check_partial_booking(63100)  # 1R
    history.append(tm.state.trade_state)
    tm.check_partial_booking(63200)  # T1 -> breakeven
    history.append(tm.state.trade_state)
    tm.check_partial_booking(63400)  # T2 -> runner
    history.append(tm.state.trade_state)

    print(f"Lifecycle states: {history}")
    assert history == [TradeState.PROFITABLE, TradeState.BREAKEVEN_PROTECTED, TradeState.TRAILING_RUNNER]

    tm.update_trailing_stop(63450, ema9=63400, ema21=63350, ema50=63200,
                              atr=10, momentum_health="DEAD", structure_broke_against=False)
    assert tm.state.trade_state == TradeState.EXITED
    assert tm.state.exit_reason == ExitReason.MOMENTUM_DECAY
    assert len(tm.state.state_history) >= 4, "Full state transition history should be logged"


if __name__ == "__main__":
    tests = [
        test_partial_booking_at_1R,
        test_partial_booking_at_target1_moves_to_breakeven,
        test_never_books_same_level_twice,
        test_stop_never_widens_long,
        test_stop_never_widens_short,
        test_trailing_tightens_on_strong_momentum,
        test_exits_immediately_on_structure_break,
        test_exits_on_dead_momentum,
        test_stop_loss_hit_closes_trade,
        test_full_lifecycle_long_trade,
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
