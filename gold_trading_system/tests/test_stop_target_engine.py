import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from target_engine.stop_target_engine import StopLossEngine, TargetEngine, StopMethod


def test_structure_stop_chosen_when_tightest_and_within_cap():
    engine = StopLossEngine(Settings())
    # entry 63000, swing low at 62950 (50pt away + buffer), ATR=100 -> ATR stop would be 150pt
    result = engine.evaluate(direction="LONG", entry_price=63000, atr=100,
                               nearest_swing_low=62950, nearest_swing_high=None)
    print(f"Structure-tightest result: {result}")
    assert result.approved == True
    assert "STRUCTURE_SWING" in result.reason
    assert result.distance_points < 100, "Structure stop should be tighter than the 150pt ATR stop"


def test_atr_stop_chosen_when_no_structure_data():
    engine = StopLossEngine(Settings())
    result = engine.evaluate(direction="LONG", entry_price=63000, atr=50,
                               nearest_swing_low=None, nearest_swing_high=None)
    assert result.approved == True
    assert "ATR" in result.reason or "VOLATILITY_BUFFER" in result.reason


def test_stop_rejected_when_all_candidates_exceed_cap():
    """If even the tightest candidate exceeds max_stop_distance_points, must reject, not force-tighten."""
    settings = Settings()
    engine = StopLossEngine(settings)
    # swing extremely far away, huge ATR -> everything exceeds the 200pt cap
    result = engine.evaluate(direction="LONG", entry_price=63000, atr=500,
                               nearest_swing_low=61000, nearest_swing_high=None)
    print(f"Over-cap rejection: {result.rejection_reason}")
    assert result.approved == False
    assert result.price is None
    assert "exceeds" in result.rejection_reason.lower()


def test_short_direction_stop_placed_above_entry():
    engine = StopLossEngine(Settings())
    result = engine.evaluate(direction="SHORT", entry_price=63000, atr=50,
                               nearest_swing_low=None, nearest_swing_high=63080)
    assert result.approved == True
    assert result.price > 63000, "SHORT stop must be placed ABOVE entry"


def test_long_direction_stop_placed_below_entry():
    engine = StopLossEngine(Settings())
    result = engine.evaluate(direction="LONG", entry_price=63000, atr=50,
                               nearest_swing_low=62950, nearest_swing_high=None)
    assert result.price < 63000, "LONG stop must be placed BELOW entry"


def test_all_candidates_returned_for_transparency():
    engine = StopLossEngine(Settings())
    result = engine.evaluate(direction="LONG", entry_price=63000, atr=50,
                               nearest_swing_low=62960, nearest_swing_high=None)
    methods = {c.method for c in result.all_candidates}
    assert StopMethod.STRUCTURE_SWING in methods
    assert StopMethod.ATR in methods
    assert StopMethod.VOLATILITY_BUFFER in methods


# ---------- Target Engine tests ----------

def test_basic_r_multiple_targets():
    engine = TargetEngine(Settings())
    result = engine.calculate(direction="LONG", entry_price=63000, stop_price=62900,
                                nearest_resistance=None, nearest_support=None, atr=50)
    print(f"Basic targets: T1={result.target_1}, T2={result.target_2}, T3={result.target_3}")
    assert result.target_1 == 63100  # 1R
    assert result.target_2 == 63200  # 2R
    assert result.target_3 == 63300  # 3R


def test_short_targets_go_downward():
    engine = TargetEngine(Settings())
    result = engine.calculate(direction="SHORT", entry_price=63000, stop_price=63100,
                                nearest_resistance=None, nearest_support=None, atr=50)
    assert result.target_1 == 62900
    assert result.target_2 == 62800
    assert result.target_3 == 62700


def test_t1_adjusted_to_nearby_resistance():
    """A resistance level between entry and the 1R target should pull T1 in."""
    engine = TargetEngine(Settings())
    # entry 63000, stop 62900 (100pt risk) -> pure 1R = 63100
    # resistance at 63050 sits between entry and 1R
    result = engine.calculate(direction="LONG", entry_price=63000, stop_price=62900,
                                nearest_resistance=63050, nearest_support=None, atr=50)
    print(f"Adjusted T1: {result.target_1}, basis: {result.basis}")
    assert result.target_1 == 63050
    assert "adjusted" in result.basis.lower()


def test_t1_reverts_if_too_close_to_entry():
    """If the nearby level is too close to entry (near-zero R:R for T1
    specifically), keep the pure R-multiple target instead."""
    engine = TargetEngine(Settings())
    # entry 63000, stop 62900 (100pt risk), resistance at 63010 -> T1 R:R would be 0.1, below the 0.3 floor
    result = engine.calculate(direction="LONG", entry_price=63000, stop_price=62900,
                                nearest_resistance=63010, nearest_support=None, atr=50)
    print(f"Reverted T1: {result.target_1}, basis: {result.basis}")
    assert result.target_1 == 63100, "Should keep pure 1R target when nearby level is too close to entry"
    assert "too close" in result.basis.lower()


def test_resistance_far_beyond_1R_not_used():
    """A resistance level BEYOND the 1R target shouldn't affect T1 at all."""
    engine = TargetEngine(Settings())
    result = engine.calculate(direction="LONG", entry_price=63000, stop_price=62900,
                                nearest_resistance=63500, nearest_support=None, atr=50)
    assert result.target_1 == 63100, "Resistance beyond the 1R target should not adjust T1"


def test_zero_risk_distance_returns_none():
    """Entry equals stop -> invalid setup, must not divide by zero."""
    engine = TargetEngine(Settings())
    result = engine.calculate(direction="LONG", entry_price=63000, stop_price=63000,
                                nearest_resistance=None, nearest_support=None, atr=50)
    assert result is None


if __name__ == "__main__":
    tests = [
        test_structure_stop_chosen_when_tightest_and_within_cap,
        test_atr_stop_chosen_when_no_structure_data,
        test_stop_rejected_when_all_candidates_exceed_cap,
        test_short_direction_stop_placed_above_entry,
        test_long_direction_stop_placed_below_entry,
        test_all_candidates_returned_for_transparency,
        test_basic_r_multiple_targets,
        test_short_targets_go_downward,
        test_t1_adjusted_to_nearby_resistance,
        test_t1_reverts_if_too_close_to_entry,
        test_resistance_far_beyond_1R_not_used,
        test_zero_risk_distance_returns_none,
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
