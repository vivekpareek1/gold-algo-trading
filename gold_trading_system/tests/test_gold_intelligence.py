import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from gold_intelligence.fair_value import (
    FairValueEngine, MacroContextEngine, MacroInputs, MoveClassification
)


def test_fair_value_basic_calculation():
    """Hand-verify the math with round numbers."""
    settings = Settings()
    settings.gold_specific.import_duty_rate = 0.06  # 6%
    settings.gold_specific.carry_cost_rate_annual = 0.0  # isolate the FX/duty math
    engine = FairValueEngine(settings)

    xauusd = 2000.0   # $/troy oz
    usdinr = 83.0
    # landed_cost_per_10g = (2000/31.1035)*10*83*1.06
    expected = (xauusd / 31.1035) * 10 * usdinr * 1.06
    mcx_price = expected  # set MCX exactly at fair value -> deviation should be ~0

    result = engine.calculate(mcx_price=mcx_price, xauusd=xauusd, usdinr=usdinr,
                                days_to_expiry=0, both_sessions_live=True)
    print(f"Theoretical: {result.theoretical_price:.2f}, expected: {expected:.2f}, "
          f"deviation: {result.deviation:.4f}")
    assert abs(result.theoretical_price - expected) < 0.01
    assert abs(result.deviation) < 0.01, f"MCX priced exactly at fair value should show ~0 deviation"
    assert result.is_reliable == True


def test_fair_value_detects_premium():
    settings = Settings()
    settings.gold_specific.import_duty_rate = 0.06
    settings.gold_specific.carry_cost_rate_annual = 0.0
    engine = FairValueEngine(settings)

    xauusd = 2000.0
    usdinr = 83.0
    fair = (xauusd / 31.1035) * 10 * usdinr * 1.06
    mcx_price = fair * 1.02  # 2% premium

    result = engine.calculate(mcx_price=mcx_price, xauusd=xauusd, usdinr=usdinr,
                                days_to_expiry=0, both_sessions_live=True)
    print(f"Premium test - deviation_pct: {result.deviation_pct:.3f}%")
    assert result.deviation_pct > 1.5, f"Expected ~2% premium detected, got {result.deviation_pct:.3f}%"


def test_fair_value_unreliable_without_session_overlap():
    """Must refuse to trust the deviation reading if sessions don't overlap."""
    settings = Settings()
    engine = FairValueEngine(settings)
    result = engine.calculate(mcx_price=63000, xauusd=2000, usdinr=83,
                                days_to_expiry=10, both_sessions_live=False)
    assert result.is_reliable == False
    assert "session" in result.unreliable_reason.lower()


def test_fair_value_unreliable_on_stale_data():
    settings = Settings()
    engine = FairValueEngine(settings)
    result = engine.calculate(mcx_price=63000, xauusd=2000, usdinr=83,
                                days_to_expiry=10, both_sessions_live=True, data_stale=True)
    assert result.is_reliable == False


def test_move_classification_metal_driven():
    settings = Settings()
    engine = FairValueEngine(settings)
    classification = engine.classify_move(xauusd_change_pct=0.5, usdinr_change_pct=0.001)
    assert classification == MoveClassification.METAL_DRIVEN


def test_move_classification_rupee_driven():
    settings = Settings()
    engine = FairValueEngine(settings)
    classification = engine.classify_move(xauusd_change_pct=0.001, usdinr_change_pct=0.5)
    assert classification == MoveClassification.RUPEE_DRIVEN


def test_move_classification_amplified():
    settings = Settings()
    engine = FairValueEngine(settings)
    classification = engine.classify_move(xauusd_change_pct=0.5, usdinr_change_pct=0.3)
    assert classification == MoveClassification.AMPLIFIED


def test_move_classification_conflicted():
    settings = Settings()
    engine = FairValueEngine(settings)
    classification = engine.classify_move(xauusd_change_pct=0.5, usdinr_change_pct=-0.3)
    assert classification == MoveClassification.CONFLICTED


def test_move_classification_flat():
    settings = Settings()
    engine = FairValueEngine(settings)
    classification = engine.classify_move(xauusd_change_pct=0.005, usdinr_change_pct=0.005)
    assert classification == MoveClassification.FLAT


def test_macro_bias_bullish_on_falling_yields_and_dxy():
    engine = MacroContextEngine()
    inputs = MacroInputs(
        dxy=103.0, dxy_prev=104.0,               # DXY falling -> bullish
        us10y_real_yield=1.8, us10y_real_yield_prev=2.0,  # real yield falling -> bullish
        usdinr=83.0, usdinr_prev=83.0,            # flat
        crude=75.0, crude_prev=75.0,              # flat
    )
    result = engine.compute(inputs)
    print(f"Macro bias (falling yields+DXY): {result.macro_bias:.2f}, components={result.components}")
    assert result.macro_bias > 0, f"Expected bullish macro bias, got {result.macro_bias}"


def test_macro_bias_never_exceeds_100():
    """Sanity: even with extreme inputs, bias must stay within -100..+100."""
    engine = MacroContextEngine()
    inputs = MacroInputs(
        dxy=50.0, dxy_prev=200.0,   # extreme move
        us10y_real_yield=0.1, us10y_real_yield_prev=10.0,
        usdinr=150.0, usdinr_prev=50.0,
        crude=200.0, crude_prev=10.0,
    )
    result = engine.compute(inputs)
    print(f"Extreme macro bias: {result.macro_bias}")
    assert -100.0 <= result.macro_bias <= 100.0


def test_macro_bias_zero_prev_handled_safely():
    """Must not crash on a zero prev value (division by zero guard)."""
    engine = MacroContextEngine()
    inputs = MacroInputs(
        dxy=103.0, dxy_prev=0.0,
        us10y_real_yield=1.8, us10y_real_yield_prev=0.0,
        usdinr=83.0, usdinr_prev=0.0,
        crude=75.0, crude_prev=0.0,
    )
    result = engine.compute(inputs)  # should not raise
    assert isinstance(result.macro_bias, float)


if __name__ == "__main__":
    tests = [
        test_fair_value_basic_calculation,
        test_fair_value_detects_premium,
        test_fair_value_unreliable_without_session_overlap,
        test_fair_value_unreliable_on_stale_data,
        test_move_classification_metal_driven,
        test_move_classification_rupee_driven,
        test_move_classification_amplified,
        test_move_classification_conflicted,
        test_move_classification_flat,
        test_macro_bias_bullish_on_falling_yields_and_dxy,
        test_macro_bias_never_exceeds_100,
        test_macro_bias_zero_prev_handled_safely,
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
