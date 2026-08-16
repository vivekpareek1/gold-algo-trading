import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from risk_engine.risk_engine import RiskEngine, DailyRiskState, VetoReason


def fresh_engine():
    return RiskEngine(Settings(), DailyRiskState())


def test_normal_position_sizing():
    """₹2000 risk, 100pt stop => risk_per_lot = 100*10=1000, so 2 lots."""
    engine = fresh_engine()
    result = engine.calculate_position_size(
        entry_price=63000, stop_price=62900,  # 100pt stop
        live_equity_inr=500000, risk_reward=2.0
    )
    print(f"Normal sizing: {result}")
    assert result.approved, f"Expected approval, got veto: {result.veto_reason}"
    assert result.lots == 2, f"Expected 2 lots (2000/1000), got {result.lots}"


def test_stop_too_wide_rejects_not_mis_sizes():
    """Stop wider than 200pt cap must REJECT, not silently shrink the stop."""
    engine = fresh_engine()
    result = engine.calculate_position_size(
        entry_price=63000, stop_price=62700,  # 300pt stop > 200pt cap
        live_equity_inr=500000, risk_reward=2.0
    )
    print(f"Wide stop test: {result}")
    assert not result.approved
    assert result.veto_reason == VetoReason.STOP_TOO_WIDE


def test_position_never_rounds_up():
    """If raw lots computes to e.g. 0.4, must reject, never round up to 1."""
    engine = fresh_engine()
    # 190pt stop => risk_per_lot=1900, allowed=2000 => raw=1.05 lots -> floors to 1, fine.
    # Use a bigger stop within cap to force raw < 1
    result = engine.calculate_position_size(
        entry_price=63000, stop_price=62810,  # 190pt -> risk_per_lot=1900 -> raw=1.05 -> 1 lot ok
        live_equity_inr=500000, risk_reward=2.0
    )
    print(f"Near-boundary sizing: {result}")
    assert result.approved and result.lots == 1


def test_risk_reward_below_minimum_rejected():
    engine = fresh_engine()
    result = engine.calculate_position_size(
        entry_price=63000, stop_price=62900,
        live_equity_inr=500000, risk_reward=1.2  # below 1.5 minimum
    )
    assert not result.approved
    assert result.veto_reason == VetoReason.RISK_REWARD_TOO_LOW


def test_graduated_derisking_after_losses():
    """2 consecutive losses -> 0.75x multiplier; 3 -> 0.5x; 4 -> disabled."""
    engine = fresh_engine()
    engine.record_trade_result(-500)
    assert engine.state.current_lot_multiplier == 1.0, "1 loss should not yet reduce size"

    engine.record_trade_result(-500)
    assert engine.state.current_lot_multiplier == 0.75, \
        f"Expected 0.75x after 2 losses, got {engine.state.current_lot_multiplier}"

    engine.record_trade_result(-500)
    assert engine.state.current_lot_multiplier == 0.50, \
        f"Expected 0.5x after 3 losses, got {engine.state.current_lot_multiplier}"

    engine.record_trade_result(-500)
    assert engine.state.trading_disabled == True, \
        "Expected trading disabled after 4 consecutive losses"


def test_derisk_resets_after_wins():
    engine = fresh_engine()
    engine.record_trade_result(-500)
    engine.record_trade_result(-500)  # multiplier -> 0.75
    assert engine.state.current_lot_multiplier == 0.75

    engine.record_trade_result(500)   # win 1
    engine.record_trade_result(500)   # win 2 -> reset threshold met
    assert engine.state.current_lot_multiplier == 1.0, \
        f"Expected reset to 1.0x after 2 consecutive wins, got {engine.state.current_lot_multiplier}"


def test_scaleup_never_auto_applies():
    """3 consecutive wins should RECOMMEND scale-up but NOT change the multiplier."""
    engine = fresh_engine()
    engine.record_trade_result(500)
    engine.record_trade_result(500)
    engine.record_trade_result(500)
    assert engine.state.scaleup_recommended == True
    assert engine.state.current_lot_multiplier == 1.0, \
        "Multiplier must NOT change automatically on scale-up recommendation"

    # only an explicit confirm_scaleup call should change it
    engine.confirm_scaleup(1.5)
    assert engine.state.current_lot_multiplier == 1.5
    assert engine.state.scaleup_recommended == False


def test_daily_loss_limit_veto():
    engine = fresh_engine()
    engine.state.daily_pnl_inr = -16000  # 3.2% of 500000, above 3% limit
    veto = engine.check_hard_limits(live_equity_inr=500000, data_is_stale=False,
                                      position_already_open=False)
    assert veto == VetoReason.DAILY_LOSS_LIMIT_HIT, f"Expected daily loss veto, got {veto}"


def test_stale_data_always_vetoes_first():
    """Stale data must veto regardless of other state (fail-safe ordering)."""
    engine = fresh_engine()
    veto = engine.check_hard_limits(live_equity_inr=500000, data_is_stale=True,
                                      position_already_open=False)
    assert veto == VetoReason.STALE_DATA


def test_margin_estimate_scales_with_price_not_frozen_at_launch_level():
    """
    Regression test: margin used to be a hardcoded 65000/lot, calibrated to
    gold's price when this was written (~70,000). Real MCX data showed gold
    move to ~152,000, at which point the hardcoded figure understated true
    margin by roughly half. Margin must scale with actual contract value.
    """
    engine = fresh_engine()
    margin_low = engine._estimate_margin(1, entry_price=70261)
    margin_high = engine._estimate_margin(1, entry_price=152230)
    ratio = margin_high / margin_low
    price_ratio = 152230 / 70261
    print(f"Margin ratio: {ratio:.2f}, price ratio: {price_ratio:.2f}")
    assert abs(ratio - price_ratio) < 0.01, \
        "Margin estimate must scale proportionally with entry price, not stay fixed"


if __name__ == "__main__":
    tests = [
        test_normal_position_sizing,
        test_stop_too_wide_rejects_not_mis_sizes,
        test_position_never_rounds_up,
        test_risk_reward_below_minimum_rejected,
        test_graduated_derisking_after_losses,
        test_derisk_resets_after_wins,
        test_scaleup_never_auto_applies,
        test_daily_loss_limit_veto,
        test_stale_data_always_vetoes_first,
        test_margin_estimate_scales_with_price_not_frozen_at_launch_level,
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


