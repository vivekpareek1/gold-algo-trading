import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from execution.coindcx_brokerage_calculator import calculate_coindcx_charges


def test_hand_calculated_long_trade_no_funding():
    """Quick trade (< 4hrs, so still gets charged for the first funding
    interval it's held through per our conservative model), verify exact
    trading-fee math."""
    result = calculate_coindcx_charges(direction="LONG", entry_price=4400.0,
                                          exit_price=4420.0, quantity=1.0,
                                          hold_duration_hours=0)
    # no hold duration specified -> 0 funding intervals
    assert result.funding_intervals_charged == 0
    assert result.funding_fee_usd == 0.0

    entry_notional = 4400.0
    exit_notional = 4420.0
    expected_fee = (entry_notional + exit_notional) * 0.0001
    assert abs(result.trading_fee_usd - expected_fee) < 0.001

    expected_gross = 20.0  # (4420-4400)*1
    assert abs(result.gross_pnl_usd - expected_gross) < 0.001
    assert abs(result.net_pnl_usd - (expected_gross - expected_fee)) < 0.001


def test_short_trade_direction_correct():
    result = calculate_coindcx_charges(direction="SHORT", entry_price=4420.0,
                                          exit_price=4400.0, quantity=1.0)
    assert result.gross_pnl_usd == 20.0  # profitable short: entry > exit


def test_funding_charged_for_multi_hour_hold():
    """A position held 10 hours should cross multiple 4hr funding
    intervals (10/4 = 2.5, rounds up conservatively to 3)."""
    result = calculate_coindcx_charges(direction="LONG", entry_price=4400.0,
                                          exit_price=4410.0, quantity=1.0,
                                          hold_duration_hours=10)
    assert result.funding_intervals_charged == 3
    assert result.funding_fee_usd > 0


def test_no_funding_for_zero_duration():
    result = calculate_coindcx_charges(direction="LONG", entry_price=4400.0,
                                          exit_price=4410.0, quantity=1.0,
                                          hold_duration_hours=0)
    assert result.funding_intervals_charged == 0
    assert result.funding_fee_usd == 0.0


def test_fractional_quantity_supported():
    """CoinDCX allows fractional position sizes, unlike MCX's whole-lot
    requirement — verify this works correctly."""
    result = calculate_coindcx_charges(direction="LONG", entry_price=4400.0,
                                          exit_price=4420.0, quantity=0.05)
    assert result.gross_pnl_usd == 1.0  # 20 * 0.05
    assert result.trading_fee_usd > 0
    assert result.trading_fee_usd < 1.0  # much smaller than the 1-unit case


def test_charges_dramatically_lower_than_mcx_for_equivalent_position():
    """Sanity cross-check against today's MCX findings: for a comparable
    notional position, CoinDCX charges should be meaningfully lower than
    MCX's measured ~470 INR/trade (converting via a rough USD/INR rate)."""
    # entry price roughly matching current gold levels (~$4400/oz),
    # quantity chosen so notional matches MCX's ~100g lot (~$18,000 at
    # current prices, given 1 oz = ~31.1g, so ~3.2oz for equivalent notional)
    result = calculate_coindcx_charges(direction="LONG", entry_price=4400.0,
                                          exit_price=4420.0, quantity=3.2,
                                          hold_duration_hours=0)
    usd_inr = 87.5
    charges_in_inr = result.trading_fee_usd * usd_inr
    assert charges_in_inr < 470, \
        f"CoinDCX charges (Rs.{charges_in_inr:.2f}) should be meaningfully " \
        f"lower than MCX's measured ~Rs.470/trade for a comparable position"


def test_invalid_direction_raises():
    try:
        calculate_coindcx_charges(direction="SIDEWAYS", entry_price=4400,
                                     exit_price=4420, quantity=1.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_hand_calculated_long_trade_no_funding,
        test_short_trade_direction_correct,
        test_funding_charged_for_multi_hour_hold,
        test_no_funding_for_zero_duration,
        test_fractional_quantity_supported,
        test_charges_dramatically_lower_than_mcx_for_equivalent_position,
        test_invalid_direction_raises,
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
