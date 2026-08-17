import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from execution.brokerage_calculator import calculate_charges


def test_hand_calculated_long_trade():
    """
    Entry 63000, exit 63200, 1 lot LONG.
    Hand-calculate every component to verify against Angel One's real rates.
    """
    result = calculate_charges(direction="LONG", entry_price=63000, exit_price=63200, lots=1)

    buy_turnover = 63000 * 10   # 630,000
    sell_turnover = 63200 * 10  # 632,000
    assert result.buy_turnover_inr == buy_turnover
    assert result.sell_turnover_inr == sell_turnover

    # brokerage: ₹20 x 2 orders = ₹40
    assert result.brokerage_inr == 40.0

    # CTT: 0.01% of SELL turnover only = 632000 * 0.0001 = 63.2
    assert abs(result.ctt_inr - 63.2) < 0.01

    # exchange txn charge: 0.0026% of TOTAL turnover
    total_turnover = buy_turnover + sell_turnover
    expected_exch = total_turnover * 0.000026
    assert abs(result.exchange_txn_charge_inr - round(expected_exch, 2)) < 0.01

    # GST: 18% of (brokerage + exchange charge)
    expected_gst = (40.0 + expected_exch) * 0.18
    assert abs(result.gst_inr - round(expected_gst, 2)) < 0.01

    # stamp duty: 0.002% of BUY turnover only
    expected_stamp = buy_turnover * 0.00002
    assert abs(result.stamp_duty_inr - round(expected_stamp, 2)) < 0.01

    # gross P&L: (exit - entry) * 10/point * 1 lot = 200 * 10 = 2000
    assert result.gross_pnl_inr == 2000.0

    # net = gross - total charges
    assert abs(result.net_pnl_inr - (2000.0 - result.total_charges_inr)) < 0.01
    print(f"LONG trade: gross={result.gross_pnl_inr}, charges={result.total_charges_inr}, "
          f"net={result.net_pnl_inr}")


def test_short_trade_swaps_buy_sell_legs():
    """
    For a SHORT trade, entry IS the sell leg and exit IS the buy leg —
    the reverse of LONG. CTT (sell-only) and stamp duty (buy-only) must
    apply to the correct leg, not just "entry" and "exit" literally.
    """
    result = calculate_charges(direction="SHORT", entry_price=63200, exit_price=63000, lots=1)

    # entry (63200) is the SELL leg, exit (63000) is the BUY leg
    assert result.sell_turnover_inr == 63200 * 10
    assert result.buy_turnover_inr == 63000 * 10

    # gross P&L for a profitable short: entry - exit = 200 points * 10 = 2000
    assert result.gross_pnl_inr == 2000.0


def test_charges_reduce_a_small_winning_trade_meaningfully():
    """A thin win should show charges taking a real, non-trivial bite —
    this is the whole point of the calculator."""
    result = calculate_charges(direction="LONG", entry_price=63000, exit_price=63050, lots=1)
    print(f"Thin win: gross={result.gross_pnl_inr}, charges={result.total_charges_inr}, "
          f"net={result.net_pnl_inr}")
    assert result.gross_pnl_inr == 500.0
    assert result.total_charges_inr > 100, \
        "Charges on a real MCX round trip should be well over ₹100, not negligible"
    assert result.net_pnl_inr < result.gross_pnl_inr


def test_charges_scale_with_lot_count():
    result_1lot = calculate_charges(direction="LONG", entry_price=63000, exit_price=63200, lots=1)
    result_2lot = calculate_charges(direction="LONG", entry_price=63000, exit_price=63200, lots=2)

    # brokerage is FLAT per order (not per lot) — this must NOT double
    assert result_1lot.brokerage_inr == result_2lot.brokerage_inr == 40.0

    # but CTT, exchange charges, stamp duty scale with turnover (2x lots = 2x turnover)
    assert abs(result_2lot.ctt_inr - result_1lot.ctt_inr * 2) < 0.02
    assert abs(result_2lot.stamp_duty_inr - result_1lot.stamp_duty_inr * 2) < 0.02

    # gross P&L must scale exactly with lots
    assert result_2lot.gross_pnl_inr == result_1lot.gross_pnl_inr * 2


def test_losing_trade_charges_still_apply():
    """Charges are owed regardless of win/loss — a losing trade's net loss
    must be WORSE than its gross loss, not better."""
    result = calculate_charges(direction="LONG", entry_price=63000, exit_price=62900, lots=1)
    assert result.gross_pnl_inr == -1000.0
    assert result.net_pnl_inr < result.gross_pnl_inr, \
        "Charges must make a loss worse, never better"


def test_invalid_direction_raises():
    try:
        calculate_charges(direction="SIDEWAYS", entry_price=63000, exit_price=63100, lots=1)
        assert False, "Expected ValueError for invalid direction"
    except ValueError:
        pass


def test_realistic_full_breakdown_matches_industry_calculator_pattern():
    """
    Sanity cross-check: for a ~₹6.3L notional round trip (1 lot GOLDM at
    ~63000), total charges should land in the same ballpark independently
    published commodity brokerage calculators show for similar-sized MCX
    trades — roughly ₹100-250 all-in for a 1-lot round trip at this price
    level, not ₹5 (too cheap to be real) or ₹5000 (wildly too expensive).
    """
    result = calculate_charges(direction="LONG", entry_price=63000, exit_price=63200, lots=1)
    print(f"Full breakdown: {result}")
    assert 100 <= result.total_charges_inr <= 300, \
        f"Total charges ({result.total_charges_inr}) should be in a realistic " \
        f"range for a 1-lot MCX round trip at this price level"


if __name__ == "__main__":
    tests = [
        test_hand_calculated_long_trade,
        test_short_trade_swaps_buy_sell_legs,
        test_charges_reduce_a_small_winning_trade_meaningfully,
        test_charges_scale_with_lot_count,
        test_losing_trade_charges_still_apply,
        test_invalid_direction_raises,
        test_realistic_full_breakdown_matches_industry_calculator_pattern,
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
