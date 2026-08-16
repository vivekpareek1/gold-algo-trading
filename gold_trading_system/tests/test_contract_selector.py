import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from market_data.contract_selector import (
    parse_goldm_expiry, select_next_month_contract, MONTH_MAP
)


def test_parse_valid_symbol():
    expiry = parse_goldm_expiry("GOLDM04SEP26FUT")
    assert expiry == datetime(2026, 9, 4)


def test_parse_all_months():
    for month_str, month_num in MONTH_MAP.items():
        symbol = f"GOLDM15{month_str}27FUT"
        expiry = parse_goldm_expiry(symbol)
        assert expiry == datetime(2027, month_num, 15), f"Failed for {month_str}"


def test_non_futures_symbol_returns_none():
    """Options symbols (CE/PE) must be ignored, not misparsed as futures."""
    assert parse_goldm_expiry("GOLDM25SEP26116500CE") is None
    assert parse_goldm_expiry("GOLDM25SEP26116500PE") is None
    assert parse_goldm_expiry("GOLDMCOM") is None
    assert parse_goldm_expiry("GOLDMAHMCOM") is None


def test_select_picks_second_nearest_not_nearest():
    """THE core requirement: must return the contract AFTER the nearest one."""
    contracts = [
        {"tradingsymbol": "GOLDM04SEP26FUT", "symboltoken": "563946"},
        {"tradingsymbol": "GOLDM05OCT26FUT", "symboltoken": "569003"},
        {"tradingsymbol": "GOLDM05NOV26FUT", "symboltoken": "571445"},
        {"tradingsymbol": "GOLDM04DEC26FUT", "symboltoken": "575011"},
    ]
    as_of = datetime(2026, 8, 15)
    result = select_next_month_contract(contracts, as_of=as_of)
    print(f"Selected: {result.tradingsymbol}")
    assert result.tradingsymbol == "GOLDM05OCT26FUT", \
        f"Expected the SECOND nearest (Oct), not nearest (Sep) or third (Nov), got {result.tradingsymbol}"
    assert result.symboltoken == "569003"


def test_ignores_expired_contracts():
    """A contract whose expiry has already passed must not be selectable at all."""
    contracts = [
        {"tradingsymbol": "GOLDM04JUL26FUT", "symboltoken": "999999"},  # already expired
        {"tradingsymbol": "GOLDM04SEP26FUT", "symboltoken": "563946"},
        {"tradingsymbol": "GOLDM05OCT26FUT", "symboltoken": "569003"},
    ]
    as_of = datetime(2026, 8, 15)
    result = select_next_month_contract(contracts, as_of=as_of)
    assert result.tradingsymbol == "GOLDM05OCT26FUT", \
        "Expired contract must be excluded from consideration entirely"


def test_ignores_options_when_selecting():
    """Options contracts (CE/PE) mixed into the same list must not interfere."""
    contracts = [
        {"tradingsymbol": "GOLDM25SEP26116500CE", "symboltoken": "580161"},
        {"tradingsymbol": "GOLDM25SEP26116500PE", "symboltoken": "580165"},
        {"tradingsymbol": "GOLDM04SEP26FUT", "symboltoken": "563946"},
        {"tradingsymbol": "GOLDM05OCT26FUT", "symboltoken": "569003"},
    ]
    as_of = datetime(2026, 8, 15)
    result = select_next_month_contract(contracts, as_of=as_of)
    assert result.tradingsymbol == "GOLDM05OCT26FUT"


def test_raises_when_fewer_than_two_contracts_available():
    """If only one future-dated FUT exists, there's no 'next' to select —
    must raise loudly rather than silently picking the wrong thing."""
    contracts = [{"tradingsymbol": "GOLDM04SEP26FUT", "symboltoken": "563946"}]
    as_of = datetime(2026, 8, 15)
    try:
        select_next_month_contract(contracts, as_of=as_of)
        assert False, "Expected ValueError with fewer than 2 available contracts"
    except ValueError as e:
        print(f"Correctly raised: {e}")


def test_realistic_full_contract_list():
    """End-to-end with the actual contract list format seen from real Angel One data."""
    contracts = [
        {"tradingsymbol": "GOLDM04DEC26FUT", "symboltoken": "575011"},
        {"tradingsymbol": "GOLDM04SEP26FUT", "symboltoken": "563946"},
        {"tradingsymbol": "GOLDM05FEB27FUT", "symboltoken": "581880"},
        {"tradingsymbol": "GOLDM05JAN27FUT", "symboltoken": "578779"},
        {"tradingsymbol": "GOLDM05NOV26FUT", "symboltoken": "571445"},
        {"tradingsymbol": "GOLDM05OCT26FUT", "symboltoken": "569003"},
        {"tradingsymbol": "GOLDM25SEP26116500CE", "symboltoken": "580161"},
        {"tradingsymbol": "GOLDMCOM", "symboltoken": "117"},
    ]
    as_of = datetime(2026, 8, 15)
    result = select_next_month_contract(contracts, as_of=as_of)
    print(f"Real-list result: {result.tradingsymbol} ({result.symboltoken})")
    assert result.tradingsymbol == "GOLDM05OCT26FUT"
    assert result.symboltoken == "569003"


def test_rolls_forward_correctly_as_time_passes():
    """As 'now' advances past September, the selection must roll forward too."""
    contracts = [
        {"tradingsymbol": "GOLDM04SEP26FUT", "symboltoken": "563946"},
        {"tradingsymbol": "GOLDM05OCT26FUT", "symboltoken": "569003"},
        {"tradingsymbol": "GOLDM05NOV26FUT", "symboltoken": "571445"},
    ]
    before_sep_expiry = select_next_month_contract(contracts, as_of=datetime(2026, 8, 15))
    after_sep_expiry = select_next_month_contract(contracts, as_of=datetime(2026, 9, 10))

    print(f"Before Sep expiry: {before_sep_expiry.tradingsymbol}")
    print(f"After Sep expiry: {after_sep_expiry.tradingsymbol}")
    assert before_sep_expiry.tradingsymbol == "GOLDM05OCT26FUT"
    assert after_sep_expiry.tradingsymbol == "GOLDM05NOV26FUT", \
        "Once September has expired, October becomes nearest and November becomes next"


if __name__ == "__main__":
    tests = [
        test_parse_valid_symbol,
        test_parse_all_months,
        test_non_futures_symbol_returns_none,
        test_select_picks_second_nearest_not_nearest,
        test_ignores_expired_contracts,
        test_ignores_options_when_selecting,
        test_raises_when_fewer_than_two_contracts_available,
        test_realistic_full_contract_list,
        test_rolls_forward_correctly_as_time_passes,
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
