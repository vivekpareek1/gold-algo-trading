"""
GOLDM Contract Selector.
Locked design decision: trade the contract AFTER the nearest expiry, not
the nearest one itself. Near-expiry contracts see declining liquidity and
carry rollover/delivery-window risk as expiry approaches — the next-month
contract is the more tradeable choice.

This module parses Angel One's GOLDM contract naming convention
(e.g. "GOLDM04SEP26FUT" = expires 4 Sept 2026) and picks the second-nearest
expiry automatically, so this never needs manual updating as contracts
roll over month to month.
"""
import re
from dataclasses import dataclass
from datetime import datetime


MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# matches e.g. "GOLDM04SEP26FUT" -> day=04, month=SEP, year=26
_FUT_PATTERN = re.compile(r"^GOLDM(\d{2})([A-Z]{3})(\d{2})FUT$")


@dataclass
class GoldmContract:
    tradingsymbol: str
    symboltoken: str
    expiry_date: datetime


def parse_goldm_expiry(tradingsymbol: str) -> datetime | None:
    """Returns the expiry date for a GOLDM futures symbol, or None if the
    symbol isn't a plain futures contract (options, spot, etc. are ignored)."""
    m = _FUT_PATTERN.match(tradingsymbol)
    if not m:
        return None
    day, month_str, year_2digit = m.groups()
    month = MONTH_MAP.get(month_str)
    if month is None:
        return None
    year = 2000 + int(year_2digit)
    try:
        return datetime(year, month, int(day))
    except ValueError:
        return None


def select_next_month_contract(contracts: list[dict], as_of: datetime | None = None) -> GoldmContract:
    """
    contracts: raw list of dicts from Angel One's searchScrip() response,
    each with 'tradingsymbol' and 'symboltoken'.
    as_of: reference date for "nearest" (defaults to now) — pass explicitly
    in tests for determinism.

    Returns the SECOND-nearest expiry (the "next expiry after current"),
    not the nearest one, per the locked design decision. Raises if fewer
    than 2 future-dated FUT contracts are found — that's a real problem
    to surface, not something to silently work around.
    """
    as_of = as_of or datetime.now()

    parsed: list[GoldmContract] = []
    for c in contracts:
        symbol = c.get("tradingsymbol", "")
        expiry = parse_goldm_expiry(symbol)
        if expiry is not None and expiry >= as_of:
            parsed.append(GoldmContract(tradingsymbol=symbol,
                                          symboltoken=c.get("symboltoken", ""),
                                          expiry_date=expiry))

    if len(parsed) < 2:
        raise ValueError(
            f"Found only {len(parsed)} future-dated GOLDM FUT contract(s) as of "
            f"{as_of.date()} — need at least 2 (nearest + next) to select the "
            f"next-month contract. Check the raw contract list."
        )

    parsed.sort(key=lambda c: c.expiry_date)
    nearest = parsed[0]
    next_month = parsed[1]
    return next_month
