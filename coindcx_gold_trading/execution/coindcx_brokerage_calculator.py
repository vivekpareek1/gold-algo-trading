"""
CoinDCX Gold (XAU-USDT / PAXG-USDT) Charges Calculator.

Verified via web search (2026-08-19):
  Trading fee:  0.01% maker, 0.01% taker (flat, promotional rate,
                confirmed as of June 2026 announcement — this is a
                promotional rate and could change; RE-VERIFY before
                trusting for real capital)
  Funding rate: XAU-USDT is a perpetual futures contract, settled every
                4 hours. Typically small (search showed near-0% for XAU
                at time of check, ~0.01% ballpark for similar assets),
                but genuinely non-zero and NOT a one-time cost — it
                applies for EVERY funding interval a position is held
                through, making it relevant for any trade lasting more
                than a few hours.

This is dramatically simpler than MCX's charge stack (no CTT, no GST,
no stamp duty, no SEBI turnover fee, no exchange transaction charge
layered separately) — just two components: trading fee + funding.

NOT YET VALIDATED against real historical data — this module can be
unit-tested on its own math, but real backtesting needs actual
XAU-USDT/PAXG-USDT historical OHLCV data (pending from Vivek).
"""
from dataclasses import dataclass


TRADING_FEE_RATE = 0.0001       # 0.01%, each side (maker or taker)
DEFAULT_FUNDING_RATE_PER_INTERVAL = 0.0001   # 0.01% per 4hr interval — a
                                                 # CONSERVATIVE placeholder;
                                                 # replace with real historical
                                                 # average once available,
                                                 # this is not yet verified
                                                 # for XAU-USDT specifically
FUNDING_INTERVAL_HOURS = 4


@dataclass
class CoinDCXChargesBreakdown:
    entry_notional_usd: float
    exit_notional_usd: float
    trading_fee_usd: float
    funding_fee_usd: float
    total_charges_usd: float
    gross_pnl_usd: float
    net_pnl_usd: float
    funding_intervals_charged: int


def calculate_coindcx_charges(direction: str, entry_price: float, exit_price: float,
                                 quantity: float, hold_duration_hours: float = 0.0,
                                 funding_rate_per_interval: float = DEFAULT_FUNDING_RATE_PER_INTERVAL
                                 ) -> CoinDCXChargesBreakdown:
    """
    direction: "LONG" or "SHORT".
    quantity: position size in units of the underlying (e.g., ounces of
      XAU, or contract-equivalent — NOT "lots" like MCX; CoinDCX allows
      fractional position sizes).
    hold_duration_hours: how long the position was open — determines how
      many funding intervals were crossed (funding only charged if a
      position is held AT a funding timestamp, not just for any partial
      time — this conservatively rounds UP to be safe about cost).
    """
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")

    entry_notional = entry_price * quantity
    exit_notional = exit_price * quantity

    if direction == "LONG":
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity

    trading_fee = (entry_notional + exit_notional) * TRADING_FEE_RATE

    funding_intervals = 0
    if hold_duration_hours > 0:
        funding_intervals = int(hold_duration_hours // FUNDING_INTERVAL_HOURS) + 1
    funding_fee = entry_notional * funding_rate_per_interval * funding_intervals

    total_charges = trading_fee + funding_fee
    net_pnl = gross_pnl - total_charges

    return CoinDCXChargesBreakdown(
        entry_notional_usd=round(entry_notional, 4), exit_notional_usd=round(exit_notional, 4),
        trading_fee_usd=round(trading_fee, 4), funding_fee_usd=round(funding_fee, 4),
        total_charges_usd=round(total_charges, 4), gross_pnl_usd=round(gross_pnl, 4),
        net_pnl_usd=round(net_pnl, 4), funding_intervals_charged=funding_intervals,
    )
