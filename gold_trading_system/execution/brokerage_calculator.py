"""
Real Brokerage & Charges Calculator — MCX Commodity Futures via Angel One.

Without this, "profit" shown anywhere is fiction — real trading always
costs brokerage, taxes, and exchange fees on every round trip, and those
add up meaningfully over hundreds of paper trades. Rates below are Angel
One's published commodity F&O rate card plus statutory charges (verified
against Angel One's own site and MCX/MCXCCL's CTT notification):

  Brokerage:            ₹20 flat per executed order (buy leg + sell leg)
  CTT:                  0.01% on the SELL leg's notional value only
                        (Commodity Transaction Tax — commodities' equivalent
                        of STT; MCXCCL circular, gold/silver/crude at 0.01%)
  Exchange txn charges: 0.0026% of turnover, both legs
  GST:                  18% on (brokerage + exchange transaction charges)
                        — NOT on CTT/stamp duty, those are taxes not services
  SEBI turnover fee:    ₹10 per crore of turnover, both legs
  Stamp duty:           0.002% on the BUY leg's notional value only

CTT applies to the SELL leg and stamp duty to the BUY leg specifically —
for a SHORT trade, entry is the sell leg and exit is the buy leg, the
reverse of a LONG trade. Getting this swapped would misstate every short
trade's real cost.

These rates are current as verified via web search; they are NOT hardcoded
assumptions — Angel One or the government can change them, and this module
should be updated if their published rate card changes.
"""
from dataclasses import dataclass


BROKERAGE_PER_ORDER_INR = 20.0
CTT_RATE = 0.0001            # 0.01%, sell leg only
EXCHANGE_TXN_CHARGE_RATE = 0.000026   # 0.0026%, both legs
GST_RATE = 0.18               # on brokerage + exchange txn charges only
SEBI_FEE_RATE = 10.0 / 10_000_000   # ₹10 per crore, both legs
STAMP_DUTY_RATE = 0.00002     # 0.002%, buy leg only


@dataclass
class ChargesBreakdown:
    buy_turnover_inr: float
    sell_turnover_inr: float
    brokerage_inr: float
    ctt_inr: float
    exchange_txn_charge_inr: float
    gst_inr: float
    sebi_fee_inr: float
    stamp_duty_inr: float
    total_charges_inr: float

    gross_pnl_inr: float
    net_pnl_inr: float


def calculate_charges(direction: str, entry_price: float, exit_price: float,
                        lots: int, point_value_inr: float = 10.0,
                        lot_multiplier: float = 10.0) -> ChargesBreakdown:
    """
    direction: "LONG" or "SHORT" — determines which leg (entry/exit) is the
    buy vs sell side, which matters for CTT (sell-only) and stamp duty
    (buy-only).
    point_value_inr: rupees per point per lot (GOLDM: ₹10).
    lot_multiplier: contract value = price * lot_multiplier per lot
    (GOLDM is quoted per 10g on a 100g contract, so multiplier = 10).
    """
    if direction == "LONG":
        buy_price, sell_price = entry_price, exit_price
        gross_pnl = (exit_price - entry_price) * point_value_inr * lots
    elif direction == "SHORT":
        buy_price, sell_price = exit_price, entry_price
        gross_pnl = (entry_price - exit_price) * point_value_inr * lots
    else:
        raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")

    buy_turnover = buy_price * lot_multiplier * lots
    sell_turnover = sell_price * lot_multiplier * lots
    total_turnover = buy_turnover + sell_turnover

    brokerage = BROKERAGE_PER_ORDER_INR * 2  # one order to enter, one to exit
    ctt = sell_turnover * CTT_RATE
    exchange_txn_charge = total_turnover * EXCHANGE_TXN_CHARGE_RATE
    gst = (brokerage + exchange_txn_charge) * GST_RATE
    sebi_fee = total_turnover * SEBI_FEE_RATE
    stamp_duty = buy_turnover * STAMP_DUTY_RATE

    total_charges = brokerage + ctt + exchange_txn_charge + gst + sebi_fee + stamp_duty
    net_pnl = gross_pnl - total_charges

    return ChargesBreakdown(
        buy_turnover_inr=round(buy_turnover, 2), sell_turnover_inr=round(sell_turnover, 2),
        brokerage_inr=round(brokerage, 2), ctt_inr=round(ctt, 2),
        exchange_txn_charge_inr=round(exchange_txn_charge, 2), gst_inr=round(gst, 2),
        sebi_fee_inr=round(sebi_fee, 2), stamp_duty_inr=round(stamp_duty, 2),
        total_charges_inr=round(total_charges, 2),
        gross_pnl_inr=round(gross_pnl, 2), net_pnl_inr=round(net_pnl, 2),
    )
