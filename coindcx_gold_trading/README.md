# CoinDCX Gold Trading System — Design Plan

## Why this exists

Today's investigation (2026-08-19) found the MCX GOLDM strategy has
genuinely positive R-multiple edge (+0.596R, verified on 2-year real
data), but real transaction costs (CTT + GST + exchange charges + stamp
duty) eat most/all of the profit — average charges (~₹470-650/trade)
often exceed average gross profit (~₹200-450/trade).

CoinDCX offers real, gold-tracking instruments (XAU-USDT, PAXG-USDT)
with a much simpler fee structure (0.01% flat, vs MCX's tax-heavy
stack) — verified cost comparison showed ~34% lower charges for an
equivalent-size position. Given how tight the MCX margin was, this could
be enough to make the SAME exact strategy logic genuinely profitable,
without any strategy redesign.

## What's built and tested so far

- `execution/coindcx_brokerage_calculator.py` — CoinDCX's real fee
  structure (0.01% trading fee + funding rate for perpetual futures
  positions held across funding intervals). 7/7 tests passing, including
  a direct comparison confirming lower cost than MCX for an equivalent
  position.

## What's NOT built yet — needs real historical data first

**This is the hard blocker.** Everything else (reusing the existing,
tested strategy logic — market structure, indicators, signal engine,
risk engine, trade manager — all of which are asset-agnostic and work on
any OHLCV candle series) is ready to be wired up, but a full backtest
needs real XAU-USDT or PAXG-USDT 5-minute (or similar) historical OHLCV
data. **Vivek will provide this when available.**

Once data arrives, next steps are:
1. Load it via the same `DataQualityGate` pattern used for MCX
2. Run the EXACT SAME strategy logic (support/resistance filter,
   reentry-cooldown, momentum-decay exits, partial-booking — all
   already built and proven in `gold_trading_system/`) against this new
   data + the CoinDCX charges model
3. Compare real rupee (or dollar) P&L directly against today's MCX
   baseline

## Open questions to resolve before live deployment (not yet answered)

- **Funding rate accuracy**: the calculator currently uses a
  conservative PLACEHOLDER funding rate (0.01%/4hr interval) since real
  XAU-USDT historical funding rate data wasn't available during this
  session. This needs verification/replacement with real data before
  trusting real-money numbers.
- **Contract specifications**: exact minimum position size, leverage
  mechanics, and margin requirements for XAU-USDT/PAXG-USDT on CoinDCX
  need to be confirmed (this session only verified the fee rate exists
  and is genuinely ~0.01%, via web search — not verified against
  CoinDCX's own live API/docs directly).
- **24/7 trading implications**: unlike MCX (limited session hours), the
  day-boundary and session-based logic (risk resets, cooldowns) in the
  existing system may need reconsideration for a 24/7 market.
- **Data feed integration**: this repo has no live CoinDCX feed handler
  yet — that comes AFTER backtesting confirms the approach is worth
  pursuing, mirroring how the MCX system's live feed was built only
  after backtesting proved out the strategy.

## Explicitly NOT done in this session (by design — avoiding today's
demonstrated risk of rushing and introducing bugs)

- No live trading engine copy/adaptation
- No CoinDCX API integration
- No full backtest run (blocked on real data)
- No deployment of any kind

This folder is a clean, tested starting point — not a working system yet.
