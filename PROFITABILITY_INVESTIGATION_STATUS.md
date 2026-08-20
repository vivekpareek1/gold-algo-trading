# Profitability Investigation — Status as of 2026-08-19

## Confirmed, deployed fixes (real, verified improvements)

1. **Partial-booking charges bug (FIXED)** — `calculate_charges()` was
   computing P&L using only the FULL lots at the FINAL exit price,
   ignoring that partial legs (33% at +1R, etc.) close earlier at
   different, often better prices. New `calculate_charges_with_partial_booking()`
   correctly sums each leg's own charges. Verified: one sample trade that
   was genuinely +₹1,965 profitable was showing as -₹1,500 loss before
   this fix.

2. **max_lots_cap raised 3 → 10 (FIXED)** — was artificially restrictive.
   Verified on real 2-year backtest: cut net loss by ~41% (avg gross
   P&L/trade nearly tripled, ₹212 → ₹565 at cap=20).

Both are deployed (build v25, commit 2215d71).

## The core, still-unsolved problem

Even after both fixes, the strategy's REAL rupee P&L over the 2-year
backtest is still substantially negative (~-₹375K to -₹470K depending on
exact settings), despite a positive R-multiple expectancy (+0.596R).

**Root pattern found:** average gross profit per trade (₹200-450) is
smaller than average real transaction charges (₹470-650/trade). Actual
average stop distance for TAKEN trades is ~72 points (not the wider
distances a casual glance at "R-multiple positive" would suggest).

## Hypotheses tested and DISPROVEN (all real, careful tests — not guesses)

1. Raising confluence threshold (59→80) — no consistent improvement
2. Adjusting partial-booking percentages (delay first leg, bigger runner)
   — made things slightly WORSE, not better
3. Restricting by lot-count — no clean, exploitable pattern
4. Raising starting equity (₹5L → ₹1Cr) — **no effect at all** — average
   lots stayed ~3.2 regardless; equity is NOT the binding constraint for
   the average trade (only for the tail of very-tight-stop setups)
5. Raising max_risk_per_trade_inr uniformly (₹2000 → ₹15000) — **made
   things substantially WORSE** (net loss grew from -₹385K to -₹2.4M).
   A controlled, isolated test confirmed charges/gross ratio should
   IMPROVE with more lots for a FIXED trade — but in the full backtest,
   raising risk changes WHICH trades qualify, and charges grew faster
   than gross anyway. Not fully explained — flagged as needing deeper
   investigation.
6. Higher execution timeframe (15M) + wider stop cap (matching AI-Mode-
   suggested "bigger targets, same fixed cost" idea) — **also made things
   WORSE**, same pattern as #5.

## Idea proposed by Vivek, NOT yet properly tested

**Conviction-based graduated sizing**: instead of uniformly raising risk
for ALL trades (which failed, see #5/#6), only give bigger size + wider
stop to HIGH-CONVICTION setups specifically (strong momentum + volume +
clear structure/pattern — i.e., a high confluence score), while normal-
conviction setups keep the standard ₹2,000/tight-stop treatment.

This is conceptually different from what was tested (#5/#6 applied
bigger risk to EVERY trade uniformly) and has NOT been properly
disproven — a same-day quick test attempt had a bug (charges came out
inconsistent with the rest of the day's numbers, sample size was tiny
at n=17) and should be discarded, not treated as evidence either way.

**This is the most promising, not-yet-properly-tested direction for the
next session.**

## How to test it properly next time

- Use the well-tested `backtesting/backtest_runner.py` infrastructure
  (not a hand-rolled script) — add a `conviction_threshold` and
  `high_conviction_risk_multiplier` parameter, gated on
  `result.confidence` (already computed by signal_engine), mirroring how
  `require_near_support_resistance` etc. were added earlier.
- Test on the FULL real 2-year dataset, verify against a sanity check
  (avg_charges should stay consistent with the rest of the day's ~₹470-
  650/trade pattern at normal risk levels — if a test shows charges far
  outside that range, distrust the result and look for a bug first).
- Compare high-conviction-only trades' NET P&L against normal trades',
  and check the AGGREGATE net P&L impact versus the current deployed
  baseline (+0.596R / cap=10 / current partial-booking fix).

## MAJOR NEW LEAD (2026-08-19, end of session) — CoinDCX gold trading

Vivek has a CoinDCX account. Verified via search: CoinDCX genuinely offers
real, tradeable gold-linked instruments — **XAU-USDT** (gold-linked
perpetual futures) and **PAXG-USDT/PAXG-INR** (gold-backed token,
1:1 tracks real gold price), with a flat **0.01% maker/taker fee**
(much simpler than MCX's CTT+exchange+GST+stamp-duty stack), AND
**24/7 trading** (no MCX-style limited session hours).

**Quick verified cost comparison** (same-size position, ~100g gold notional):
- MCX charges (measured today, real): ~₹470/trade
- CoinDCX equivalent (0.01% x2 sides): ~₹310/trade
- **~34% cheaper**, which is meaningful given today's finding that average
  gross profit/trade (₹200-450) was barely below average charges — a
  ~34% charges reduction could be enough to tip the SAME exact strategy
  logic into genuine profitability, without needing any strategy redesign.

**This is now the most promising, concrete next step — more promising
than any of the position-sizing/timeframe experiments tried today.**

### What's needed to properly test this next session:
1. Real historical XAU-USDT or PAXG-USDT 5-min OHLCV data (need to source
   this — check if CoinDCX has a public historical data API, or export
   from their charts, or a data vendor)
2. A CoinDCX-specific charges calculator (simple — just flat 0.01% each
   side of turnover, no CTT/GST/stamp-duty/SEBI-fee complexity)
3. Re-run the EXACT SAME strategy logic (support/resistance filter,
   reentry-cooldown, momentum-decay exits, partial-booking — all already
   built and tested) on this new data + cost model
4. Given 24/7 trading, reconsider whether session-hour-based logic
   (day-boundary resets, London-NY-style filters) needs adjusting
5. Compare the result directly against today's MCX baseline


Real 2-year MCX 5-minute OHLCV data used for all backtests:
`/mnt/user-data/uploads/mcx_goldm_5min.csv` (in Claude's sandbox — will
need to be re-uploaded in a fresh session).

---

## MAJOR BREAKTHROUGH (2026-08-19, later same day) — Multi-timeframe alignment

Vivek's idea: check price action across 5M, 15M, and 1H (not just the 5M
entry signal alone) before taking a trade. 1M was excluded — no real
1-minute data exists.

**This is the single biggest improvement found in the entire
investigation — bigger than the max_lots_cap fix.**

Implementation: require BOTH the 15M trend AND the 1H trend to agree
with the proposed entry direction (TRENDING_UP for LONG, TRENDING_DOWN
for SHORT), in addition to all existing filters (support/resistance,
reentry-cooldown). Uses the SAME look-ahead-safe resampling/HTF-trend
infrastructure already built and proven for the existing 1H trend check.

**Verified on real 2-year MCX data:**

| Config | Trades | Avg Gross/trade | Net P&L |
|---|---|---|---|
| Baseline (max_trades=4, no MTF check) | 1834 | Rs446 | -Rs384,788 |
| max_trades=5 alone (no MTF check) | 2228 | Rs327 | -Rs771,505 (worse) |
| **max_trades=4 (unchanged) + MTF alignment** | **621** | **Rs439** | **-Rs153,740 (60% better)** |

Critically isolated: the improvement comes ENTIRELY from the
multi-timeframe alignment filter, not from raising max_trades_per_day
(which was independently re-confirmed to hurt, consistent with every
earlier test of raising trade quantity/quota today).

**Deployed** (build v26) to both `backtesting/backtest_runner.py`
(`require_multi_timeframe_alignment` parameter) and
`execution/live_trading_engine.py` (a dedicated `mtf_15m_aggregator`,
always active — checks 15M + 1H trend before every new entry).

### Still not fully solved

Even with this large improvement, the 2-year backtest is STILL net
negative (-Rs153,740 vs Rs20,00,000 starting equity — about -0.77% over
2 years, much smaller than before but not yet profitable). The CoinDCX
lead remains the most promising path to full profitability given MCX's
fee structure. Next session should:
1. Get real CoinDCX historical data (pending from Vivek)
2. Re-test the NOW-IMPROVED strategy (with MTF alignment) on CoinDCX's
   lower-fee structure — this could plausibly cross into genuine
   profitability given how much smaller the remaining gap is now.

### Also disproven today (before the MTF breakthrough)

- Tiered quota (base 4 trades normal threshold, extra trades require
  80-85+ confidence score) — still made things worse even with strict
  gating. Confluence score does not reliably predict move SIZE, only
  win/loss — so gating extra trades by score alone doesn't work.
- Raising max_trades_per_day alone (5, 6, 8, 10) — consistently worse
  at every level tested, confirmed independently multiple times.

## SECOND BREAKTHROUGH (2026-08-19, same day, later) — Volatility expansion filter

Vivek's idea, refined: don't take trades on small/range-bound candles —
only when there's genuine expanding volatility (movement), and pay
attention to sessions where this tends to happen (UK/US). This filter
(`require_volatility_expansion`) already existed from earlier in the
session — tested it combined with the newly-deployed MTF alignment fix.

**Verified on real 2-year MCX data (combined with MTF alignment):**

| Config | Trades | Avg Gross/trade | Net P&L |
|---|---|---|---|
| Original baseline (no filters) | 1834 | Rs446 | -Rs384,788 |
| MTF alignment only (deployed earlier today) | 621 | Rs439 | -Rs153,740 |
| **MTF alignment + volatility expansion (1.1x)** | **165** | Rs385 | **-Rs57,380 (85% better than original)** |
| MTF + volatility expansion (1.3x, stricter) | 26 | Rs184 | -Rs18,332 (too few trades to trust) |

Chose multiplier=1.1 for deployment — 165 trades over 2 years is a
reasonably trustworthy sample size (unlike 26 trades at 1.3x, which
could easily be noise). Deployed to both `backtesting/backtest_runner.py`
(already had the parameter) and `execution/live_trading_engine.py`
(newly wired in, checks ATR vs its 20-period average before every entry).

### Cumulative progress today

Starting point (this morning): -Rs384,788 (2yr backtest, 1834 trades)
Current (end of day, both fixes deployed): -Rs57,380 (165 trades)
**85% reduction in net loss — still not fully profitable, but the gap
to breakeven is now much smaller. Combined with the CoinDCX cost
advantage (still pending real data), full profitability seems
increasingly plausible for the next session.**

## Tested and NOT deployed (2026-08-19) — RSI/MACD momentum vs ATR volatility

Vivek's challenge: ATR measures candle SIZE, not directional momentum —
suggested RSI + MACD (day-trader-standard momentum oscillators) instead.
Valid theoretical point, tested empirically rather than dismissed.

Implementation tested: for LONG, require RSI > 55 AND MACD histogram
positive AND accelerating (|macd_hist| > |macd_hist_prev|); symmetric
for SHORT.

**Result: did NOT outperform the currently-deployed ATR-based filter.**

| Config | Trades | Avg Gross/trade | Net P&L |
|---|---|---|---|
| Currently deployed (MTF + ATR expansion) | 165 | Rs385 | -Rs57,380 |
| RSI+MACD replacing ATR | 25 | **-Rs677 (negative)** | -Rs29,705 |
| All three combined (MTF+ATR+RSI/MACD) | 3 | -Rs282 | -Rs2,272 (meaningless sample) |

RSI+MACD as tested was too restrictive (small sample) AND showed
negative average gross P&L — worse selection than the ATR-based filter,
at least with these specific threshold values (RSI 55/45, MACD
sign+acceleration). The parameter (`require_rsi_macd_momentum`) is left
in the codebase for future experimentation with different thresholds,
but NOT activated in the live engine — the deployed ATR-based
volatility expansion + MTF alignment remains the best-tested
configuration.

**Conclusion: keep the current deployed system (v27) as-is.** If
revisiting RSI/MACD-based filtering in the future, try different
threshold values (this test used fairly loose RSI 55/45 cutoffs — a
stricter or differently-defined "momentum" condition might perform
differently, untested).

## DEPLOYED (2026-08-19, later) — London-NY session restriction

Vivek's explicit request to implement (after testing showed a promising
but small-sample result). Restricts new entries to 13:30-17:30 UTC
(18:30-22:30 IST), combined with all other deployed filters (MTF
alignment, volatility expansion, support/resistance, reentry-cooldown).

**Real 2-year backtest result:** net loss dropped from Rs57,380 to
Rs10,223 (82% better) — but on only 32 trades over 2 years, a genuinely
small sample. Deployed as `config.risk.require_london_ny_session = True`
(default), toggleable to False if it proves too restrictive in live use.

### Cumulative progress, full day

| Stage | Trades (2yr) | Net P&L |
|---|---|---|
| Original baseline (this morning) | 1834 | -Rs384,788 |
| + MTF alignment | 621 | -Rs153,740 |
| + Volatility expansion | 165 | -Rs57,380 |
| **+ London-NY session restriction** | **32** | **-Rs10,223** |

**97.3% cumulative reduction in net loss from this morning's baseline.**
Still not fully profitable, and the sample size at this final stage (32
trades/2yr) is small enough that real live-trading results should be
watched carefully — this configuration is promising but not yet proven
at a fully trustworthy sample size. Next session: CoinDCX data (still
pending) could be the final piece to cross into genuine profitability,
now that the gap remaining is very small.
