"""
Backtest Runner — replays historical candles through the real pipeline
(structure -> indicators -> situation -> signal -> stop/target -> risk ->
trade manager) using the paper broker for fills. This is what proves
whether the system has genuine edge before ANY real capital touches it.

Stops and targets are computed by the real StopLossEngine/TargetEngine
(structure-aware, multi-candidate) — not a hardcoded placeholder.

v1 simplification: single-timeframe replay by default (5M), with the
higher-timeframe trend either held constant or computed via real
resampling when htf_timeframe is supplied (see build_htf_trend_lookup).

CRITICAL: iterates candles in strict chronological order and only ever
looks at data up to and including the current index — never future
candles. This is the look-ahead-bias guard from Sprint 1 §28.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

from market_structure.structure_engine import (
    MarketStructureEngine, Candle as StructCandle, TrendState, StructureState
)
from indicators.incremental import IndicatorEngine, momentum_health_from_indicator_result
from situation_analysis.situation_analyzer import SituationAnalyzer, IndicatorSnapshot, MacroContext
from signal_engine.signal_engine import SignalEngine, ConfluenceInputs, Decision
from risk_engine.risk_engine import RiskEngine, DailyRiskState
from trade_manager.trade_manager import TradeManager, TradeManagerState
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from execution.broker_adapters.base import OrderRequest, OrderSide
from backtesting.metrics import compute_metrics, BacktestMetrics
from target_engine.stop_target_engine import StopLossEngine, TargetEngine
from execution.brokerage_calculator import calculate_charges, calculate_charges_with_partial_booking


@dataclass
class OHLCV:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trade_log: list   # list of dicts, one per closed trade
    signal_log: list  # every decision including NO_TRADE, for analysis


def build_htf_trend_lookup(base_candles: list[OHLCV], base_timeframe: str,
                             htf_timeframe: str) -> dict[int, TrendState]:
    """
    Resamples base_candles into htf_timeframe and runs them through a
    separate MarketStructureEngine to get a real trend per COMPLETED bucket.

    Returns {bucket_start_ts: trend_as_of_that_bucket_closing}. Only
    completed buckets are included — this is the look-ahead-bias guard:
    a base candle can only ever see the HTF trend from the most recently
    CLOSED higher-timeframe candle, never one still in progress.
    """
    from market_data.resampler import resample

    resampled = resample(base_candles, base_timeframe, htf_timeframe)
    htf_structure = MarketStructureEngine()
    htf_atr_placeholder = 1.0

    lookup = {}
    for rc in resampled:
        if not rc.is_complete:
            continue  # never use an in-progress HTF candle's trend
        struct_candle = StructCandle(ts=rc.ohlcv.ts, open=rc.ohlcv.open, high=rc.ohlcv.high,
                                       low=rc.ohlcv.low, close=rc.ohlcv.close, volume=rc.ohlcv.volume)
        state = htf_structure.update(struct_candle, current_atr=htf_atr_placeholder)
        lookup[rc.ohlcv.ts] = state.trend
    return lookup


def _htf_trend_as_of(base_ts: int, htf_lookup: dict[int, TrendState],
                       default: TrendState = TrendState.RANGE) -> TrendState:
    """Finds the most recent COMPLETED HTF bucket at or before base_ts.
    Never looks forward — if no HTF data exists yet, defaults to RANGE
    (conservative: an unknown trend should not bias the confluence score)."""
    applicable = [ts for ts in htf_lookup if ts <= base_ts]
    if not applicable:
        return default
    return htf_lookup[max(applicable)]


def run_backtest(candles: list[OHLCV], config, htf_trend_override: TrendState | None = None,
                  base_timeframe: str = "5M", htf_timeframe: str | None = None,
                  cooldown_days_after_disable: int | None = 1,
                  same_direction_reentry_cooldown_sec: int | None = None,
                  require_near_support_resistance: bool | None = None,
                  support_resistance_proximity_atr_mult: float = 1.5,
                  require_london_ny_overlap: bool | None = None,
                  use_wider_movement_windows: bool = False,
                  use_split_morning_evening_caps: bool = False,
                  require_volatility_expansion: bool | None = None,
                  volatility_expansion_mult: float = 1.3,
                  require_rsi_macd_momentum: bool | None = None,
                  require_gold_tactical_cci_confirmation: bool | None = None,
                  starting_equity_inr: float = 500_000.0,
                  base_trades_normal_threshold: int | None = None,
                  extended_trades_min_confidence: float = 80.0,
                  extended_trades_require_uk_us_session: bool = False,
                  exceptional_conviction_threshold: float | None = None,
                  exceptional_conviction_risk_multiplier: float = 2.0,
                  require_multi_timeframe_alignment: bool | None = None,
                  verbose: bool = False) -> BacktestResult:
    """
    base_trades_normal_threshold: if set (e.g. 4), the first N trades each
    day use the normal confluence threshold (config.thresholds.no_trade_max)
    as usual. Trades BEYOND that count (up to max_trades_per_day) require
    a MUCH higher confidence score (extended_trades_min_confidence) to
    fire — reserving the "extra" daily quota specifically for
    high-conviction, likely-bigger-movement setups, rather than diluting
    it with more average-quality trades (which was tested and shown to
    make things worse). None (default) preserves original behavior.
    require_london_ny_overlap: if True, only allows new entries between
    13:00-17:00 UTC (the London-New York session overlap — the period of
    peak institutional gold volume). Found via real-data analysis: trades
    in this window showed +0.844R expectancy / PF 3.05 vs +0.477R / PF
    2.24 outside it — both profitable, but the overlap window notably
    stronger. None (default) preserves original behavior.
    candles: chronologically ordered OHLCV list (already historical, no
    forward-looking data).

    same_direction_reentry_cooldown_sec: if set, blocks a new entry in the
    SAME direction as the most recent MOMENTUM_DECAY exit, for this many
    seconds after that exit. Found via real-data analysis: re-entering the
    same direction shortly after a momentum-decay exit performs far worse
    than fresh entries (LONG: +0.101R vs +0.378R; SHORT: -0.100R vs
    +0.647R on real 2-year MCX data) — the opposite direction is NOT
    blocked, since momentum decay doesn't mean "avoid trading," just
    "don't immediately chase the same direction again." None (default)
    preserves the original behavior for backward compatibility.

    Two ways to supply the higher-timeframe trend:
    - htf_trend_override: hold it constant (fast, simplified — useful for
      component smoke tests, NOT for real strategy validation).
    - htf_timeframe (e.g. "1H"): resample the SAME base candles into that
      timeframe and compute a real, look-ahead-safe trend per candle. This
      is the correct mode for genuine multi-timeframe backtesting.
    If both are omitted, defaults to RANGE (conservative, no bias).

    cooldown_days_after_disable: after MAX_CONSECUTIVE_LOSSES disables
    trading, the backtest simulates a realistic operator reviewing and
    resuming after this many calendar days (default 1 = next trading day).
    Set to None to disable this simulation and let a disable persist for
    the rest of the backtest, matching literal live-trading behavior where
    a human must manually intervene.
    """
    htf_lookup = None
    if htf_timeframe is not None:
        htf_lookup = build_htf_trend_lookup(candles, base_timeframe, htf_timeframe)

    # For multi-timeframe alignment: also build a 15M trend lookup
    # (independent of htf_timeframe, which is typically 1H/4H) — used to
    # require 15M AND 1H trend to agree with the base-timeframe direction
    # before entry, per Vivek's request to check price action across
    # multiple timeframes (1M excluded — no real 1-minute data exists).
    mtf_15m_lookup = None
    if require_multi_timeframe_alignment:
        mtf_15m_lookup = build_htf_trend_lookup(candles, base_timeframe, "15M")

    structure = MarketStructureEngine()
    indicators = IndicatorEngine()
    situation_analyzer = SituationAnalyzer(config)
    signal_engine = SignalEngine(config)
    risk_engine = RiskEngine(config, DailyRiskState())
    stop_engine = StopLossEngine(config)
    target_engine = TargetEngine(config)
    broker = PaperBrokerProvider(starting_equity_inr=starting_equity_inr)
    broker.connect()

    trade_log = []
    signal_log = []
    open_trade_manager: TradeManager | None = None
    open_trade_entry_regime: str | None = None   # tracks the regime active at entry, for regime analytics
    r_multiples: list[float] = []
    # See same_direction_reentry_cooldown_sec docstring — tracks the most
    # recent MOMENTUM_DECAY exit so a same-direction re-entry can be
    # gated for a cooldown window.
    last_momentum_decay_exit: dict | None = None
    # BUGFIX: max_trades_per_day / daily_pnl were never reset across calendar
    # day boundaries, so "4 trades per day" silently became "4 trades across
    # the ENTIRE backtest" — every prior multi-day/multi-year backtest result
    # was capped at exactly max_trades_per_day trades regardless of how much
    # data was fed in, which invalidated every metric produced so far.
    current_day: object = None
    morning_window_trades_today = 0
    evening_window_trades_today = 0
    disabled_since_day: object = None

    for i, candle in enumerate(candles):
        # BUGFIX: reset daily risk counters at each calendar day boundary.
        # candle.ts is epoch seconds — derive the calendar date to detect
        # a new trading day and reset trades_taken_today/daily_pnl_inr,
        # matching what a real system would do at each session start.
        candle_day = datetime.fromtimestamp(candle.ts, tz=timezone.utc).date()
        if current_day is None:
            current_day = candle_day
        elif candle_day != current_day:
            risk_engine.state.trades_taken_today = 0
            risk_engine.state.daily_pnl_inr = 0.0
            morning_window_trades_today = 0
            evening_window_trades_today = 0
            # BUGFIX: trading_disabled (set after max_consecutive_losses) had
            # no reset path anywhere — a real system correctly requires manual
            # human review before resuming (per spec), but a backtest replay
            # has no human, so it silently simulated "shut down forever after
            # one bad streak" for the rest of the dataset. Real 2-year test
            # data showed exactly this: trading stopped after 18 trades near
            # the start and never resumed for ~64,000 remaining candles.
            # Simulating a realistic operator reviewing and resuming after a
            # cooldown (default: next trading day) makes the backtest reflect
            # the strategy's ongoing behavior rather than one early streak.
            if risk_engine.state.trading_disabled and cooldown_days_after_disable is not None:
                if disabled_since_day is not None:
                    days_since_disable = (candle_day - disabled_since_day).days
                    if days_since_disable >= cooldown_days_after_disable:
                        risk_engine.manual_reset()
                        disabled_since_day = None
            current_day = candle_day

        if risk_engine.state.trading_disabled and disabled_since_day is None:
            disabled_since_day = current_day

        if htf_lookup is not None:
            htf_trend = _htf_trend_as_of(candle.ts, htf_lookup)
        else:
            htf_trend = htf_trend_override or TrendState.RANGE

        struct_state = structure.update(
            StructCandle(ts=candle.ts, open=candle.open, high=candle.high,
                         low=candle.low, close=candle.close, volume=candle.volume),
            current_atr=indicators.atr.value or 1.0,
            higher_tf_trend=htf_trend,
        )
        ind_result = indicators.update(candle.high, candle.low, candle.close, candle.volume)
        broker.set_quote("GOLDM", ltp=candle.close, volume=candle.volume)

        # ---- manage an already-open trade first ----
        if open_trade_manager is not None:
            momentum_health = momentum_health_from_indicator_result(ind_result)
            structure_broke = struct_state.last_event.value.startswith("CHOCH") and (
                (open_trade_manager.state.direction == "LONG" and "BEARISH" in struct_state.last_event.value) or
                (open_trade_manager.state.direction == "SHORT" and "BULLISH" in struct_state.last_event.value)
            )

            open_trade_manager.check_partial_booking(candle.close)
            open_trade_manager.update_trailing_stop(
                candle.close, ind_result["ema9"], ind_result["ema21"], ind_result["ema50"],
                ind_result["atr"], momentum_health, structure_broke,
            )
            if open_trade_manager.state.trade_state.value != "EXITED":
                # BUGFIX: intrabar check — a candle that pierced the stop and
                # recovered must still register as a stop-out
                open_trade_manager.check_stop_hit_intrabar(
                    high=candle.high, low=candle.low, close=candle.close)

            if open_trade_manager.state.trade_state.value == "EXITED":
                # exit price is whatever the state machine actually closed at
                # (stop price on an intrabar hit, not the candle close)
                exit_price = (open_trade_manager.state.state_history[-1].price_at_event
                              if open_trade_manager.state.state_history else candle.close)
                # BUGFIX: blended R accounts for partials booked along the way
                r = open_trade_manager.blended_r_multiple(exit_price)
                r_multiples.append(r)
                risk_engine.record_trade_result(r * config.risk.max_risk_per_trade_inr)
                charges = calculate_charges_with_partial_booking(
                    direction=open_trade_manager.state.direction,
                    entry_price=open_trade_manager.state.entry_price,
                    original_risk_points=open_trade_manager.state.original_risk_points,
                    realized_legs=open_trade_manager.state.realized_legs,
                    final_exit_price=exit_price,
                    quantity_remaining_pct=open_trade_manager.state.quantity_remaining_pct,
                    total_lots=open_trade_lots,
                    point_value_inr=config.instrument.point_value_inr,
                )
                trade_log.append({
                    "entry_price": open_trade_manager.state.entry_price,
                    "exit_price": exit_price,
                    "direction": open_trade_manager.state.direction,
                    "r_multiple": r,
                    "exit_reason": open_trade_manager.state.exit_reason.value,
                    "ts": candle.ts,
                    "entry_regime": open_trade_entry_regime,
                    "lots": open_trade_lots,
                    "gross_pnl_inr": charges.gross_pnl_inr,
                    "total_charges_inr": charges.total_charges_inr,
                    "net_pnl_inr": charges.net_pnl_inr,
                })
                risk_engine.register_position_closed()
                # track for the same-direction re-entry cooldown check
                if open_trade_manager.state.exit_reason.value == "MOMENTUM_DECAY":
                    last_momentum_decay_exit = {
                        "direction": open_trade_manager.state.direction, "ts": candle.ts,
                    }
                open_trade_manager = None
                open_trade_entry_regime = None
            continue  # don't evaluate a new entry the same candle we're managing an exit

        # ---- otherwise, evaluate for a new entry ----
        if i < 30:
            continue  # let indicators warm up before trusting them

        ind_snapshot = IndicatorSnapshot(
            ema9=ind_result["ema9"], ema21=ind_result["ema21"], ema50=ind_result["ema50"],
            ema200=ind_result["ema200"], rsi=ind_result["rsi"],
            macd_hist=ind_result["macd_hist"], macd_hist_prev=ind_result["macd_hist_prev"],
            atr=ind_result["atr"], atr_avg_20=ind_result["atr_avg_20"],
            rel_volume=ind_result["rel_volume"],
        )
        macro = MacroContext()  # no live macro feed in this simplified backtest pass

        # BUGFIX: previously passed `struct_state` as BOTH the higher- and
        # lower-timeframe structure. _trend_alignment_score then compared a
        # trend against itself, so it always returned 90 (or 40 in RANGE) and
        # could NEVER detect a cross-timeframe conflict (score 15). That made
        # the entire multi-timeframe apparatus inert: htf_timeframe='1H' and
        # htf_trend_override=RANGE produced byte-identical results (1920
        # trades, 0.272R, PF 1.73), and htf_trend_alignment — 15% of the
        # confluence weight — was effectively a constant.
        htf_struct_state = StructureState()
        htf_struct_state.trend = htf_trend
        situation = situation_analyzer.analyze(htf_struct_state, struct_state, ind_snapshot, macro)

        conf_inputs = ConfluenceInputs(
            ltf_structure=struct_state, situation=situation,
            fair_value=_neutral_fair_value(candle.close), macro=_neutral_macro(),
            ema_aligned_bullish=ind_result["ema9"] > ind_result["ema21"] > ind_result["ema50"],
            ema_aligned_bearish=ind_result["ema9"] < ind_result["ema21"] < ind_result["ema50"],
            macd_bullish=ind_result["macd_hist"] > 0,
            macd_bearish=ind_result["macd_hist"] < 0,
            rsi=ind_result["rsi"], price_above_vwap=candle.close > (ind_result["vwap"] or candle.close),
            volume_supportive=ind_result["rel_volume"] >= 1.0,
            bb_squeeze=(ind_result["bb_upper"] - ind_result["bb_lower"]) < ind_result["atr"] * 2,
            session_quality_ok=True,   # no live session feed in backtest replay
        )
        result = signal_engine.evaluate(conf_inputs)
        signal_log.append({"ts": candle.ts, "decision": result.decision.value,
                             "long_score": result.long_score, "short_score": result.short_score})

        if result.decision == Decision.NO_TRADE:
            continue

        # Reserve the "extended" daily quota (beyond base_trades_normal_threshold)
        # for genuinely high-conviction setups only — don't waste it on
        # average-quality trades just because the day's early quota is used up.
        if base_trades_normal_threshold is not None:
            trades_today_so_far = risk_engine.state.trades_taken_today
            if trades_today_so_far >= base_trades_normal_threshold:
                if result.confidence < extended_trades_min_confidence:
                    continue
                if extended_trades_require_uk_us_session:
                    entry_hour_utc = datetime.fromtimestamp(candle.ts, tz=timezone.utc).hour
                    # UK afternoon + US session, roughly 13:00-21:00 UTC
                    if not (13 <= entry_hour_utc < 21):
                        continue

        direction = "LONG" if result.decision == Decision.BUY else "SHORT"

        if require_multi_timeframe_alignment and mtf_15m_lookup is not None and htf_lookup is not None:
            trend_15m = _htf_trend_as_of(candle.ts, mtf_15m_lookup)
            trend_1h = _htf_trend_as_of(candle.ts, htf_lookup)
            wanted_trend = TrendState.TRENDING_UP if direction == "LONG" else TrendState.TRENDING_DOWN
            if trend_15m != wanted_trend or trend_1h != wanted_trend:
                continue   # 15M and 1H must BOTH agree with the entry direction

        if (same_direction_reentry_cooldown_sec is not None and last_momentum_decay_exit
                and last_momentum_decay_exit["direction"] == direction
                and candle.ts - last_momentum_decay_exit["ts"] <= same_direction_reentry_cooldown_sec):
            continue   # blocked: chasing the same direction right after it decayed

        entry_price = candle.close
        nearest_swing_low = (struct_state.swing_lows[-1].price if struct_state.swing_lows else None)
        nearest_swing_high = (struct_state.swing_highs[-1].price if struct_state.swing_highs else None)

        if require_london_ny_overlap:
            entry_dt = datetime.fromtimestamp(candle.ts, tz=timezone.utc)
            entry_minutes_utc = entry_dt.hour * 60 + entry_dt.minute
            if use_wider_movement_windows:
                # Vivek's own market observation: morning (9-11 AM IST =
                # 3:30-5:30 UTC) and evening (4-9 PM IST = 10:30-15:30
                # UTC) both show significant movement — wider than the
                # narrow London-NY-only window.
                in_morning = (3 * 60 + 30 <= entry_minutes_utc < 5 * 60 + 30)
                in_evening = (10 * 60 + 30 <= entry_minutes_utc < 15 * 60 + 30)
                if not (in_morning or in_evening):
                    continue
                if use_split_morning_evening_caps:
                    if in_morning and morning_window_trades_today >= 2:
                        continue
                    if in_evening and evening_window_trades_today >= 4:
                        continue
            else:
                # 13:30-17:30 UTC (18:30-22:30 IST) — the specific
                # London-NY overlap window Vivek originally requested.
                if not (13 * 60 + 30 <= entry_minutes_utc < 17 * 60 + 30):
                    continue

        # Real-data finding: LONG entries far from a recent swing low
        # (support) and SHORT entries far from a recent swing high
        # (resistance) were, as a group, net LOSERS (-0.059R and -0.079R
        # respectively) on 2-year real MCX data, while entries near
        # support/resistance were strongly profitable (+0.913R / +0.363R).
        # "Wait when there's no movement, act when there is" — only enter
        # when CURRENT volatility (ATR) is genuinely EXPANDING relative to
        # its recent average, not just when confluence score passes.
        # This directly targets Vivek's observation: the system was taking
        # trades during quiet periods and missing genuinely large moves.
        if require_volatility_expansion:
            atr_avg = ind_result["atr_avg_20"] or ind_result["atr"] or 1.0
            if ind_result["atr"] < atr_avg * volatility_expansion_mult:
                continue

        # Gold Tactical CCI momentum-confirmation (Vivek's shared Pine
        # Script, reimplemented — see indicators/incremental.py
        # GoldTacticalCCIState). Requires the CCI oscillator to have JUST
        # crossed its extreme threshold in the SAME direction as the
        # proposed entry — confirms genuine, extended momentum is
        # actively building, not just present.
        if require_gold_tactical_cci_confirmation:
            if direction == "LONG":
                if not ind_result.get("gold_tactical_cci_crossed_above"):
                    continue
            else:
                if not ind_result.get("gold_tactical_cci_crossed_below"):
                    continue

        # Vivek's alternative to ATR-based "is there movement": ATR only
        # measures candle SIZE, not genuine directional PUSH — a market
        # can have big, choppy, non-directional candles. RSI + MACD are
        # momentum oscillators that specifically show directional
        # conviction, which day traders traditionally use over ATR
        # (a volatility/range measure more associated with position
        # sizing / longer-horizon stop placement).
        if require_rsi_macd_momentum:
            rsi = ind_result["rsi"]
            macd_hist = ind_result["macd_hist"]
            macd_hist_prev = ind_result["macd_hist_prev"]
            macd_accelerating = abs(macd_hist) > abs(macd_hist_prev)
            if direction == "LONG":
                if not (rsi > 55 and macd_hist > 0 and macd_accelerating):
                    continue
            else:
                if not (rsi < 45 and macd_hist < 0 and macd_accelerating):
                    continue

        if require_near_support_resistance:
            atr_now = ind_result["atr"] or 10.0
            proximity = atr_now * support_resistance_proximity_atr_mult
            if direction == "LONG":
                if nearest_swing_low is None or abs(entry_price - nearest_swing_low) > proximity:
                    continue
            else:
                if nearest_swing_high is None or abs(entry_price - nearest_swing_high) > proximity:
                    continue

        stop_result = stop_engine.evaluate(
            direction=direction, entry_price=entry_price, atr=ind_result["atr"],
            nearest_swing_low=nearest_swing_low, nearest_swing_high=nearest_swing_high,
        )
        if not stop_result.approved:
            continue  # market structure doesn't fit the risk budget — NO_TRADE, not a forced stop

        stop_price = stop_result.price
        stop_distance = stop_result.distance_points

        target_result = target_engine.calculate(
            direction=direction, entry_price=entry_price, stop_price=stop_price,
            nearest_resistance=nearest_swing_high, nearest_support=nearest_swing_low,
            atr=ind_result["atr"],
        )
        if target_result is None:
            continue

        risk_reward = abs(target_result.target_2 - entry_price) / stop_distance

        veto = risk_engine.check_hard_limits(live_equity_inr=broker.get_balance().equity_inr,
                                               data_is_stale=False, position_already_open=False)
        if veto.value != "NONE":
            continue

        sizing = risk_engine.calculate_position_size(
            entry_price=entry_price, stop_price=stop_price,
            live_equity_inr=broker.get_balance().equity_inr, risk_reward=risk_reward,
        )
        # EXPERIMENTAL, explicitly requested test: normally confidence
        # affects whether to trade, never how much to risk (deliberate
        # design principle — avoids over-trusting the confluence score's
        # magnitude, which has NOT been shown to reliably predict move
        # SIZE, only win/loss, per earlier same-day testing). Testing
        # this anyway on explicit request: does giving genuinely
        # EXCEPTIONAL setups (score >= threshold) a bigger risk budget,
        # mimicking discretionary-trader judgment, actually help?
        if (exceptional_conviction_threshold is not None
                and result.confidence >= exceptional_conviction_threshold
                and sizing.approved):
            boosted_risk_budget = config.risk.max_risk_per_trade_inr * exceptional_conviction_risk_multiplier
            risk_per_lot_inr = abs(entry_price - stop_price) * config.instrument.point_value_inr
            if risk_per_lot_inr > 0:
                boosted_lots = min(int(boosted_risk_budget / risk_per_lot_inr), config.risk.max_lots_cap)
                if boosted_lots > sizing.lots:
                    sizing.lots = boosted_lots
        if not sizing.approved:
            continue

        side = OrderSide.BUY if result.decision == Decision.BUY else OrderSide.SELL
        order = OrderRequest(client_order_id=f"BT-{i}", symbol="GOLDM", side=side,
                              quantity=sizing.lots)
        fill = broker.place_order(order)
        if fill.status.value != "FILLED":
            continue

        tm_state = TradeManagerState(
            direction=direction, entry_price=fill.filled_price,
            original_stop=stop_price, current_stop=stop_price,
            original_risk_points=stop_distance,
            target_1=target_result.target_1, target_2=target_result.target_2,
            target_3=target_result.target_3,
        )
        open_trade_manager = TradeManager(config, tm_state)
        open_trade_entry_regime = situation.regime.value
        open_trade_lots = sizing.lots
        risk_engine.register_position_opened()
        if use_split_morning_evening_caps and require_london_ny_overlap and use_wider_movement_windows:
            entry_minutes_check = datetime.fromtimestamp(candle.ts, tz=timezone.utc).hour * 60 + \
                                     datetime.fromtimestamp(candle.ts, tz=timezone.utc).minute
            if 3 * 60 + 30 <= entry_minutes_check < 5 * 60 + 30:
                morning_window_trades_today += 1
            elif 10 * 60 + 30 <= entry_minutes_check < 15 * 60 + 30:
                evening_window_trades_today += 1


    metrics = compute_metrics(r_multiples)
    return BacktestResult(metrics=metrics, trade_log=trade_log, signal_log=signal_log)


def _neutral_fair_value(price: float):
    from gold_intelligence.fair_value import FairValueResult
    return FairValueResult(mcx_price=price, theoretical_price=price, deviation=0,
                             deviation_pct=0, deviation_zscore=0.0, is_reliable=False,
                             unreliable_reason="No live macro feed in this backtest pass")


def _neutral_macro():
    from gold_intelligence.fair_value import MacroBiasResult
    return MacroBiasResult(macro_bias=0.0, components={"session_quality_ok": True})
