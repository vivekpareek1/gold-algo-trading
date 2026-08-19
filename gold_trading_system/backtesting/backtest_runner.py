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
                  verbose: bool = False) -> BacktestResult:
    """
    require_london_ny_overlap: if True, only allows new entries between
    13:00-17:00 UTC (the London-New York session overlap — the period of
    peak institutional gold volume). Found via real-data analysis: trades
    in this window showed +0.844R expectancy / PF 3.05 vs +0.477R / PF
    2.24 outside it — both profitable, but the overlap window notably
    stronger. None (default) preserves original behavior.
    """
    """
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

    structure = MarketStructureEngine()
    indicators = IndicatorEngine()
    situation_analyzer = SituationAnalyzer(config)
    signal_engine = SignalEngine(config)
    risk_engine = RiskEngine(config, DailyRiskState())
    stop_engine = StopLossEngine(config)
    target_engine = TargetEngine(config)
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
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

        direction = "LONG" if result.decision == Decision.BUY else "SHORT"

        if (same_direction_reentry_cooldown_sec is not None and last_momentum_decay_exit
                and last_momentum_decay_exit["direction"] == direction
                and candle.ts - last_momentum_decay_exit["ts"] <= same_direction_reentry_cooldown_sec):
            continue   # blocked: chasing the same direction right after it decayed

        entry_price = candle.close
        nearest_swing_low = (struct_state.swing_lows[-1].price if struct_state.swing_lows else None)
        nearest_swing_high = (struct_state.swing_highs[-1].price if struct_state.swing_highs else None)

        if require_london_ny_overlap:
            entry_hour_utc = datetime.fromtimestamp(candle.ts, tz=timezone.utc).hour
            if not (13 <= entry_hour_utc < 17):
                continue

        # Real-data finding: LONG entries far from a recent swing low
        # (support) and SHORT entries far from a recent swing high
        # (resistance) were, as a group, net LOSERS (-0.059R and -0.079R
        # respectively) on 2-year real MCX data, while entries near
        # support/resistance were strongly profitable (+0.913R / +0.363R).
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
