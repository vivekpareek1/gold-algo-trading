"""
Live/Paper Trading Engine — the actual continuous trading loop, missing
until now. The backtest_runner proved the pipeline works on historical
replay; this module runs the SAME logic (structure -> indicators ->
situation -> signal -> stop/target -> risk -> trade manager) tick-by-tick
against a live/simulated feed, with state that PERSISTS across calls
rather than resetting each time — which is what "paper trading" actually
requires and what api/main.py's snapshot endpoints did not do.

Every bug fix from the backtest validation applies here identically:
intrabar stop checks, blended R on partials, daily counter resets,
cooldown-based resume after a disable, and real position tracking.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

from market_structure.structure_engine import (
    MarketStructureEngine, Candle as StructCandle, TrendState, StructureState
)
from indicators.incremental import IndicatorEngine
from situation_analysis.situation_analyzer import SituationAnalyzer, IndicatorSnapshot, MacroContext
from signal_engine.signal_engine import SignalEngine, ConfluenceInputs, Decision
from risk_engine.risk_engine import RiskEngine, DailyRiskState
from trade_manager.trade_manager import TradeManager, TradeManagerState
from target_engine.stop_target_engine import StopLossEngine, TargetEngine
from execution.broker_adapters.base import BrokerProvider, OrderRequest, OrderSide
from gold_intelligence.fair_value import FairValueResult, MacroBiasResult


@dataclass
class LiveTick:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class _LiveHTFAggregator:
    """
    Builds genuine higher-timeframe candles from the incoming tick stream and
    maintains a SEPARATE MarketStructureEngine on them, so the live engine has
    a real higher-timeframe trend rather than a placeholder.

    Only COMPLETED buckets are fed to the structure engine — the in-progress
    bucket is never used, which is the same look-ahead guard the backtest
    resampler applies.
    """

    def __init__(self, htf_minutes: int = 60):
        self.htf_minutes = htf_minutes
        self.htf_structure = MarketStructureEngine()
        self._bucket_start: int | None = None
        self._o = self._h = self._l = self._c = None
        self._v = 0.0
        self.current_trend: TrendState = TrendState.RANGE

    def _bucket_of(self, ts: int) -> int:
        span = self.htf_minutes * 60
        return (ts // span) * span

    def update(self, tick) -> TrendState:
        b = self._bucket_of(tick.ts)

        if self._bucket_start is None:
            self._start_bucket(b, tick)
            return self.current_trend

        if b != self._bucket_start:
            # the previous bucket has fully elapsed — only now is it safe to use
            completed = StructCandle(ts=self._bucket_start, open=self._o, high=self._h,
                                       low=self._l, close=self._c, volume=self._v)
            state = self.htf_structure.update(completed,
                                                current_atr=max(self._h - self._l, 1e-6))
            self.current_trend = state.trend
            self._start_bucket(b, tick)
        else:
            self._h = max(self._h, tick.high)
            self._l = min(self._l, tick.low)
            self._c = tick.close
            self._v += tick.volume

        return self.current_trend

    def _start_bucket(self, bucket_start: int, tick):
        self._bucket_start = bucket_start
        self._o, self._h, self._l, self._c = tick.open, tick.high, tick.low, tick.close
        self._v = tick.volume


@dataclass
class LiveEngineState:
    tick_count: int = 0
    current_day: object = None
    disabled_since_day: object = None
    open_trade_manager: TradeManager | None = None
    open_trade_entry_regime: str | None = None
    last_snapshot: dict = field(default_factory=dict)
    trade_log: list = field(default_factory=list)
    signal_log: list = field(default_factory=list)


class LiveTradingEngine:
    """
    One instance per (instrument, session). Call on_tick() for every new
    closed candle. All state persists on self between calls — this is the
    core difference from the old api/main.py endpoints, which recreated
    the "does a signal exist right now" question from scratch each request
    without ever actually opening/managing/closing a position.
    """

    def __init__(self, config, broker: BrokerProvider, symbol: str = "GOLDM",
                 htf_timeframe_lookup=None, cooldown_days_after_disable: int | None = 1):
        self.config = config
        self.broker = broker
        self.symbol = symbol
        self.cooldown_days_after_disable = cooldown_days_after_disable
        # htf_timeframe_lookup: optional callable(ts) -> TrendState. If not
        # supplied, the engine now builds its OWN higher-timeframe trend from
        # the incoming tick stream via _LiveHTFAggregator, rather than falling
        # back to a constant RANGE placeholder (which silently disabled the
        # multi-timeframe layer the strategy was validated with).
        self.htf_lookup_fn = htf_timeframe_lookup
        self.htf_aggregator = _LiveHTFAggregator(htf_minutes=60) if htf_timeframe_lookup is None else None

        self.structure = MarketStructureEngine()
        self.indicators = IndicatorEngine()
        self.situation_analyzer = SituationAnalyzer(config)
        self.signal_engine = SignalEngine(config)
        self.risk_engine = RiskEngine(config, DailyRiskState())
        self.stop_engine = StopLossEngine(config)
        self.target_engine = TargetEngine(config)

        self.state = LiveEngineState()

    def on_tick(self, tick: LiveTick) -> dict:
        """
        Processes one new CLOSED candle. Returns a snapshot dict suitable
        for API/dashboard consumption. This is the single entry point —
        everything else is internal.
        """
        self.state.tick_count += 1

        self._handle_day_boundary(tick)

        struct_candle = StructCandle(ts=tick.ts, open=tick.open, high=tick.high,
                                       low=tick.low, close=tick.close, volume=tick.volume)
        if self.htf_lookup_fn is not None:
            htf_trend = self.htf_lookup_fn(tick.ts)
        else:
            htf_trend = self.htf_aggregator.update(tick)
        self._current_htf_trend = htf_trend
        struct_state = self.structure.update(struct_candle,
                                                current_atr=self.indicators.atr.value or 1.0,
                                                higher_tf_trend=htf_trend)
        ind_result = self.indicators.update(tick.high, tick.low, tick.close, tick.volume)
        self.broker.set_quote(self.symbol, ltp=tick.close, volume=tick.volume) \
            if hasattr(self.broker, "set_quote") else None

        if self.state.open_trade_manager is not None:
            self._manage_open_trade(tick, ind_result, struct_state)
        elif self.state.tick_count > 30:  # let indicators warm up
            self._evaluate_new_entry(tick, ind_result, struct_state)

        snapshot = self._build_snapshot(tick, struct_state, ind_result)
        self.state.last_snapshot = snapshot
        return snapshot

    # ---------- day boundary handling (identical logic to backtest_runner) ----------
    def _handle_day_boundary(self, tick: LiveTick):
        day = datetime.fromtimestamp(tick.ts, tz=timezone.utc).date()
        if self.state.current_day is None:
            self.state.current_day = day
            return
        if day == self.state.current_day:
            return

        self.risk_engine.state.trades_taken_today = 0
        self.risk_engine.state.daily_pnl_inr = 0.0

        if (self.risk_engine.state.trading_disabled
                and self.cooldown_days_after_disable is not None
                and self.state.disabled_since_day is not None):
            days_since = (day - self.state.disabled_since_day).days
            if days_since >= self.cooldown_days_after_disable:
                self.risk_engine.manual_reset()
                self.state.disabled_since_day = None

        self.state.current_day = day

    # ---------- managing an open position ----------
    def _manage_open_trade(self, tick: LiveTick, ind_result: dict, struct_state):
        tm = self.state.open_trade_manager
        momentum_health = self._momentum_from_indicators(ind_result)
        structure_broke = struct_state.last_event.value.startswith("CHOCH") and (
            (tm.state.direction == "LONG" and "BEARISH" in struct_state.last_event.value) or
            (tm.state.direction == "SHORT" and "BULLISH" in struct_state.last_event.value)
        )

        tm.check_partial_booking(tick.close)
        tm.update_trailing_stop(tick.close, ind_result["ema9"], ind_result["ema21"],
                                  ind_result["ema50"], ind_result["atr"], momentum_health,
                                  structure_broke)
        if tm.state.trade_state.value != "EXITED":
            tm.check_stop_hit_intrabar(high=tick.high, low=tick.low, close=tick.close)

        if tm.state.trade_state.value == "EXITED":
            exit_price = (tm.state.state_history[-1].price_at_event
                          if tm.state.state_history else tick.close)
            r = tm.blended_r_multiple(exit_price)
            self.risk_engine.record_trade_result(r * self.config.risk.max_risk_per_trade_inr)
            if self.risk_engine.state.trading_disabled and self.state.disabled_since_day is None:
                self.state.disabled_since_day = self.state.current_day

            self.state.trade_log.append({
                "entry_price": tm.state.entry_price, "exit_price": exit_price,
                "direction": tm.state.direction, "r_multiple": r,
                "exit_reason": tm.state.exit_reason.value, "ts": tick.ts,
                "entry_regime": self.state.open_trade_entry_regime,
            })
            # bounded like signal_log — a long-running live session must not
            # grow memory without limit. Persist to the DB layer for full history.
            self.state.trade_log = self.state.trade_log[-1000:]
            self.risk_engine.register_position_closed()
            self.state.open_trade_manager = None
            self.state.open_trade_entry_regime = None

    # ---------- evaluating a new entry ----------
    def _evaluate_new_entry(self, tick: LiveTick, ind_result: dict, struct_state):
        ind_snapshot = IndicatorSnapshot(
            ema9=ind_result["ema9"], ema21=ind_result["ema21"], ema50=ind_result["ema50"],
            ema200=ind_result["ema200"], rsi=ind_result["rsi"],
            macd_hist=ind_result["macd_hist"], macd_hist_prev=ind_result["macd_hist_prev"],
            atr=ind_result["atr"], atr_avg_20=ind_result["atr_avg_20"],
            rel_volume=ind_result["rel_volume"],
        )
        # BUGFIX (same as backtest_runner): passing struct_state as both the
        # higher- and lower-timeframe structure made trend_alignment_score
        # compare a trend to itself, permanently pinning it at 90/40 and
        # making cross-timeframe conflict undetectable.
        htf_struct_state = StructureState()
        htf_struct_state.trend = getattr(self, "_current_htf_trend", TrendState.RANGE)
        situation = self.situation_analyzer.analyze(htf_struct_state, struct_state, ind_snapshot,
                                                       MacroContext())

        conf_inputs = ConfluenceInputs(
            ltf_structure=struct_state, situation=situation,
            fair_value=FairValueResult(mcx_price=tick.close, theoretical_price=tick.close,
                                         deviation=0, deviation_pct=0, deviation_zscore=0.0,
                                         is_reliable=False),
            macro=MacroBiasResult(macro_bias=0.0, components={"session_quality_ok": True}),
            ema_aligned_bullish=ind_result["ema9"] > ind_result["ema21"] > ind_result["ema50"],
            ema_aligned_bearish=ind_result["ema9"] < ind_result["ema21"] < ind_result["ema50"],
            macd_bullish=ind_result["macd_hist"] > 0, macd_bearish=ind_result["macd_hist"] < 0,
            rsi=ind_result["rsi"], price_above_vwap=tick.close > (ind_result["vwap"] or tick.close),
            volume_supportive=ind_result["rel_volume"] >= 1.0,
            bb_squeeze=(ind_result["bb_upper"] - ind_result["bb_lower"]) < ind_result["atr"] * 2,
            session_quality_ok=True,
        )
        result = self.signal_engine.evaluate(conf_inputs)
        self.state.signal_log.append({"ts": tick.ts, "decision": result.decision.value,
                                        "long_score": result.long_score, "short_score": result.short_score})
        self.state.signal_log = self.state.signal_log[-500:]  # bounded memory for a long-running session

        if result.decision == Decision.NO_TRADE:
            return

        direction = "LONG" if result.decision == Decision.BUY else "SHORT"
        nsl = struct_state.swing_lows[-1].price if struct_state.swing_lows else None
        nsh = struct_state.swing_highs[-1].price if struct_state.swing_highs else None

        stop_result = self.stop_engine.evaluate(direction=direction, entry_price=tick.close,
                                                    atr=ind_result["atr"],
                                                    nearest_swing_low=nsl, nearest_swing_high=nsh)
        if not stop_result.approved:
            return

        target_result = self.target_engine.calculate(
            direction=direction, entry_price=tick.close, stop_price=stop_result.price,
            nearest_resistance=nsh, nearest_support=nsl, atr=ind_result["atr"],
        )
        if target_result is None:
            return

        risk_reward = abs(target_result.target_2 - tick.close) / stop_result.distance_points

        veto = self.risk_engine.check_hard_limits(
            live_equity_inr=self.broker.get_balance().equity_inr,
            data_is_stale=False, position_already_open=False,
        )
        if veto.value != "NONE":
            return

        sizing = self.risk_engine.calculate_position_size(
            entry_price=tick.close, stop_price=stop_result.price,
            live_equity_inr=self.broker.get_balance().equity_inr, risk_reward=risk_reward,
        )
        if not sizing.approved:
            return

        side = OrderSide.BUY if result.decision == Decision.BUY else OrderSide.SELL
        order = OrderRequest(client_order_id=f"LIVE-{tick.ts}-{self.state.tick_count}",
                               symbol=self.symbol, side=side, quantity=sizing.lots)
        fill = self.broker.place_order(order)
        if fill.status.value != "FILLED":
            return

        tm_state = TradeManagerState(
            direction=direction, entry_price=fill.filled_price,
            original_stop=stop_result.price, current_stop=stop_result.price,
            original_risk_points=stop_result.distance_points,
            target_1=target_result.target_1, target_2=target_result.target_2,
            target_3=target_result.target_3,
        )
        self.state.open_trade_manager = TradeManager(self.config, tm_state)
        self.state.open_trade_entry_regime = situation.regime.value
        self.risk_engine.register_position_opened()

    def _momentum_from_indicators(self, ind_result: dict) -> str:
        macd_accel = abs(ind_result["macd_hist"]) > abs(ind_result["macd_hist_prev"])
        volume_ok = ind_result["rel_volume"] >= 1.0
        score = sum([macd_accel, volume_ok])
        return "STRONG" if score == 2 else ("WEAKENING" if score == 1 else "DEAD")

    def _build_snapshot(self, tick: LiveTick, struct_state, ind_result: dict) -> dict:
        tm = self.state.open_trade_manager
        return {
            "ts": tick.ts, "ltp": tick.close,
            "regime_trend": struct_state.trend.value,
            "last_structure_event": struct_state.last_event.value,
            "has_open_position": tm is not None,
            "open_position": None if tm is None else {
                "direction": tm.state.direction, "entry_price": tm.state.entry_price,
                "current_stop": tm.state.current_stop, "state": tm.state.trade_state.value,
                "quantity_remaining_pct": tm.state.quantity_remaining_pct,
            },
            "risk_state": {
                "trading_disabled": self.risk_engine.state.trading_disabled,
                "trades_taken_today": self.risk_engine.state.trades_taken_today,
                "consecutive_losses": self.risk_engine.state.consecutive_losses,
                "lot_multiplier": self.risk_engine.state.current_lot_multiplier,
            },
            "total_trades_this_session": len(self.state.trade_log),
        }
