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
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

from market_structure.structure_engine import (
    MarketStructureEngine, Candle as StructCandle, TrendState, StructureState
)
from indicators.incremental import IndicatorEngine, momentum_health_from_indicator_result
from situation_analysis.situation_analyzer import SituationAnalyzer, IndicatorSnapshot, MacroContext
from signal_engine.signal_engine import SignalEngine, ConfluenceInputs, Decision
from risk_engine.risk_engine import RiskEngine, DailyRiskState
from trade_manager.trade_manager import (
    TradeManager, TradeManagerState, TradeState, ExitReason, StateTransition, TrailUpdate
)
from target_engine.stop_target_engine import StopLossEngine, TargetEngine
from execution.broker_adapters.base import BrokerProvider, OrderRequest, OrderSide
from execution.brokerage_calculator import calculate_charges, calculate_charges_with_partial_booking
from gold_intelligence.fair_value import FairValueResult, MacroBiasResult, FairValueEngine


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
    open_trade_lots: int = 1
    # Real, brokerage-adjusted P&L for today — DISTINCT from
    # risk_engine.state.daily_pnl_inr, which is an R-multiple-based
    # approximation used for risk-management decisions (de-risking,
    # daily loss limit). This field is for DISPLAY accuracy: without it,
    # the "Today's Performance" panel would show a different, charges-blind
    # number than what each individual trade row displays.
    real_daily_net_pnl_inr: float = 0.0
    last_snapshot: dict = field(default_factory=dict)
    # Latest raw tick price, updated on EVERY tick rather than only when a
    # candle closes. The full pipeline still runs per completed candle —
    # this exists purely so the dashboard can show a genuinely current
    # price instead of one up to a whole bar stale.
    last_tick_price: float | None = None
    last_tick_ts: int | None = None
    # Wall-clock time the last tick actually ARRIVED. Distinct from
    # last_tick_ts (the exchange's own timestamp): if the feed dies, the
    # exchange timestamp simply stops updating and looks indistinguishable
    # from a quiet market. Comparing this against time.time() is what makes
    # a dead feed detectable rather than silently frozen.
    last_tick_received_at: float | None = None
    # Set True once persisted candles have been replayed through the
    # indicator/structure engines on startup (see
    # _replay_persisted_candles_for_warmup). Separate from tick_count,
    # which still tracks genuinely NEW real-time ticks this session —
    # this flag lets entry evaluation start immediately after a restart
    # instead of needing 30 fresh candles (2.5 hours) rebuilt from scratch.
    indicators_warmed_up_from_replay: bool = False
    # Real-data finding: a same-direction entry within 2 hours of a
    # MOMENTUM_DECAY exit performs far worse than a fresh entry (LONG:
    # +0.101R vs +0.378R; SHORT: -0.100R vs +0.647R on 2-year real MCX
    # data — confirmed via backtest, aggregate expectancy improved
    # +0.262R to +0.332R with this gate active). Tracks the most recent
    # decay exit so a same-direction re-entry can be blocked for a
    # cooldown window; the OPPOSITE direction is never blocked.
    last_momentum_decay_exit_direction: str | None = None
    last_momentum_decay_exit_ts: int | None = None
    # External reference data (COMEX gold via GC=F, USD/INR) for the
    # FairValueEngine — pushed in periodically from outside (api/main.py's
    # external_quotes_poller), since this engine has no direct internet
    # access of its own. None until the first push arrives; the fair-value
    # calculation gracefully falls back to "unreliable" until then.
    external_xauusd: float | None = None
    external_usdinr: float | None = None
    external_data_updated_at: float | None = None
    # Tracks consecutive REJECTED ticks, specifically for the price-jump
    # check. See _tick_is_sane — without this, a genuine large move (rare,
    # but real gaps do happen) would be rejected FOREVER, since the
    # reference price it's compared against never updates on rejection.
    # Worse: on_tick() returns early on rejection, meaning an OPEN
    # POSITION's stop-loss/trailing check never runs either — a permanent
    # cascade here would leave a live trade's risk management silently
    # frozen. Same class of bug as an earlier data_loader cascade found
    # during backtesting; same fix shape applies.
    consecutive_price_jump_rejections: int = 0
    # The reference price for a meaningful, stable "change" indicator.
    # BUGFIX: the dashboard used to compute change against the close of the
    # most recently completed 5-minute candle, which flips sign on ordinary
    # tick noise every few minutes regardless of the day's real trend —
    # showing red/down even while the actual session is strongly up. Real
    # trading platforms compare against the DAY'S OPEN (or previous close);
    # this is set once, on the first candle of each new trading day.
    day_open_price: float | None = None
    morning_window_trades_today: int = 0
    evening_window_trades_today: int = 0
    # Real trading platforms compute "change" against the PREVIOUS trading
    # day's closing price, not today's opening price (which can gap from
    # yesterday's close — international gold trades near 24hrs, so this
    # gap can be meaningful). This was a real correction to an earlier fix
    # that had compared against today's open instead.
    prev_day_close_price: float | None = None
    trade_log: list = field(default_factory=list)
    signal_log: list = field(default_factory=list)
    # bounded OHLCV history for chart display — not the full session (that
    # would grow unbounded), just enough recent candles to draw a chart
    candle_history: list = field(default_factory=list)
    max_candle_history: int = 500


class LiveTradingEngine:
    """
    One instance per (instrument, session). Call on_tick() for every new
    closed candle. All state persists on self between calls — this is the
    core difference from the old api/main.py endpoints, which recreated
    the "does a signal exist right now" question from scratch each request
    without ever actually opening/managing/closing a position.
    """

    def __init__(self, config, broker: BrokerProvider, symbol: str = "GOLDM",
                 htf_timeframe_lookup=None, cooldown_days_after_disable: int | None = 1,
                 persistence_path: str | None = "trade_history.jsonl",
                 candle_persistence_path: str | None = "candle_history.jsonl",
                 open_position_path: str | None = "open_position.json"):
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
        # Multi-timeframe alignment: a SEPARATE 15M trend tracker, checked
        # alongside the existing 1H one before allowing a new entry.
        # Real-data finding: requiring BOTH 15M and 1H trend to agree with
        # the proposed direction cut backtest net loss by 60% (Rs384,788
        # -> Rs153,740 over 2 years) — the single biggest improvement
        # found in the whole profitability investigation.
        self.mtf_15m_aggregator = _LiveHTFAggregator(htf_minutes=15)

        # Persistence: without this, a service restart (crash, redeploy,
        # server reboot) silently wiped the entire in-memory trade history —
        # not a financial risk in paper mode, but a data-completeness one:
        # the whole point of a multi-week live paper run is the trade
        # record, and losing it mid-run would corrupt the very analysis this
        # was built for. Appends one JSON line per closed trade; loaded back
        # on startup so history survives restarts. None disables persistence
        # entirely (used by tests, which should not touch disk).
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.candle_persistence_path = Path(candle_persistence_path) if candle_persistence_path else None
        self.open_position_path = Path(open_position_path) if open_position_path else None

        self.structure = MarketStructureEngine()
        self.indicators = IndicatorEngine()
        self.situation_analyzer = SituationAnalyzer(config)
        self.signal_engine = SignalEngine(config)
        self.risk_engine = RiskEngine(config, DailyRiskState())
        self.stop_engine = StopLossEngine(config)
        self.fair_value_engine = FairValueEngine(config)
        self.target_engine = TargetEngine(config)

        self.state = LiveEngineState()
        self._load_persisted_trades()
        self._load_persisted_candles()
        self._replay_persisted_candles_for_warmup()
        self._load_persisted_open_position()

    def _load_persisted_trades(self):
        """
        Load any trade history from a previous run. Resilient per-line: a
        real crash mid-write typically corrupts only the LAST line (the
        write in progress when the process died), not earlier ones —
        aborting on the first bad line would discard an entire session's
        valid history over one truncated write at the end.
        """
        if self.persistence_path is None or not self.persistence_path.exists():
            return
        loaded, skipped = 0, 0
        try:
            with open(self.persistence_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.state.trade_log.append(json.loads(line))
                        loaded += 1
                    except json.JSONDecodeError:
                        skipped += 1
        except OSError as e:
            print(f"Warning: could not read persisted trade history "
                  f"({type(e).__name__}: {e}) — starting with empty history.")
            return
        if skipped:
            print(f"Loaded {loaded} persisted trades, skipped {skipped} corrupted line(s).")

    def _persist_trade(self, trade: dict):
        """Append one closed trade to disk. A write failure must not crash
        the trading loop — losing one persisted record is far better than
        losing the ability to trade."""
        if self.persistence_path is None:
            return
        try:
            with open(self.persistence_path, "a") as f:
                f.write(json.dumps(trade) + "\n")
        except OSError as e:
            print(f"Warning: could not persist trade to disk: {type(e).__name__}: {e}")

    def get_daily_pnl_history(self, max_days: int = 30) -> list[dict]:
        """
        Real, brokerage-adjusted P&L grouped by calendar day, most recent
        first. Reads from the FULL persisted trade history on disk (not
        just the bounded 1000-entry in-memory trade_log), so this stays
        accurate over a multi-week paper trading run even once the
        in-memory list has been trimmed.

        Day grouping uses the same UTC-date convention as the risk engine's
        own daily-reset logic (_handle_day_boundary) — using a different
        convention here (e.g. IST calendar days) would make this summary
        disagree with when the system itself considers a "day" to have
        rolled over, which would be a confusing, silent inconsistency.
        """
        all_trades = list(self.state.trade_log)
        if self.persistence_path is not None and self.persistence_path.exists():
            all_trades = []
            try:
                with open(self.persistence_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_trades.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                all_trades = list(self.state.trade_log)

        by_day: dict = {}
        for t in all_trades:
            if "net_pnl_inr" not in t:
                continue   # older-format trades persisted before brokerage tracking existed
            day = datetime.fromtimestamp(t["ts"], tz=timezone.utc).date()
            bucket = by_day.setdefault(day, {"trades": [], "gross": 0.0, "charges": 0.0, "net": 0.0})
            bucket["trades"].append(t)
            bucket["gross"] += t["gross_pnl_inr"]
            bucket["charges"] += t["total_charges_inr"]
            bucket["net"] += t["net_pnl_inr"]

        result = []
        for day in sorted(by_day.keys(), reverse=True)[:max_days]:
            b = by_day[day]
            wins = sum(1 for t in b["trades"] if t["net_pnl_inr"] > 0)
            result.append({
                "date": day.isoformat(),
                "trade_count": len(b["trades"]),
                "wins": wins,
                "losses": len(b["trades"]) - wins,
                "gross_pnl_inr": round(b["gross"], 2),
                "total_charges_inr": round(b["charges"], 2),
                "net_pnl_inr": round(b["net"], 2),
            })
        return result

    def _load_persisted_candles(self):
        """Same rationale as trade persistence: without this, EVERY service
        restart (and there were many during active development) wiped the
        chart back to empty, making it look broken even though the
        underlying pipeline was fine — there was just never enough
        uninterrupted runtime to accumulate more than 1-2 candles."""
        if self.candle_persistence_path is None or not self.candle_persistence_path.exists():
            return
        loaded, skipped = 0, 0
        try:
            with open(self.candle_persistence_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.state.candle_history.append(json.loads(line))
                        loaded += 1
                    except json.JSONDecodeError:
                        skipped += 1
            self.state.candle_history = self.state.candle_history[-self.state.max_candle_history:]
        except OSError as e:
            print(f"Warning: could not read persisted candle history: {type(e).__name__}: {e}")
            return
        if loaded:
            print(f"Loaded {loaded} persisted candles" + (f", skipped {skipped} corrupted" if skipped else ""))

    def _persist_candle(self, candle: dict):
        if self.candle_persistence_path is None:
            return
        try:
            with open(self.candle_persistence_path, "a") as f:
                f.write(json.dumps(candle) + "\n")
        except OSError as e:
            print(f"Warning: could not persist candle to disk: {type(e).__name__}: {e}")

    def _persist_open_position(self):
        """
        Writes the CURRENT open trade's full state to disk — called after
        opening, and after every trailing-stop/partial-booking update, so
        the persisted snapshot is always current. Without this, a service
        restart while a position was open silently lost it entirely: not
        just the P&L outcome, but risk-engine's position-open tracking and
        the broker's margin/equity bookkeeping went out of sync too (a
        real incident — a genuine open trade vanished on restart with no
        record of its eventual outcome).
        """
        if self.open_position_path is None or self.state.open_trade_manager is None:
            return
        tm_state = self.state.open_trade_manager.state
        bundle = {
            "trade_manager_state": {
                "trade_state": tm_state.trade_state.value,
                "direction": tm_state.direction,
                "entry_price": tm_state.entry_price,
                "original_stop": tm_state.original_stop,
                "current_stop": tm_state.current_stop,
                "original_risk_points": tm_state.original_risk_points,
                "quantity_remaining_pct": tm_state.quantity_remaining_pct,
                "booked_at_1R": tm_state.booked_at_1R,
                "booked_at_t1": tm_state.booked_at_t1,
                "booked_at_t2": tm_state.booked_at_t2,
                "target_1": tm_state.target_1,
                "target_2": tm_state.target_2,
                "target_3": tm_state.target_3,
                "state_history": [
                    {"from_state": h.from_state.value, "to_state": h.to_state.value,
                      "trigger_reason": h.trigger_reason, "price_at_event": h.price_at_event}
                    for h in tm_state.state_history
                ],
                "trail_history": [
                    {"old_stop": h.old_stop, "new_stop": h.new_stop,
                      "method_used": h.method_used, "reason": h.reason}
                    for h in tm_state.trail_history
                ],
                "exit_reason": tm_state.exit_reason.value,
                "realized_legs": [list(leg) for leg in tm_state.realized_legs],
            },
            "entry_regime": self.state.open_trade_entry_regime,
            "lots": self.state.open_trade_lots,
            "margin_used_inr": self.broker.get_balance().margin_used_inr,
            "equity_inr": self.broker.get_balance().equity_inr,
        }
        try:
            with open(self.open_position_path, "w") as f:
                json.dump(bundle, f)
        except OSError as e:
            print(f"Warning: could not persist open position: {type(e).__name__}: {e}")

    def _clear_persisted_open_position(self):
        """Called the moment a trade closes — a stale open-position file
        left behind after the real trade closed would cause the NEXT
        restart to wrongly resurrect a position that no longer exists."""
        if self.open_position_path is None:
            return
        try:
            if self.open_position_path.exists():
                self.open_position_path.unlink()
        except OSError as e:
            print(f"Warning: could not clear persisted open position: {type(e).__name__}: {e}")

    def _load_persisted_open_position(self):
        """
        Restores an open trade's full state on startup, including
        re-syncing the broker's position/margin/equity bookkeeping —
        WITHOUT re-charging commission or slippage, since the original
        entry already did that before the restart (see
        PaperBrokerProvider.restore_position).
        """
        if self.open_position_path is None or not self.open_position_path.exists():
            return
        try:
            with open(self.open_position_path) as f:
                bundle = json.load(f)

            tms_data = bundle["trade_manager_state"]
            state_history = [
                StateTransition(from_state=TradeState(h["from_state"]),
                                  to_state=TradeState(h["to_state"]),
                                  trigger_reason=h["trigger_reason"],
                                  price_at_event=h["price_at_event"])
                for h in tms_data["state_history"]
            ]
            trail_history = [
                TrailUpdate(old_stop=h["old_stop"], new_stop=h["new_stop"],
                              method_used=h["method_used"], reason=h["reason"])
                for h in tms_data["trail_history"]
            ]
            tm_state = TradeManagerState(
                trade_state=TradeState(tms_data["trade_state"]),
                direction=tms_data["direction"], entry_price=tms_data["entry_price"],
                original_stop=tms_data["original_stop"], current_stop=tms_data["current_stop"],
                original_risk_points=tms_data["original_risk_points"],
                quantity_remaining_pct=tms_data["quantity_remaining_pct"],
                booked_at_1R=tms_data["booked_at_1R"], booked_at_t1=tms_data["booked_at_t1"],
                booked_at_t2=tms_data["booked_at_t2"],
                target_1=tms_data["target_1"], target_2=tms_data["target_2"],
                target_3=tms_data["target_3"], state_history=state_history,
                trail_history=trail_history, exit_reason=ExitReason(tms_data["exit_reason"]),
                realized_legs=[tuple(leg) for leg in tms_data["realized_legs"]],
            )

            self.state.open_trade_manager = TradeManager(self.config, tm_state)
            self.state.open_trade_entry_regime = bundle["entry_regime"]
            self.state.open_trade_lots = bundle["lots"]

            signed_qty = bundle["lots"] * (1 if tm_state.direction == "LONG" else -1)
            if hasattr(self.broker, "restore_position"):
                self.broker.restore_position(
                    symbol=self.symbol, quantity=signed_qty, avg_price=tm_state.entry_price,
                    margin_used_inr=bundle["margin_used_inr"], equity_inr=bundle["equity_inr"],
                )
            self.risk_engine.register_position_opened()

            print(f"RESTORED open position from disk: {tm_state.direction} @ "
                  f"{tm_state.entry_price}, stop={tm_state.current_stop}, "
                  f"state={tm_state.trade_state.value} — trade management resumes normally.")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: could not restore persisted open position "
                  f"({type(e).__name__}: {e}) — starting with no open position. "
                  f"If a trade was genuinely open, its outcome will not be tracked.")

    def _replay_persisted_candles_for_warmup(self):
        """
        BUGFIX (real incident): candle_history was persisted across restarts
        (for the chart), but the indicator/structure engines were NOT — they
        rebuilt from scratch every restart, meaning tick_count needed 30
        genuinely NEW real-time candles (2.5 hours) before any signal could
        be evaluated, even when perfectly good recent history already sat
        on disk unused. A real restart at 14:27 UTC would have delayed
        warmup past 22:27 IST, costing nearly the entire remaining session.

        Fix: replay the already-persisted candles through the structure and
        indicator engines on startup (silently, building up their internal
        state — EMAs, ATR, swing points — exactly as if they'd been running
        continuously) WITHOUT evaluating any trade decisions on them. This
        does NOT affect tick_count (which still gates live decision-making
        on genuinely fresh data) — it only fast-forwards the indicators so
        that once real ticks resume, they're immediately usable rather than
        needing another multi-hour rebuild.
        """
        if not self.state.candle_history:
            return
        replayed = 0
        for c in self.state.candle_history:
            try:
                struct_candle = StructCandle(ts=c["ts"], open=c["open"], high=c["high"],
                                                low=c["low"], close=c["close"], volume=c["volume"])
                self.structure.update(struct_candle, current_atr=self.indicators.atr.value or 1.0,
                                         higher_tf_trend=TrendState.RANGE)
                self.indicators.update(c["high"], c["low"], c["close"], c["volume"])
                if self.htf_aggregator is not None:
                    replay_tick = LiveTick(ts=c["ts"], open=c["open"], high=c["high"],
                                             low=c["low"], close=c["close"], volume=c["volume"])
                    self.htf_aggregator.update(replay_tick)
                replayed += 1
            except (KeyError, TypeError):
                continue   # a malformed persisted entry must not abort the whole replay
        if replayed:
            print(f"Replayed {replayed} persisted candles to warm up indicators — "
                  f"no multi-hour wait needed after this restart.")
            if replayed >= 30:
                self.state.indicators_warmed_up_from_replay = True

    def _compute_fair_value(self, mcx_price: float) -> FairValueResult:
        """
        Real FairValueEngine calculation using live external COMEX
        gold + USD/INR data, when available and fresh. Falls back to
        an explicitly "unreliable" placeholder — never a fabricated
        number — if external data hasn't arrived yet or has gone stale
        (>10 minutes old, generous given the external poller refreshes
        every 60s and this is informational, not safety-critical).
        """
        EXTERNAL_DATA_MAX_AGE_SEC = 600
        xauusd = self.state.external_xauusd
        usdinr = self.state.external_usdinr
        updated_at = self.state.external_data_updated_at

        if xauusd is None or usdinr is None or updated_at is None:
            return FairValueResult(mcx_price=mcx_price, theoretical_price=mcx_price,
                                      deviation=0, deviation_pct=0, deviation_zscore=0.0,
                                      is_reliable=False,
                                      unreliable_reason="No external COMEX/USD-INR data received yet")
        if time.time() - updated_at > EXTERNAL_DATA_MAX_AGE_SEC:
            return FairValueResult(mcx_price=mcx_price, theoretical_price=mcx_price,
                                      deviation=0, deviation_pct=0, deviation_zscore=0.0,
                                      is_reliable=False,
                                      unreliable_reason="External reference data is stale")

        # v1 simplification: a fixed mid-cycle estimate for days-to-expiry
        # rather than tracking the exact selected contract's real expiry
        # date here (that lives in the feed handler, not this engine).
        # carry_cost's contribution to theoretical price is small relative
        # to the metal-price/FX components, so this approximation doesn't
        # materially distort the deviation reading.
        APPROX_DAYS_TO_EXPIRY = 30

        return self.fair_value_engine.calculate(
            mcx_price=mcx_price, xauusd=xauusd, usdinr=usdinr,
            days_to_expiry=APPROX_DAYS_TO_EXPIRY, both_sessions_live=True, data_stale=False,
        )

    def set_external_reference_data(self, xauusd: float, usdinr: float):
        """
        Called periodically from outside (api/main.py, using its own
        external_quotes_poller) to feed live COMEX gold and USD/INR into
        this engine's fair-value calculation. xauusd here is COMEX gold
        futures (GC=F) used as a spot-price proxy — there is a small,
        normally minor basis between futures and true spot XAU/USD; this
        is a documented v1 simplification, not an attempt at
        futures-curve-accurate pricing.
        """
        self.state.external_xauusd = xauusd
        self.state.external_usdinr = usdinr
        self.state.external_data_updated_at = time.time()

    def seconds_since_last_tick(self) -> float | None:
        """Wall-clock seconds since a tick last arrived, or None if none ever
        has. Used to distinguish 'market is quiet' from 'the feed is dead'."""
        if self.state.last_tick_received_at is None:
            return None
        return time.time() - self.state.last_tick_received_at

    def update_live_price(self, ltp: float, ts: int) -> None:
        """
        Record the latest traded price without running any strategy logic.
        Called on every raw tick by the feed handler; the full pipeline
        (structure, indicators, signals, trade management) still runs only
        on completed candles via on_tick(). Deliberately does nothing else —
        no analysis should ever run on an unclosed bar.
        """
        self.state.last_tick_price = ltp
        self.state.last_tick_ts = ts
        self.state.last_tick_received_at = time.time()

    def _tick_is_sane(self, tick: LiveTick) -> bool:
        """
        Input validation on every incoming candle before it ever reaches
        structure/indicator/signal logic. Without this, a single malformed
        feed message (network glitch, broker API bug, decimal/unit error)
        with a zero, negative, or wildly wrong price would silently
        corrupt ATR/EMA calculations for every candle after it, and could
        even let a signal fire and a position open at a nonsensical price.
        This is exactly the class of gap that must be closed before real
        capital is ever at risk — a bad feed tick must be REJECTED and
        logged, never quietly trusted.
        """
        if tick.open <= 0 or tick.high <= 0 or tick.low <= 0 or tick.close <= 0:
            print(f"REJECTED tick at ts={tick.ts}: non-positive price "
                  f"(O={tick.open} H={tick.high} L={tick.low} C={tick.close})")
            return False
        if tick.high < tick.low:
            print(f"REJECTED tick at ts={tick.ts}: high ({tick.high}) < low ({tick.low})")
            return False
        if not (tick.low <= tick.open <= tick.high and tick.low <= tick.close <= tick.high):
            print(f"REJECTED tick at ts={tick.ts}: open/close outside high-low range "
                  f"(O={tick.open} H={tick.high} L={tick.low} C={tick.close})")
            return False
        if tick.volume < 0:
            print(f"REJECTED tick at ts={tick.ts}: negative volume ({tick.volume})")
            self.state.consecutive_price_jump_rejections = 0
            return False
        # sanity-check against the last known genuine price — gold does not
        # move 15%+ in a single 5-minute bar under NORMAL conditions; a jump
        # that large is more likely a feed/decimal error than a real move.
        #
        # BUGFIX (found during live-trade review, real incident risk): the
        # ORIGINAL version of this check rejected such a tick forever if the
        # move was genuine — the reference price (last_snapshot) only
        # updates on ACCEPTED ticks, so a real large gap would be rejected,
        # the reference would stay frozen, and every SUBSEQUENT tick would
        # ALSO fail the same check against that same stale reference —
        # a permanent rejection cascade. Worse: on_tick() returns early on
        # rejection, so an OPEN POSITION's stop-loss/trailing check would
        # never run again either, silently freezing risk management on a
        # live trade. Same failure shape as an earlier data_loader cascade
        # bug found during backtesting.
        #
        # Fix: after 3 CONSECUTIVE rejections for this same reason, treat
        # it as a genuine move rather than a glitch — a real feed glitch is
        # transient (one bad tick), but a genuine gap persists across
        # multiple readings at the new level. This distinguishes the two
        # without ever getting permanently stuck.
        last_price = self.state.last_snapshot.get("ltp") if self.state.last_snapshot else None
        if last_price and last_price > 0:
            pct_change = abs(tick.close - last_price) / last_price
            if pct_change > 0.15:
                self.state.consecutive_price_jump_rejections += 1
                if self.state.consecutive_price_jump_rejections >= 3:
                    print(f"ACCEPTING tick at ts={tick.ts} despite {pct_change*100:.1f}% jump — "
                          f"3 consecutive readings at this level, treating as a genuine "
                          f"move, not a glitch.")
                    self.state.consecutive_price_jump_rejections = 0
                    return True
                print(f"REJECTED tick at ts={tick.ts}: {pct_change*100:.1f}% jump from last "
                      f"known price {last_price} to {tick.close} — treating as feed corruption, "
                      f"not a real move ({self.state.consecutive_price_jump_rejections}/3 "
                      f"consecutive; will accept if this persists)")
                return False
        self.state.consecutive_price_jump_rejections = 0
        return True

    def on_tick(self, tick: LiveTick) -> dict:
        """
        Processes one new CLOSED candle. Returns a snapshot dict suitable
        for API/dashboard consumption. This is the single entry point —
        everything else is internal.
        """
        if not self._tick_is_sane(tick):
            # return the last good snapshot unchanged rather than a crash or
            # a corrupted one — the dashboard keeps showing the last known
            # genuine state until a valid tick arrives
            return self.state.last_snapshot or {
                "ts": tick.ts, "ltp": 0.0, "day_open_price": None,
                "regime_trend": "RANGE", "last_structure_event": "NONE",
                "has_open_position": False, "open_position": None,
                "risk_state": {"trading_disabled": False, "trades_taken_today": 0,
                                "consecutive_losses": 0, "lot_multiplier": 1.0},
                "total_trades_this_session": 0,
            }

        self.state.tick_count += 1

        self._handle_day_boundary(tick)

        # record candle history for chart display, bounded to avoid unbounded growth
        new_candle = {
            "ts": tick.ts, "open": tick.open, "high": tick.high,
            "low": tick.low, "close": tick.close, "volume": tick.volume,
        }
        self.state.candle_history.append(new_candle)
        self._persist_candle(new_candle)
        if len(self.state.candle_history) > self.state.max_candle_history:
            self.state.candle_history = self.state.candle_history[-self.state.max_candle_history:]

        struct_candle = StructCandle(ts=tick.ts, open=tick.open, high=tick.high,
                                       low=tick.low, close=tick.close, volume=tick.volume)
        if self.htf_lookup_fn is not None:
            htf_trend = self.htf_lookup_fn(tick.ts)
        else:
            htf_trend = self.htf_aggregator.update(tick)
        self._current_htf_trend = htf_trend
        self._current_15m_trend = self.mtf_15m_aggregator.update(tick)
        struct_state = self.structure.update(struct_candle,
                                                current_atr=self.indicators.atr.value or 1.0,
                                                higher_tf_trend=htf_trend)
        ind_result = self.indicators.update(tick.high, tick.low, tick.close, tick.volume)
        self.broker.set_quote(self.symbol, ltp=tick.close, volume=tick.volume) \
            if hasattr(self.broker, "set_quote") else None

        if self.state.open_trade_manager is not None:
            self._manage_open_trade(tick, ind_result, struct_state)
        elif self.state.tick_count > 30 or self.state.indicators_warmed_up_from_replay:
            # let indicators warm up — either via 30 genuinely fresh candles
            # this session, OR via a startup replay of persisted history
            # (see _replay_persisted_candles_for_warmup)
            self._evaluate_new_entry(tick, ind_result, struct_state)

        snapshot = self._build_snapshot(tick, struct_state, ind_result)
        self.state.last_snapshot = snapshot
        return snapshot

    def _derive_prev_day_close_from_history(self, current_ref_day):
        """
        Called when the engine starts (fresh or after a restart) to find
        the previous trading day's closing price from already-persisted
        candle_history, so "change" is correct even before any live
        cross-day transition has been observed this session.
        """
        for c in reversed(self.state.candle_history):
            c_day = datetime.fromtimestamp(c["ts"], tz=timezone.utc).date()
            if c_day < current_ref_day:
                self.state.prev_day_close_price = c["close"]
                return

    def _handle_day_boundary(self, tick: LiveTick):
        day = datetime.fromtimestamp(tick.ts, tz=timezone.utc).date()
        if self.state.current_day is None:
            self.state.current_day = day
            self.state.day_open_price = tick.open
            self._derive_prev_day_close_from_history(day)
            return
        if day == self.state.current_day:
            return

        # A genuine day transition — capture the LAST known price (from
        # the day that just ended) as the reference for "change" going
        # forward, before anything else updates.
        if self.state.last_snapshot:
            self.state.prev_day_close_price = self.state.last_snapshot.get("ltp")

        self.risk_engine.state.trades_taken_today = 0
        self.risk_engine.state.daily_pnl_inr = 0.0
        self.state.morning_window_trades_today = 0
        self.state.evening_window_trades_today = 0
        self.state.real_daily_net_pnl_inr = 0.0
        self.state.day_open_price = tick.open

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
        momentum_health = momentum_health_from_indicator_result(ind_result)
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

        if tm.state.trade_state.value != "EXITED":
            # trade is still open — persist the UPDATED state (trailed stop,
            # any partial booking) so a restart resumes from here, not from
            # the original entry parameters
            self._persist_open_position()

        if tm.state.trade_state.value == "EXITED":
            # BUGFIX (real accounting bug found via live-trade equity check):
            # exits NEVER called broker.place_order() — only entries did.
            # This meant the broker's own position/equity bookkeeping never
            # actually closed, so: (a) the NEXT entry's place_order() would
            # find a still-"open" position in the broker's books and could
            # mishandle quantity/avg_price tracking, and (b) equity only
            # ever reflected the entry commission, never a proper close.
            # Fix: place a REAL opposite-side closing order, use the
            # broker's ACTUAL fill price (not an analytically-estimated
            # one) as the authoritative exit price, and reconcile equity
            # to exactly match the realistic charges shown in the
            # dashboard (CTT/GST/exchange/stamp — not the broker's own
            # simplified flat commission model), so "starting equity + sum
            # of displayed net P&L" always holds exactly.
            analytical_exit_price = (tm.state.state_history[-1].price_at_event
                          if tm.state.state_history else tick.close)
            close_side = OrderSide.SELL if tm.state.direction == "LONG" else OrderSide.BUY
            close_order = OrderRequest(
                client_order_id=f"CLOSE-{tick.ts}-{self.state.tick_count}",
                symbol=self.symbol, side=close_side, quantity=self.state.open_trade_lots,
            )
            equity_before_close = self.broker.get_balance().equity_inr
            # BUGFIX (found via a real -1.22R stop-loss trade, should be
            # ~-1.0R): the quote was last set to the CANDLE'S CLOSE price
            # (from the routine set_quote() call earlier this tick), not
            # the actual stop/target level where the exit was triggered.
            # On a candle that moved hard through the stop, the market
            # order filled at the candle's close — potentially far beyond
            # the intended stop level — instead of near the stop itself,
            # like a real stop order would. Re-anchor the quote to the
            # actual exit level right before placing the closing order, so
            # only normal spread/slippage applies, not the candle's full
            # excess travel beyond the stop.
            if hasattr(self.broker, "set_quote"):
                self.broker.set_quote(self.symbol, ltp=analytical_exit_price, volume=tick.volume)
            close_fill = self.broker.place_order(close_order)
            exit_price = (close_fill.filled_price if close_fill.status.value == "FILLED"
                           else analytical_exit_price)

            r = tm.blended_r_multiple(exit_price)
            charges = calculate_charges_with_partial_booking(
                direction=tm.state.direction, entry_price=tm.state.entry_price,
                original_risk_points=tm.state.original_risk_points,
                realized_legs=tm.state.realized_legs,
                final_exit_price=exit_price,
                quantity_remaining_pct=tm.state.quantity_remaining_pct,
                total_lots=self.state.open_trade_lots,
                point_value_inr=self.config.instrument.point_value_inr,
            )
            # reconcile: the broker's own place_order() already applied its
            # own (simpler) commission + realized P&L to equity. Adjust by
            # the delta so the FINAL equity matches our realistic charges
            # model exactly, rather than silently disagreeing with the
            # dashboard's displayed numbers.
            equity_after_broker_close = self.broker.get_balance().equity_inr
            broker_applied_delta = equity_after_broker_close - equity_before_close
            target_delta = charges.net_pnl_inr
            reconciliation = target_delta - broker_applied_delta
            if hasattr(self.broker, "adjust_equity"):
                self.broker.adjust_equity(reconciliation)

            self.risk_engine.record_trade_result(r * self.config.risk.max_risk_per_trade_inr)
            self.state.real_daily_net_pnl_inr += charges.net_pnl_inr
            if self.risk_engine.state.trading_disabled and self.state.disabled_since_day is None:
                self.state.disabled_since_day = self.state.current_day

            closed_trade = {
                "entry_price": tm.state.entry_price, "exit_price": exit_price,
                "direction": tm.state.direction, "r_multiple": r,
                "exit_reason": tm.state.exit_reason.value, "ts": tick.ts,
                "entry_regime": self.state.open_trade_entry_regime,
                "lots": self.state.open_trade_lots,
                "gross_pnl_inr": charges.gross_pnl_inr,
                "total_charges_inr": charges.total_charges_inr,
                "net_pnl_inr": charges.net_pnl_inr,
            }
            self.state.trade_log.append(closed_trade)
            self._persist_trade(closed_trade)
            # bounded like signal_log — a long-running live session must not
            # grow memory without limit. Full history lives in persistence_path.
            self.state.trade_log = self.state.trade_log[-1000:]
            self.risk_engine.register_position_closed()
            if tm.state.exit_reason.value == "MOMENTUM_DECAY":
                self.state.last_momentum_decay_exit_direction = tm.state.direction
                self.state.last_momentum_decay_exit_ts = tick.ts
            self.state.open_trade_manager = None
            self.state.open_trade_entry_regime = None
            self._clear_persisted_open_position()

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

        fair_value = self._compute_fair_value(tick.close)

        conf_inputs = ConfluenceInputs(
            ltf_structure=struct_state, situation=situation,
            fair_value=fair_value,
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

        # Multi-timeframe alignment (Vivek's idea, verified as the day's
        # biggest improvement): require the 15M AND 1H trend to BOTH agree
        # with the proposed direction, not just the 5M entry signal alone.
        wanted_trend = TrendState.TRENDING_UP if direction == "LONG" else TrendState.TRENDING_DOWN
        if self._current_15m_trend != wanted_trend or self._current_htf_trend != wanted_trend:
            return   # 15M and 1H must BOTH confirm the same direction

        # Volatility expansion (Vivek's idea, verified on real data): only
        # trade when CURRENT ATR is genuinely EXPANDING vs its recent
        # average — avoids range-bound/quiet candles, targets genuine
        # movement. Combined with MTF alignment, this cut backtest net
        # loss by 85% vs original baseline (2-year real MCX data,
        # multiplier=1.1 chosen for a statistically meaningful sample —
        # tighter multipliers showed even better per-trade results but
        # too few trades to trust).
        VOLATILITY_EXPANSION_MULT = 1.1
        atr_avg = ind_result["atr_avg_20"] or ind_result["atr"] or 1.0
        if ind_result["atr"] < atr_avg * VOLATILITY_EXPANSION_MULT:
            return   # current volatility isn't genuinely expanding — skip

        # Vivek's spec: morning window (9-11 AM IST = 3:30-5:30 UTC, max
        # 2 trades) + evening window (4-9:30 PM IST = 10:30-16:00 UTC,
        # max 4 trades). Deployed directly per explicit request, without
        # the full real-data backtest validation given to earlier
        # filters today — monitor live results carefully.
        if self.config.risk.require_london_ny_session:
            entry_dt = datetime.fromtimestamp(tick.ts, tz=timezone.utc)
            entry_minutes_utc = entry_dt.hour * 60 + entry_dt.minute
            in_morning = (3 * 60 + 30 <= entry_minutes_utc < 5 * 60 + 30)
            in_evening = (10 * 60 + 30 <= entry_minutes_utc < 16 * 60)
            if not (in_morning or in_evening):
                return
            if in_morning and self.state.morning_window_trades_today >= 2:
                return
            if in_evening and self.state.evening_window_trades_today >= 4:
                return

        SAME_DIRECTION_REENTRY_COOLDOWN_SEC = 7200  # 2 hours — see LiveEngineState field docstring
        if (self.state.last_momentum_decay_exit_direction == direction
                and self.state.last_momentum_decay_exit_ts is not None
                and tick.ts - self.state.last_momentum_decay_exit_ts <= SAME_DIRECTION_REENTRY_COOLDOWN_SEC):
            return   # blocked: chasing the same direction right after it decayed

        nsl = struct_state.swing_lows[-1].price if struct_state.swing_lows else None
        nsh = struct_state.swing_highs[-1].price if struct_state.swing_highs else None

        # Real-data finding: LONG entries far from a recent swing low
        # (support) and SHORT entries far from a recent swing high
        # (resistance) were, as a group, net LOSERS on 2-year real MCX
        # data (-0.059R / -0.079R), while entries near support/resistance
        # were strongly profitable (+0.913R / +0.363R standalone). Adding
        # this filter on top of the reentry cooldown improved the
        # AGGREGATE backtest from +0.332R to +0.596R (PF 1.78 -> 2.51),
        # confirmed robust across a range of proximity thresholds (smooth,
        # monotonic improvement as the filter tightens — the signature of
        # a real effect, not overfitting to one lucky number).
        SUPPORT_RESISTANCE_PROXIMITY_ATR_MULT = 1.5
        atr_now = ind_result["atr"] or 10.0
        proximity = atr_now * SUPPORT_RESISTANCE_PROXIMITY_ATR_MULT
        if direction == "LONG":
            if nsl is None or abs(tick.close - nsl) > proximity:
                return
        else:
            if nsh is None or abs(tick.close - nsh) > proximity:
                return

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

        # BUGFIX: data_is_stale was hardcoded False — dead code, since the
        # protection never actually engaged. A naive "seconds since last
        # tick received" check turned out to be structurally useless here
        # too: by the time this runs, a tick just arrived by construction,
        # so it would always read as fresh. The real signal that matters is
        # a GAP between consecutive completed candles — if the feed dropped
        # for 15+ minutes and reconnected, the candle immediately after
        # that gap was built on an interrupted data stream (indicators may
        # be stale/wrong), and must not be trusted for a fresh entry even
        # though the CURRENT tick itself is perfectly fresh.
        expected_interval_sec = 300  # 5-minute candles
        candle_gap_detected = False
        if len(self.state.candle_history) >= 2:
            prev_candle_ts = self.state.candle_history[-2]["ts"]
            actual_gap = tick.ts - prev_candle_ts
            if actual_gap > expected_interval_sec * 2.5:
                candle_gap_detected = True
                print(f"Data gap detected: {actual_gap}s between candles "
                      f"(expected ~{expected_interval_sec}s) — skipping entry "
                      f"evaluation on this candle, feed likely dropped and recovered.")

        veto = self.risk_engine.check_hard_limits(
            live_equity_inr=self.broker.get_balance().equity_inr,
            data_is_stale=candle_gap_detected, position_already_open=False,
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
        self.state.open_trade_lots = sizing.lots
        self.risk_engine.register_position_opened()
        if self.config.risk.require_london_ny_session:
            entry_dt = datetime.fromtimestamp(tick.ts, tz=timezone.utc)
            entry_minutes_utc = entry_dt.hour * 60 + entry_dt.minute
            if 3 * 60 + 30 <= entry_minutes_utc < 5 * 60 + 30:
                self.state.morning_window_trades_today += 1
            elif 10 * 60 + 30 <= entry_minutes_utc < 16 * 60:
                self.state.evening_window_trades_today += 1
        self._persist_open_position()

    def _build_snapshot(self, tick: LiveTick, struct_state, ind_result: dict) -> dict:
        tm = self.state.open_trade_manager
        return {
            "ts": tick.ts, "ltp": tick.close,
            "day_open_price": self.state.day_open_price,
            "prev_day_close_price": self.state.prev_day_close_price,
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
