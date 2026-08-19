import sys
sys.path.insert(0, '.')
from config.settings import Settings
from market_data.data_loader import DataQualityGate
from indicators.incremental import IndicatorEngine
from market_structure.structure_engine import MarketStructureEngine, Candle as StructCandle, TrendState
from situation_analysis.situation_analyzer import SituationAnalyzer, IndicatorSnapshot, MacroContext
from signal_engine.signal_engine import SignalEngine, ConfluenceInputs, Decision
from risk_engine.risk_engine import RiskEngine, DailyRiskState
from target_engine.stop_target_engine import StopLossEngine, TargetEngine
from trade_manager.trade_manager import TradeManager, TradeManagerState
from execution.brokerage_calculator import calculate_charges_with_partial_booking
from gold_intelligence.fair_value import FairValueResult, MacroBiasResult
from backtesting.metrics import compute_metrics
from datetime import datetime, timezone

settings = Settings()
gate = DataQualityGate(settings, expected_interval_minutes=5)
candles = gate.load_csv('/mnt/user-data/uploads/mcx_goldm_5min.csv').candles

structure = MarketStructureEngine()
indicators = IndicatorEngine()
situation_analyzer = SituationAnalyzer(settings)
signal_engine = SignalEngine(settings)
risk_engine = RiskEngine(settings, DailyRiskState())
stop_engine = StopLossEngine(settings)
target_engine = TargetEngine(settings)

open_tm = None
open_meta = {}
current_day = None
disabled_since_day = None
last_decay_exit = {}
COOLDOWN_SEC = 7200

# CONVICTION-BASED SIZING: normal trades get standard Rs2000/200pt cap.
# HIGH-CONVICTION trades (score >= HIGH_CONVICTION_THRESHOLD, i.e. strong
# momentum + volume + structure alignment already baked into the
# confluence score) get a WIDER stop allowance AND proportionally more
# risk budget — testing the user's idea: bigger size only for genuinely
# strong setups, not uniformly for everything.
HIGH_CONVICTION_THRESHOLD = 80  # score out of 100
NORMAL_RISK = 2000
HIGH_CONVICTION_RISK = 4000  # 2x risk, ONLY for strong setups
NORMAL_MAX_STOP = 200
HIGH_CONVICTION_MAX_STOP = 400

trades = []

for i, candle in enumerate(candles):
    day = datetime.fromtimestamp(candle.ts, tz=timezone.utc).date()
    if current_day is None:
        current_day = day
    elif day != current_day:
        risk_engine.state.trades_taken_today = 0
        risk_engine.state.daily_pnl_inr = 0.0
        if risk_engine.state.trading_disabled and disabled_since_day is not None:
            if (day - disabled_since_day).days >= 1:
                risk_engine.manual_reset()
                disabled_since_day = None
        current_day = day

    sc = StructCandle(ts=candle.ts, open=candle.open, high=candle.high, low=candle.low,
                        close=candle.close, volume=candle.volume)
    struct_state = structure.update(sc, current_atr=indicators.atr.value or 1.0, higher_tf_trend=TrendState.RANGE)
    ind = indicators.update(candle.high, candle.low, candle.close, candle.volume)

    if open_tm is not None:
        momentum = "STRONG" if (abs(ind["macd_hist"]) > abs(ind["macd_hist_prev"]) and ind["rel_volume"] >= 1.0) else \
                   ("WEAKENING" if (abs(ind["macd_hist"]) > abs(ind["macd_hist_prev"])) != (ind["rel_volume"] >= 1.0) else "DEAD")
        broke = struct_state.last_event.value.startswith("CHOCH") and (
            (open_tm.state.direction == "LONG" and "BEARISH" in struct_state.last_event.value) or
            (open_tm.state.direction == "SHORT" and "BULLISH" in struct_state.last_event.value))
        open_tm.check_partial_booking(candle.close)
        open_tm.update_trailing_stop(candle.close, ind["ema9"], ind["ema21"], ind["ema50"],
                                       ind["atr"], momentum, broke)
        if open_tm.state.trade_state.value != "EXITED":
            open_tm.check_stop_hit_intrabar(high=candle.high, low=candle.low, close=candle.close)
        if open_tm.state.trade_state.value == "EXITED":
            exit_price = open_tm.state.state_history[-1].price_at_event if open_tm.state.state_history else candle.close
            r = open_tm.blended_r_multiple(exit_price)
            charges = calculate_charges_with_partial_booking(
                direction=open_tm.state.direction, entry_price=open_tm.state.entry_price,
                original_risk_points=open_tm.state.original_risk_points,
                realized_legs=open_tm.state.realized_legs, final_exit_price=exit_price,
                quantity_remaining_pct=open_tm.state.quantity_remaining_pct,
                total_lots=open_meta['lots'], point_value_inr=settings.instrument.point_value_inr,
            )
            risk_engine.record_trade_result(charges.net_pnl_inr)
            if risk_engine.state.trading_disabled and disabled_since_day is None:
                disabled_since_day = current_day
            if open_tm.state.exit_reason.value == "MOMENTUM_DECAY":
                last_decay_exit[open_tm.state.direction] = candle.ts
            trades.append({"r_multiple": r, "net_pnl_inr": charges.net_pnl_inr,
                             "gross_pnl_inr": charges.gross_pnl_inr, "charges_inr": charges.total_charges_inr,
                             "high_conviction": open_meta['high_conviction']})
            risk_engine.register_position_closed()
            open_tm = None
        continue

    if i < 30:
        continue

    ind_snap = IndicatorSnapshot(ema9=ind["ema9"], ema21=ind["ema21"], ema50=ind["ema50"],
                                    ema200=ind["ema200"], rsi=ind["rsi"], macd_hist=ind["macd_hist"],
                                    macd_hist_prev=ind["macd_hist_prev"], atr=ind["atr"],
                                    atr_avg_20=ind["atr_avg_20"], rel_volume=ind["rel_volume"])
    situation = situation_analyzer.analyze(type(struct_state)(), struct_state, ind_snap, MacroContext())

    nsl = struct_state.swing_lows[-1].price if struct_state.swing_lows else None
    nsh = struct_state.swing_highs[-1].price if struct_state.swing_highs else None
    ema_bullish = ind["ema9"] > ind["ema21"] > ind["ema50"]
    ema_bearish = ind["ema9"] < ind["ema21"] < ind["ema50"]
    conf_inputs = ConfluenceInputs(
        ltf_structure=struct_state, situation=situation,
        fair_value=FairValueResult(mcx_price=candle.close, theoretical_price=candle.close,
                                     deviation=0, deviation_pct=0, deviation_zscore=0.0, is_reliable=False),
        macro=MacroBiasResult(macro_bias=0.0, components={"session_quality_ok": True}),
        ema_aligned_bullish=ema_bullish, ema_aligned_bearish=ema_bearish,
        macd_bullish=ind["macd_hist"] > 0, macd_bearish=ind["macd_hist"] < 0, rsi=ind["rsi"],
        price_above_vwap=candle.close > (ind["vwap"] or candle.close),
        volume_supportive=ind["rel_volume"] >= 1.0,
        bb_squeeze=(ind["bb_upper"] - ind["bb_lower"]) < ind["atr"] * 2, session_quality_ok=True,
    )
    result = signal_engine.evaluate(conf_inputs)
    if result.decision == Decision.NO_TRADE:
        continue
    direction = "LONG" if result.decision == Decision.BUY else "SHORT"

    if direction in last_decay_exit and candle.ts - last_decay_exit[direction] <= COOLDOWN_SEC:
        continue

    atr_now = ind["atr"] or 10.0
    proximity = atr_now * 1.5
    if direction == "LONG":
        if nsl is None or abs(candle.close - nsl) > proximity:
            continue
    else:
        if nsh is None or abs(candle.close - nsh) > proximity:
            continue

    # CONVICTION CHECK: is this a high-quality setup per the user's
    # criteria (momentum + volume + pattern)? Use the confluence score
    # itself (already incorporates momentum, volume, EMA alignment,
    # structure) as the proxy.
    is_high_conviction = result.confidence >= HIGH_CONVICTION_THRESHOLD
    max_stop = HIGH_CONVICTION_MAX_STOP if is_high_conviction else NORMAL_MAX_STOP
    risk_budget = HIGH_CONVICTION_RISK if is_high_conviction else NORMAL_RISK

    stop_result = stop_engine.evaluate(direction=direction, entry_price=candle.close, atr=ind["atr"],
                                          nearest_swing_low=nsl, nearest_swing_high=nsh)
    if not stop_result.approved:
        continue
    if stop_result.distance_points > max_stop:
        continue  # even high-conviction setups have SOME limit

    target_result = target_engine.calculate(direction=direction, entry_price=candle.close,
                                               stop_price=stop_result.price, nearest_resistance=nsh,
                                               nearest_support=nsl, atr=ind["atr"])
    if target_result is None:
        continue
    risk_reward = abs(target_result.target_2 - candle.close) / stop_result.distance_points

    veto = risk_engine.check_hard_limits(live_equity_inr=2000000, data_is_stale=False, position_already_open=False)
    if veto.value != "NONE":
        continue

    risk_per_lot = stop_result.distance_points * settings.instrument.point_value_inr
    raw_lots = risk_budget / risk_per_lot if risk_per_lot > 0 else 0
    lots = max(1, int(raw_lots)) if raw_lots >= 1 else 0
    if lots == 0:
        continue
    lots = min(lots, 10)

    open_meta = {"lots": lots, "high_conviction": is_high_conviction}
    tm_state = TradeManagerState(direction=direction, entry_price=candle.close,
                                    original_stop=stop_result.price, current_stop=stop_result.price,
                                    original_risk_points=stop_result.distance_points,
                                    target_1=target_result.target_1, target_2=target_result.target_2,
                                    target_3=target_result.target_3)
    open_tm = TradeManager(settings, tm_state)
    risk_engine.register_position_opened()

# ==== REPORT ====
print(f"Total trades: {len(trades)}")
high_conv_trades = [t for t in trades if t['high_conviction']]
normal_trades = [t for t in trades if not t['high_conviction']]
print(f"High-conviction: {len(high_conv_trades)}, Normal: {len(normal_trades)}")
print()

total_net = sum(t['net_pnl_inr'] for t in trades)
total_gross = sum(t['gross_pnl_inr'] for t in trades)
total_charges = sum(t['charges_inr'] for t in trades)
print(f"OVERALL: avg_gross=Rs.{total_gross/len(trades):.0f}, avg_charges=Rs.{total_charges/len(trades):.0f}, NET=Rs.{total_net:,.0f}")

if high_conv_trades:
    hc_net = sum(t['net_pnl_inr'] for t in high_conv_trades)
    hc_gross = sum(t['gross_pnl_inr'] for t in high_conv_trades)
    hc_charges = sum(t['charges_inr'] for t in high_conv_trades)
    print(f"HIGH-CONVICTION only: n={len(high_conv_trades)}, avg_gross=Rs.{hc_gross/len(high_conv_trades):.0f}, "
          f"avg_charges=Rs.{hc_charges/len(high_conv_trades):.0f}, NET=Rs.{hc_net:,.0f}")
if normal_trades:
    n_net = sum(t['net_pnl_inr'] for t in normal_trades)
    n_gross = sum(t['gross_pnl_inr'] for t in normal_trades)
    n_charges = sum(t['charges_inr'] for t in normal_trades)
    print(f"NORMAL: n={len(normal_trades)}, avg_gross=Rs.{n_gross/len(normal_trades):.0f}, "
          f"avg_charges=Rs.{n_charges/len(normal_trades):.0f}, NET=Rs.{n_net:,.0f}")
