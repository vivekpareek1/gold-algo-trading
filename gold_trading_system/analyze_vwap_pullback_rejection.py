import sys
sys.path.insert(0, '.')
from config.settings import Settings
from market_data.data_loader import DataQualityGate
from indicators.incremental import IndicatorEngine
from market_structure.structure_engine import MarketStructureEngine, Candle as StructCandle, TrendState
from execution.brokerage_calculator import calculate_charges
from backtesting.metrics import compute_metrics


def is_pin_bar(candle, direction):
    """A pin bar: small body, long wick in the direction of the expected
    reversal (long lower wick for bullish rejection, long upper wick for
    bearish rejection), body near one end of the range."""
    body = abs(candle.close - candle.open)
    total_range = candle.high - candle.low
    if total_range <= 0:
        return False
    if direction == 'LONG':
        lower_wick = min(candle.open, candle.close) - candle.low
        return lower_wick > body * 2 and lower_wick > total_range * 0.5
    else:
        upper_wick = candle.high - max(candle.open, candle.close)
        return upper_wick > body * 2 and upper_wick > total_range * 0.5


def is_engulfing(prev_candle, candle, direction):
    """A bullish engulfing candle's body must fully engulf the previous
    (opposite-colored) candle's body; symmetric for bearish."""
    if direction == 'LONG':
        prev_bearish = prev_candle.close < prev_candle.open
        curr_bullish = candle.close > candle.open
        return (prev_bearish and curr_bullish and
                 candle.open <= prev_candle.close and candle.close >= prev_candle.open)
    else:
        prev_bullish = prev_candle.close > prev_candle.open
        curr_bearish = candle.close < candle.open
        return (prev_bullish and curr_bearish and
                 candle.open >= prev_candle.close and candle.close <= prev_candle.open)


settings = Settings()
gate = DataQualityGate(settings, expected_interval_minutes=5)
candles = gate.load_csv('/mnt/user-data/uploads/mcx_goldm_5min.csv').candles

structure = MarketStructureEngine()
indicators = IndicatorEngine()

open_trade = None
trades = []
prev_candle = None

VWAP_EMA_PROXIMITY_ATR_MULT = 0.5  # "near" VWAP/EMA21 means within this
ATR_STOP_MULT = 1.5
RISK_REWARD = 2.0
RISK_PER_TRADE = 2000
POINT_VALUE = 10
MAX_LOTS = 10

for i, candle in enumerate(candles):
    sc = StructCandle(ts=candle.ts, open=candle.open, high=candle.high, low=candle.low,
                        close=candle.close, volume=candle.volume)
    structure.update(sc, current_atr=indicators.atr.value or 1.0, higher_tf_trend=TrendState.RANGE)
    ind = indicators.update(candle.high, candle.low, candle.close, candle.volume)

    if open_trade is not None:
        direction = open_trade['direction']
        stop, target = open_trade['stop'], open_trade['target']
        exit_price = None
        if direction == 'LONG':
            if candle.low <= stop:
                exit_price = stop
            elif candle.high >= target:
                exit_price = target
        else:
            if candle.high >= stop:
                exit_price = stop
            elif candle.low <= target:
                exit_price = target
        if exit_price is not None:
            charges = calculate_charges(direction=direction, entry_price=open_trade['entry'],
                                           exit_price=exit_price, lots=open_trade['lots'],
                                           point_value_inr=POINT_VALUE)
            trades.append({'net_pnl_inr': charges.net_pnl_inr, 'gross_pnl_inr': charges.gross_pnl_inr,
                             'r_multiple': RISK_REWARD if exit_price == target else -1.0})
            open_trade = None
        prev_candle = candle
        continue

    if i < 30 or ind['ema21'] is None or ind['atr'] is None or ind['vwap'] is None:
        prev_candle = candle
        continue

    atr = ind['atr'] or 10.0
    proximity = atr * VWAP_EMA_PROXIMITY_ATR_MULT

    # is price near VWAP or EMA21 (a pullback zone)?
    near_vwap = abs(candle.close - ind['vwap']) <= proximity
    near_ema21 = abs(candle.close - ind['ema21']) <= proximity

    if not (near_vwap or near_ema21):
        prev_candle = candle
        continue

    # higher-timeframe-ish trend bias via EMA9/21/50 stack (simple proxy)
    ema_bullish = ind['ema9'] > ind['ema21'] > ind['ema50']
    ema_bearish = ind['ema9'] < ind['ema21'] < ind['ema50']

    direction = None
    if ema_bullish and prev_candle is not None:
        if is_pin_bar(candle, 'LONG') or is_engulfing(prev_candle, candle, 'LONG'):
            direction = 'LONG'
    elif ema_bearish and prev_candle is not None:
        if is_pin_bar(candle, 'SHORT') or is_engulfing(prev_candle, candle, 'SHORT'):
            direction = 'SHORT'

    prev_candle = candle
    if direction is None:
        continue

    stop_distance = atr * ATR_STOP_MULT
    risk_per_lot = stop_distance * POINT_VALUE
    lots = min(int(RISK_PER_TRADE / risk_per_lot), MAX_LOTS) if risk_per_lot > 0 else 0
    if lots < 1:
        continue

    entry = candle.close
    if direction == 'LONG':
        stop = entry - stop_distance
        target = entry + stop_distance * RISK_REWARD
    else:
        stop = entry + stop_distance
        target = entry - stop_distance * RISK_REWARD

    open_trade = {'direction': direction, 'entry': entry, 'stop': stop, 'target': target, 'lots': lots}

print(f"VWAP/EMA pullback + rejection-candle strategy — Total trades: {len(trades)}")
if trades:
    r_values = [t['r_multiple'] for t in trades]
    m = compute_metrics(r_values)
    total_net = sum(t['net_pnl_inr'] for t in trades)
    total_gross = sum(t['gross_pnl_inr'] for t in trades)
    print(f"Win rate: {m.win_rate:.1f}%  Expectancy: {m.expectancy_r:+.3f}R  PF: {m.profit_factor:.2f}")
    print(f"Total gross: Rs.{total_gross:,.2f}")
    print(f"Total NET: Rs.{total_net:,.2f}")
    print(f"Avg gross/trade: Rs.{total_gross/len(trades):.0f}")
else:
    print("No trades triggered.")
