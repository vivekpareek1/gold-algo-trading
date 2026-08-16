import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_structure.structure_engine import (
    MarketStructureEngine, Candle, TrendState, StructureEvent
)


def make_candle(ts, o, h, l, c, v=1000):
    return Candle(ts=ts, open=o, high=h, low=l, close=c, volume=v)


def _build_zigzag(pivots, kinds, start_close=100.0):
    """Build clean OHLC candles from a pivot sequence: small-body drift
    candles with a wick reaching each pivot value, so neighbouring candle
    bodies don't contaminate the fractal swing-detection check."""
    candles = []
    prev_close = start_close
    for p, k in zip(pivots, kinds):
        o = prev_close
        if k == "LOW":
            c = o - 2
            h = max(o, c) + 1
            l = p
        else:
            c = o + 2
            l = min(o, c) - 1
            h = p
        candles.append((o, h, l, c))
        prev_close = c
    return candles


def test_uptrend_detection():
    """Higher highs + higher lows should register as TRENDING_UP."""
    engine = MarketStructureEngine(swing_lookback=1)
    # ascending envelope: HIGH,LOW,HIGH,LOW,HIGH,LOW,HIGH each pivot better than the last
    pivots = [95, 70, 105, 80, 115, 90, 125]
    kinds = ["HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH"]
    candles = _build_zigzag(pivots, kinds)
    state = None
    for i, (o, h, l, c) in enumerate(candles):
        state = engine.update(make_candle(i, o, h, l, c), current_atr=2.0)

    print(f"Trend after uptrend sequence: {state.trend}, "
          f"highs={[s.price for s in state.swing_highs]}, lows={[s.price for s in state.swing_lows]}")
    assert state.trend == TrendState.TRENDING_UP, \
        f"Expected TRENDING_UP, got {state.trend}"


def test_liquidity_sweep_detection():
    """Wick beyond swing high, close back inside => LIQUIDITY_SWEEP_HIGH."""
    engine = MarketStructureEngine(swing_lookback=1)
    # build a clear swing high at 110
    setup = [
        (100, 102, 99, 101),
        (101, 105, 100, 103),
        (103, 110, 102, 105),   # swing high candidate ~110
        (105, 106, 103, 104),
        (104, 107, 102, 105),
    ]
    for i, (o, h, l, c) in enumerate(setup):
        engine.update(make_candle(i, o, h, l, c), current_atr=2.0)

    # now sweep it: wick to 111 (above 110) but close back at 106 (below 110)
    sweep_candle = make_candle(len(setup), 105, 111, 104, 106)
    state = engine.update(sweep_candle, current_atr=2.0)

    print(f"Event after sweep candle: {state.last_event}")
    print(f"Swing highs: {[(s.price, s.swept) for s in state.swing_highs]}")
    assert state.last_event == StructureEvent.LIQUIDITY_SWEEP_HIGH, \
        f"Expected LIQUIDITY_SWEEP_HIGH, got {state.last_event}"


def test_fvg_detection():
    """Gap between candle[-3].high and candle[-1].low => bullish FVG."""
    engine = MarketStructureEngine(swing_lookback=1)
    c1 = make_candle(0, 100, 102, 99, 101)   # high = 102
    c2 = make_candle(1, 101, 108, 100, 107)  # big bullish candle
    c3 = make_candle(2, 107, 112, 106, 110)  # low = 106, gap vs c1.high=102

    engine.update(c1, current_atr=2.0)
    engine.update(c2, current_atr=2.0)
    state = engine.update(c3, current_atr=2.0)

    print(f"Active FVGs: {[(f.direction, f.bottom, f.top) for f in state.active_fvgs]}")
    bullish_fvgs = [f for f in state.active_fvgs if f.direction == "BULLISH"]
    assert len(bullish_fvgs) >= 1, "Expected at least one bullish FVG detected"
    fvg = bullish_fvgs[0]
    assert fvg.bottom == 102 and fvg.top == 106, \
        f"FVG boundaries wrong: expected (102,106), got ({fvg.bottom},{fvg.top})"


def test_pullback_vs_conflict_classification():
    """Lower TF genuinely in a down-move should be flagged as pullback when
    higher TF is up, not as conflict."""
    engine = MarketStructureEngine(swing_lookback=1)
    pivots = [90, 118, 80, 108, 70, 98, 60]  # descending envelope, verified working above
    kinds = ["LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW"]
    candles = _build_zigzag(pivots, kinds)
    state = None
    for i, (o, h, l, c) in enumerate(candles):
        state = engine.update(make_candle(i, o, h, l, c), current_atr=2.0,
                               higher_tf_trend=TrendState.TRENDING_UP)

    print(f"Lower TF trend detected: {state.trend}, is_pullback: {state.is_pullback_in_trend}")
    assert state.trend == TrendState.TRENDING_DOWN, \
        f"Setup should produce a detected downtrend on the lower TF, got {state.trend}"
    assert state.is_pullback_in_trend == True, \
        "Lower TF down-move while higher TF is up should be classified as pullback, not conflict"


def test_no_unclosed_candle_corruption():
    """Sanity: engine should not crash on minimal data (edge case: startup)."""
    engine = MarketStructureEngine(swing_lookback=3)
    engine.update(make_candle(0, 100, 101, 99, 100), current_atr=1.0)
    assert engine.state.trend == TrendState.RANGE  # not enough data yet, must not crash


if __name__ == "__main__":
    tests = [
        test_uptrend_detection,
        test_liquidity_sweep_detection,
        test_fvg_detection,
        test_pullback_vs_conflict_classification,
        test_no_unclosed_candle_corruption,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__} -> {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
