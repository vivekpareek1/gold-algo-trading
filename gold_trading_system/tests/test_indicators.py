import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from indicators.incremental import IndicatorEngine, EMAState, RSIState, ATRState


def test_ema_converges_to_price_in_flat_market():
    """EMA of a constant price series should converge to that price."""
    ema = EMAState(period=9)
    for _ in range(30):
        val = ema.update(100.0)
    print(f"EMA9 after 30 updates of flat 100.0: {val}")
    assert abs(val - 100.0) < 0.01, f"Expected EMA to converge to 100.0, got {val}"


def test_ema_seeds_with_first_price():
    ema = EMAState(period=9)
    val = ema.update(50.0)
    assert val == 50.0, f"First EMA value should equal first price, got {val}"


def test_rsi_all_gains_hits_100():
    """A strictly rising price series should push RSI toward 100."""
    rsi = RSIState(period=14)
    price = 100.0
    val = 50.0
    for _ in range(30):
        price += 1.0
        val = rsi.update(price)
    print(f"RSI after 30 consecutive gains: {val}")
    assert val > 95, f"Expected RSI near 100 for all-gains series, got {val}"


def test_rsi_all_losses_hits_0():
    rsi = RSIState(period=14)
    price = 100.0
    val = 50.0
    for _ in range(30):
        price -= 1.0
        val = rsi.update(price)
    print(f"RSI after 30 consecutive losses: {val}")
    assert val < 5, f"Expected RSI near 0 for all-losses series, got {val}"


def test_rsi_never_out_of_bounds():
    """RSI must always stay within [0, 100] regardless of input pattern."""
    rsi = RSIState(period=14)
    import random
    random.seed(42)
    price = 100.0
    for _ in range(200):
        price += random.uniform(-5, 5)
        val = rsi.update(price)
        assert 0 <= val <= 100, f"RSI out of bounds: {val}"


def test_atr_zero_when_flat():
    """ATR of a perfectly flat market (no range) should be 0."""
    atr = ATRState(period=14)
    val = 0.0
    for _ in range(20):
        val = atr.update(high=100.0, low=100.0, close=100.0)
    print(f"ATR after 20 flat candles: {val}")
    assert abs(val) < 0.001, f"Expected ATR ~0 for flat market, got {val}"


def test_atr_captures_gap():
    """ATR must account for gaps (high/low vs prior close), not just current range."""
    atr = ATRState(period=14)
    atr.update(high=100, low=99, close=100)   # establish prev_close = 100
    val = atr.update(high=101, low=100.5, close=100.8)  # small range but gapped up from 100
    # true range should be max(101-100.5=0.5, |101-100|=1, |100.5-100|=0.5) = 1.0, not 0.5
    print(f"ATR with gap: {val}")
    assert val > 0.5, f"ATR should reflect the gap (TR=1.0), not just the candle's own range (0.5), got {val}"


def test_full_engine_produces_all_fields_no_none_after_warmup():
    engine = IndicatorEngine()
    result = None
    for i in range(60):
        price = 100 + i * 0.1
        result = engine.update(high=price + 0.5, low=price - 0.5, close=price, volume=1000 + i * 10)

    print(f"Engine output after warmup: {result}")
    for key, val in result.items():
        assert val is not None, f"Field {key} is still None after 60 candles of warmup"


def test_indicator_engine_handles_single_candle_no_crash():
    """Sanity: must not crash on the very first candle (startup edge case)."""
    engine = IndicatorEngine()
    result = engine.update(high=101, low=99, close=100, volume=1000)
    assert result["ema9"] == 100.0  # seeded correctly
    assert result["rsi"] == 50.0    # neutral default before enough data


def test_rel_volume_calculation():
    """rel_volume should reflect current volume vs its own rolling average."""
    engine = IndicatorEngine(volume_avg_period=5)
    for _ in range(5):
        engine.update(high=101, low=99, close=100, volume=1000)  # steady volume
    result = engine.update(high=101, low=99, close=100, volume=2000)  # spike
    print(f"Relative volume on spike: {result['rel_volume']}")
    assert result["rel_volume"] > 1.5, \
        f"Expected rel_volume to reflect the spike clearly, got {result['rel_volume']}"


if __name__ == "__main__":
    tests = [
        test_ema_converges_to_price_in_flat_market,
        test_ema_seeds_with_first_price,
        test_rsi_all_gains_hits_100,
        test_rsi_all_losses_hits_0,
        test_rsi_never_out_of_bounds,
        test_atr_zero_when_flat,
        test_atr_captures_gap,
        test_full_engine_produces_all_fields_no_none_after_warmup,
        test_indicator_engine_handles_single_candle_no_crash,
        test_rel_volume_calculation,
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
