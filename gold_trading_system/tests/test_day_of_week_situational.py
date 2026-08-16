import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from situation_analysis.day_of_week_situational import DayOfWeekAnalyzer, DailyCandle, Weekday


def make_candle(day, open_, close, high=None, low=None):
    high = high or max(open_, close) + 1
    low = low or min(open_, close) - 1
    return DailyCandle(ts_epoch_day=0, weekday=day, open=open_, high=high, low=low, close=close)


def test_hand_calculated_stats():
    """5 Monday candles, hand-calculate expected return/win rate."""
    analyzer = DayOfWeekAnalyzer()
    candles = [
        make_candle(Weekday.MONDAY, 100, 102),   # +2%, win
        make_candle(Weekday.MONDAY, 100, 101),   # +1%, win
        make_candle(Weekday.MONDAY, 100, 99),    # -1%, loss
        make_candle(Weekday.MONDAY, 100, 103),   # +3%, win
        make_candle(Weekday.MONDAY, 100, 98),    # -2%, loss
    ]
    stats = analyzer.compute_weekday_stats(candles)
    mon = stats[Weekday.MONDAY]
    print(f"Monday stats: {mon}")

    assert mon.sample_size == 5
    # win_rate = 3/5 = 60%
    assert abs(mon.win_rate_pct - 60.0) < 0.001
    # avg_return = (2+1-1+3-2)/5 = 3/5 = 0.6%
    assert abs(mon.avg_return_pct - 0.6) < 0.001


def test_insufficient_sample_flagged_unreliable():
    analyzer = DayOfWeekAnalyzer(min_sample_size=20)
    candles = [make_candle(Weekday.TUESDAY, 100, 105) for _ in range(5)]  # only 5, need 20
    stats = analyzer.compute_weekday_stats(candles)
    tue = stats[Weekday.TUESDAY]
    assert tue.is_reliable == False, "5 samples with min_sample_size=20 must be flagged unreliable"

    bias = analyzer.get_bias(Weekday.TUESDAY, stats)
    assert bias.bias_score == 0.0, "Unreliable pattern must produce a neutral (0) bias, never a guess"
    assert bias.is_reliable == False


def test_strong_genuine_edge_detected():
    """30 Wednesdays, 26 green (87% win rate) -> should be flagged as a real edge."""
    analyzer = DayOfWeekAnalyzer(min_sample_size=20, min_win_rate_edge_pct=8.0)
    candles = []
    for i in range(26):
        candles.append(make_candle(Weekday.WEDNESDAY, 100, 101))  # green
    for i in range(4):
        candles.append(make_candle(Weekday.WEDNESDAY, 100, 99))   # red

    stats = analyzer.compute_weekday_stats(candles)
    bias = analyzer.get_bias(Weekday.WEDNESDAY, stats)
    print(f"Strong edge bias: {bias}")

    assert bias.is_reliable == True
    assert bias.bias_score > 0, "87% win rate should produce a positive (bullish) bias"


def test_noise_level_edge_not_flagged():
    """30 samples but win rate only 53% (near 50/50) -> should NOT be flagged as a real edge."""
    analyzer = DayOfWeekAnalyzer(min_sample_size=20, min_win_rate_edge_pct=8.0)
    candles = []
    for i in range(16):
        candles.append(make_candle(Weekday.THURSDAY, 100, 101))
    for i in range(14):
        candles.append(make_candle(Weekday.THURSDAY, 100, 99))

    stats = analyzer.compute_weekday_stats(candles)
    bias = analyzer.get_bias(Weekday.THURSDAY, stats)
    print(f"Noise-level bias: {bias}")

    assert bias.is_reliable == True  # enough samples, just no meaningful edge
    assert bias.bias_score == 0.0, \
        f"53% win rate (within noise of 50%) should NOT be flagged as a directional edge, " \
        f"got bias_score={bias.bias_score}"


def test_zero_candles_no_crash():
    analyzer = DayOfWeekAnalyzer()
    stats = analyzer.compute_weekday_stats([])
    assert all(s.sample_size == 0 for s in stats.values())
    bias = analyzer.get_bias(Weekday.MONDAY, stats)
    assert bias.bias_score == 0.0
    assert bias.is_reliable == False


def test_bearish_edge_produces_negative_bias():
    analyzer = DayOfWeekAnalyzer(min_sample_size=20, min_win_rate_edge_pct=8.0)
    candles = []
    for i in range(24):
        candles.append(make_candle(Weekday.FRIDAY, 100, 98))  # red
    for i in range(6):
        candles.append(make_candle(Weekday.FRIDAY, 100, 102))  # green

    stats = analyzer.compute_weekday_stats(candles)
    bias = analyzer.get_bias(Weekday.FRIDAY, stats)
    print(f"Bearish edge bias: {bias}")
    assert bias.bias_score < 0, "80% red days should produce a negative (bearish) bias"


def test_bias_score_stays_within_bounds():
    """Even a 100% win rate should not exceed +/-100."""
    analyzer = DayOfWeekAnalyzer(min_sample_size=20, min_win_rate_edge_pct=8.0)
    candles = [make_candle(Weekday.MONDAY, 100, 105) for _ in range(30)]  # 100% win rate
    stats = analyzer.compute_weekday_stats(candles)
    bias = analyzer.get_bias(Weekday.MONDAY, stats)
    assert -100.0 <= bias.bias_score <= 100.0


if __name__ == "__main__":
    tests = [
        test_hand_calculated_stats,
        test_insufficient_sample_flagged_unreliable,
        test_strong_genuine_edge_detected,
        test_noise_level_edge_not_flagged,
        test_zero_candles_no_crash,
        test_bearish_edge_produces_negative_bias,
        test_bias_score_stays_within_bounds,
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
