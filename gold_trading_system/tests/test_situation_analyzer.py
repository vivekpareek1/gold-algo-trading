import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_structure.structure_engine import StructureState, TrendState, StructureEvent
from situation_analysis.situation_analyzer import (
    SituationAnalyzer, IndicatorSnapshot, MacroContext, MarketRegime
)


def make_structure(trend, last_event=StructureEvent.NONE, pullback=False):
    s = StructureState()
    s.trend = trend
    s.last_event = last_event
    s.is_pullback_in_trend = pullback
    return s


def test_aligned_uptrend_strong_momentum():
    analyzer = SituationAnalyzer(Settings())
    htf = make_structure(TrendState.TRENDING_UP)
    ltf = make_structure(TrendState.TRENDING_UP)
    ind = IndicatorSnapshot(ema9=105, ema21=102, ema50=98, macd_hist=2.0, macd_hist_prev=1.0,
                             rel_volume=1.3, atr=10, atr_avg_20=10)
    macro = MacroContext(macro_bias=30)

    snap = analyzer.analyze(htf, ltf, ind, macro)
    print(f"Regime: {snap.regime}, alignment: {snap.trend_alignment_score}, "
          f"momentum: {snap.momentum_health}, macro: {snap.macro_alignment}")
    print(f"Explanation: {snap.explanation}")

    assert snap.regime == MarketRegime.TRENDING_UP
    assert snap.trend_alignment_score == 90
    assert snap.momentum_health == "STRONG"
    assert snap.macro_alignment == "SUPPORTIVE"
    assert snap.ready_for_ai_review == True


def test_pullback_classified_correctly_not_conflict():
    analyzer = SituationAnalyzer(Settings())
    htf = make_structure(TrendState.TRENDING_UP)
    ltf = make_structure(TrendState.TRENDING_DOWN, pullback=True)
    ind = IndicatorSnapshot(ema9=100, ema21=101, ema50=99, macd_hist=-0.5, macd_hist_prev=-1.0,
                             rel_volume=0.9, atr=10, atr_avg_20=10)
    macro = MacroContext(macro_bias=10)

    snap = analyzer.analyze(htf, ltf, ind, macro)
    print(f"Pullback test - alignment: {snap.trend_alignment_score}, pullback: {snap.is_pullback_opportunity}")

    assert snap.is_pullback_opportunity == True
    assert snap.trend_alignment_score == 65, \
        f"Pullback should score 65 (opportunity, not conflict), got {snap.trend_alignment_score}"


def test_genuine_conflict_scores_low():
    analyzer = SituationAnalyzer(Settings())
    htf = make_structure(TrendState.TRENDING_UP)
    ltf = make_structure(TrendState.TRENDING_DOWN, pullback=False)  # NOT classified as pullback
    ind = IndicatorSnapshot(rel_volume=1.0, atr=10, atr_avg_20=10)
    macro = MacroContext(macro_bias=0)

    snap = analyzer.analyze(htf, ltf, ind, macro)
    assert snap.trend_alignment_score == 15, \
        f"Genuine conflict should score 15, got {snap.trend_alignment_score}"
    assert snap.ready_for_ai_review == False, \
        "Low alignment score should NOT be ready for AI review (keeps AI out of hot path)"


def test_reversal_possible_on_choch():
    analyzer = SituationAnalyzer(Settings())
    htf = make_structure(TrendState.TRENDING_UP)
    ltf = make_structure(TrendState.RANGE, last_event=StructureEvent.CHOCH_BEARISH)
    ind = IndicatorSnapshot(rel_volume=1.2, atr=10, atr_avg_20=10)
    macro = MacroContext(macro_bias=0)

    snap = analyzer.analyze(htf, ltf, ind, macro)
    print(f"CHOCH test regime: {snap.regime}")
    assert snap.regime == MarketRegime.REVERSAL_POSSIBLE


def test_high_volatility_detected():
    analyzer = SituationAnalyzer(Settings())
    htf = make_structure(TrendState.RANGE)
    ltf = make_structure(TrendState.RANGE)
    ind = IndicatorSnapshot(atr=25, atr_avg_20=10, rel_volume=1.0)  # 2.5x normal ATR
    macro = MacroContext(macro_bias=0)

    snap = analyzer.analyze(htf, ltf, ind, macro)
    assert snap.regime == MarketRegime.HIGH_VOLATILITY
    assert any("abnormal volatility" in w for w in snap.warnings), \
        "Expected a volatility warning to be raised"


def test_noisy_market_not_ready_for_ai():
    """Dead momentum + range on both TFs -> noise, AI should NOT be triggered."""
    analyzer = SituationAnalyzer(Settings())
    htf = make_structure(TrendState.RANGE)
    ltf = make_structure(TrendState.RANGE)
    # genuinely flat: EMAs identical, macd histogram unchanged, low relative volume
    ind = IndicatorSnapshot(ema9=100.0, ema21=100.0, ema50=100.0, macd_hist=0.01, macd_hist_prev=0.01,
                             rel_volume=0.5, atr=10, atr_avg_20=10)
    macro = MacroContext(macro_bias=0)

    snap = analyzer.analyze(htf, ltf, ind, macro)
    print(f"Noisy market regime: {snap.regime}, ready_for_ai: {snap.ready_for_ai_review}")
    assert snap.regime == MarketRegime.NOISY_MARKET
    assert snap.ready_for_ai_review == False


def test_sweep_generates_warning():
    analyzer = SituationAnalyzer(Settings())
    htf = make_structure(TrendState.TRENDING_UP)
    ltf = make_structure(TrendState.TRENDING_UP, last_event=StructureEvent.LIQUIDITY_SWEEP_LOW)
    ind = IndicatorSnapshot(rel_volume=1.0, atr=10, atr_avg_20=10)
    macro = MacroContext(macro_bias=0, session_quality_ok=True)

    snap = analyzer.analyze(htf, ltf, ind, macro)
    assert any("LIQUIDITY_SWEEP_LOW" in w for w in snap.warnings), \
        f"Expected sweep warning, got warnings={snap.warnings}"


if __name__ == "__main__":
    tests = [
        test_aligned_uptrend_strong_momentum,
        test_pullback_classified_correctly_not_conflict,
        test_genuine_conflict_scores_low,
        test_reversal_possible_on_choch,
        test_high_volatility_detected,
        test_noisy_market_not_ready_for_ai,
        test_sweep_generates_warning,
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
