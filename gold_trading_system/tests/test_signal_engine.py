import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_structure.structure_engine import StructureState, TrendState, StructureEvent, FVG
from situation_analysis.situation_analyzer import SituationSnapshot, MarketRegime
from gold_intelligence.fair_value import FairValueResult, MacroBiasResult
from signal_engine.signal_engine import SignalEngine, ConfluenceInputs, Decision


def make_situation(regime=MarketRegime.TRENDING_UP, alignment=90, momentum="STRONG",
                    ready=True, pullback=False, warnings=None):
    return SituationSnapshot(
        regime=regime, htf_trend=TrendState.TRENDING_UP, ltf_trend=TrendState.TRENDING_UP,
        trend_alignment_score=alignment, momentum_health=momentum,
        last_structure_event=StructureEvent.NONE, is_pullback_opportunity=pullback,
        macro_alignment="SUPPORTIVE", ready_for_ai_review=ready,
        explanation="test explanation", warnings=warnings or [],
    )


def make_structure(trend=TrendState.TRENDING_UP, event=StructureEvent.NONE, fvgs=None):
    s = StructureState()
    s.trend = trend
    s.last_event = event
    s.active_fvgs = fvgs or []
    return s


def make_fair_value(reliable=True, zscore=0.0):
    return FairValueResult(
        mcx_price=63000, theoretical_price=63000, deviation=0, deviation_pct=0,
        deviation_zscore=zscore, is_reliable=reliable,
    )


def make_macro(bias=0.0):
    return MacroBiasResult(macro_bias=bias, components={"session_quality_ok": True})


def strong_bullish_inputs(stacked=False):
    fvgs = [FVG(start_index=0, end_index=2, top=63050, bottom=63000, direction="BULLISH")] if stacked else []
    event = StructureEvent.LIQUIDITY_SWEEP_LOW if stacked else StructureEvent.NONE
    return ConfluenceInputs(
        ltf_structure=make_structure(trend=TrendState.TRENDING_UP, event=event, fvgs=fvgs),
        situation=make_situation(),
        fair_value=make_fair_value(),
        macro=make_macro(bias=30),
        ema_aligned_bullish=True, ema_aligned_bearish=False,
        macd_bullish=True, macd_bearish=False,
        rsi=60, price_above_vwap=True, volume_supportive=True, bb_squeeze=False,
    )


def test_strong_bullish_setup_produces_buy():
    engine = SignalEngine(Settings())
    inputs = strong_bullish_inputs()
    result = engine.evaluate(inputs)
    print(f"Long score: {result.long_score}, Short score: {result.short_score}, "
          f"Decision: {result.decision}, Confidence: {result.confidence}")
    assert result.decision == Decision.BUY
    assert result.long_score > result.short_score
    assert result.long_score >= 70, f"Strong aligned setup should score >=70, got {result.long_score}"


def test_stacked_confluence_detected_and_boosts_confidence():
    engine = SignalEngine(Settings())
    plain = engine.evaluate(strong_bullish_inputs(stacked=False))
    stacked = engine.evaluate(strong_bullish_inputs(stacked=True))
    print(f"Plain confidence: {plain.confidence}, Stacked confidence: {stacked.confidence}, "
          f"stacked_flag={stacked.stacked_confluence}")
    assert stacked.stacked_confluence == True
    assert plain.stacked_confluence == False
    assert stacked.confidence >= plain.confidence, \
        "Stacked confluence (sweep+FVG+volume) should boost confidence, not reduce it"


def test_not_ready_for_ai_forces_no_trade():
    """The Sprint 1 latency gate: if situation says not ready, must be NO_TRADE regardless of scores."""
    engine = SignalEngine(Settings())
    inputs = strong_bullish_inputs()
    inputs.situation = make_situation(ready=False)
    result = engine.evaluate(inputs)
    assert result.decision == Decision.NO_TRADE
    assert result.confidence == 0


def test_ambiguous_market_no_coin_flip_trade():
    """When long and short score similarly, must NOT force a directional trade."""
    engine = SignalEngine(Settings())
    inputs = ConfluenceInputs(
        ltf_structure=make_structure(trend=TrendState.RANGE),
        situation=make_situation(regime=MarketRegime.RANGE, alignment=50, momentum="WEAKENING"),
        fair_value=make_fair_value(),
        macro=make_macro(bias=0),
        ema_aligned_bullish=False, ema_aligned_bearish=False,
        macd_bullish=False, macd_bearish=False,
        rsi=50, price_above_vwap=True, volume_supportive=False, bb_squeeze=True,
    )
    result = engine.evaluate(inputs)
    print(f"Ambiguous market: long={result.long_score}, short={result.short_score}, "
          f"decision={result.decision}")
    assert result.decision == Decision.NO_TRADE


def test_rsi_not_naively_shorted_above_70():
    """Per spec: don't auto-short just because RSI>70 in a strong uptrend."""
    engine = SignalEngine(Settings())
    inputs = strong_bullish_inputs()
    inputs.rsi = 78  # overbought but trend is genuinely strong
    result = engine.evaluate(inputs)
    print(f"High RSI in strong uptrend: decision={result.decision}, long_score={result.long_score}")
    # should still be able to go long — RSI alone must not veto a strong trend
    assert result.decision == Decision.BUY, \
        "A strong aligned uptrend should not be blocked purely because RSI is overbought"


def test_macro_never_flips_decision_alone():
    """macro_bias must be a modifier, not able to single-handedly flip BUY to SELL."""
    engine = SignalEngine(Settings())
    inputs = strong_bullish_inputs()
    inputs.macro = make_macro(bias=-100)  # maximally opposing macro
    result = engine.evaluate(inputs)
    print(f"Strong technical setup with opposing macro: long={result.long_score}, "
          f"short={result.short_score}, decision={result.decision}")
    # macro can reduce conviction but a technically strong setup should still
    # win over an empty short case (short side has no technical support at all)
    assert result.decision != Decision.SELL, \
        "Macro alone should not flip a technically strong long setup into a short"


def test_no_trade_populates_warnings_as_reasons_against():
    engine = SignalEngine(Settings())
    inputs = ConfluenceInputs(
        ltf_structure=make_structure(trend=TrendState.RANGE),
        situation=make_situation(regime=MarketRegime.NOISY_MARKET, alignment=20,
                                   momentum="DEAD", ready=False,
                                   warnings=["Session quality flag active"]),
        fair_value=make_fair_value(),
        macro=make_macro(),
        ema_aligned_bullish=False, ema_aligned_bearish=False,
        macd_bullish=False, macd_bearish=False,
        rsi=50, price_above_vwap=True, volume_supportive=False, bb_squeeze=False,
    )
    result = engine.evaluate(inputs)
    assert result.decision == Decision.NO_TRADE
    assert any("Session quality" in r for r in result.reasons_against)


def test_invalidation_condition_set_for_buy():
    engine = SignalEngine(Settings())
    result = engine.evaluate(strong_bullish_inputs())
    assert "CHOCH" in result.invalidation_condition


if __name__ == "__main__":
    tests = [
        test_strong_bullish_setup_produces_buy,
        test_stacked_confluence_detected_and_boosts_confidence,
        test_not_ready_for_ai_forces_no_trade,
        test_ambiguous_market_no_coin_flip_trade,
        test_rsi_not_naively_shorted_above_70,
        test_macro_never_flips_decision_alone,
        test_no_trade_populates_warnings_as_reasons_against,
        test_invalidation_condition_set_for_buy,
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
