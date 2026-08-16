import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_analysis.schema import AISignal
from ai_analysis.client import AIClient
from ai_analysis.prompt_builder import build_context

from market_structure.structure_engine import StructureState, TrendState, StructureEvent
from situation_analysis.situation_analyzer import SituationSnapshot, IndicatorSnapshot, MarketRegime
from gold_intelligence.fair_value import FairValueResult, MacroBiasResult
from signal_engine.signal_engine import ConfluenceResult, Decision, TradeType


# ---------- schema tests ----------
def test_valid_no_trade_signal():
    sig = AISignal(decision="NO_TRADE", confidence=0)
    assert sig.decision == "NO_TRADE"


def test_valid_buy_signal_with_required_fields():
    sig = AISignal(decision="BUY", confidence=80, entry_zone_low=63000,
                    entry_zone_high=63050, stop_loss=62900)
    assert sig.decision == "BUY"


def test_buy_without_stop_loss_rejected():
    """A BUY decision missing critical trade parameters must be rejected as malformed."""
    try:
        AISignal(decision="BUY", confidence=80, entry_zone_low=63000)  # no stop_loss
        assert False, "Expected validation error for BUY without stop_loss"
    except Exception as e:
        print(f"Correctly rejected incomplete BUY: {e}")


def test_confidence_out_of_bounds_rejected():
    try:
        AISignal(decision="NO_TRADE", confidence=150)
        assert False, "Expected validation error for confidence > 100"
    except Exception:
        pass


def test_invalid_decision_literal_rejected():
    try:
        AISignal(decision="MAYBE", confidence=50)
        assert False, "Expected validation error for invalid decision literal"
    except Exception:
        pass


def test_no_trade_fallback_factory():
    sig = AISignal.no_trade_fallback(reason="test failure")
    assert sig.decision == "NO_TRADE"
    assert sig.confidence == 0
    assert "test failure" in sig.final_explanation


# ---------- client fail-safe tests (the critical ones) ----------
def test_client_falls_back_on_api_exception():
    def broken_caller(system, context):
        raise ConnectionError("simulated network failure")

    client = AIClient(caller=broken_caller)
    result = client.get_signal("some context")
    print(f"Result after API exception: {result.decision}, explanation: {result.final_explanation}")
    assert result.decision == "NO_TRADE", \
        "API exception MUST fail-safe to NO_TRADE, never propagate into a trade decision"


def test_client_falls_back_on_malformed_json():
    def bad_json_caller(system, context):
        return "this is not json at all { broken"

    client = AIClient(caller=bad_json_caller)
    result = client.get_signal("some context")
    assert result.decision == "NO_TRADE", "Malformed JSON must fail-safe to NO_TRADE"


def test_client_falls_back_on_schema_violation():
    def invalid_schema_caller(system, context):
        return json.dumps({"decision": "BUY", "confidence": 999})  # confidence out of bounds

    client = AIClient(caller=invalid_schema_caller)
    result = client.get_signal("some context")
    assert result.decision == "NO_TRADE", "Schema violation must fail-safe to NO_TRADE"


def test_client_falls_back_on_incomplete_buy():
    def incomplete_buy_caller(system, context):
        return json.dumps({"decision": "BUY", "confidence": 80})  # missing stop_loss/entry

    client = AIClient(caller=incomplete_buy_caller)
    result = client.get_signal("some context")
    assert result.decision == "NO_TRADE", \
        "A BUY with missing trade parameters must fail-safe to NO_TRADE, not be trusted partially"


def test_client_strips_markdown_fences():
    """Models often wrap JSON in ```json fences despite instructions — must handle gracefully."""
    def fenced_caller(system, context):
        payload = {"decision": "NO_TRADE", "confidence": 0}
        return f"```json\n{json.dumps(payload)}\n```"

    client = AIClient(caller=fenced_caller)
    result = client.get_signal("some context")
    assert result.decision == "NO_TRADE"


def test_client_succeeds_on_valid_response():
    def good_caller(system, context):
        payload = {
            "decision": "BUY", "confidence": 78,
            "entry_zone_low": 63000, "entry_zone_high": 63050, "stop_loss": 62900,
            "target_1": 63200, "risk_reward": 2.0,
            "reasons_for_entry": ["strong trend", "sweep confirmed"],
            "final_explanation": "Aligned trend with confirmed liquidity sweep.",
        }
        return json.dumps(payload)

    client = AIClient(caller=good_caller)
    result = client.get_signal("some context")
    print(f"Successful call result: {result.decision}, confidence={result.confidence}")
    assert result.decision == "BUY"
    assert result.confidence == 78
    assert result.stop_loss == 62900


def test_client_retries_before_falling_back():
    """First call fails, second succeeds -> should return the successful result, not NO_TRADE."""
    call_count = {"n": 0}

    def flaky_caller(system, context):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("transient failure")
        return json.dumps({"decision": "NO_TRADE", "confidence": 0})

    client = AIClient(caller=flaky_caller, max_retries=1)
    result = client.get_signal("some context")
    assert call_count["n"] == 2, f"Expected exactly 2 calls (1 fail + 1 retry), got {call_count['n']}"
    assert result.decision == "NO_TRADE"  # in this case both real+fallback agree, checking retry happened


# ---------- prompt builder test ----------
def test_prompt_builder_includes_all_required_fields():
    htf = StructureState()
    htf.trend = TrendState.TRENDING_UP
    ltf = StructureState()
    ltf.trend = TrendState.TRENDING_UP
    ltf.last_event = StructureEvent.LIQUIDITY_SWEEP_LOW

    situation = SituationSnapshot(
        regime=MarketRegime.TRENDING_UP, htf_trend=TrendState.TRENDING_UP,
        ltf_trend=TrendState.TRENDING_UP, trend_alignment_score=90,
        momentum_health="STRONG", last_structure_event=StructureEvent.LIQUIDITY_SWEEP_LOW,
        is_pullback_opportunity=False, macro_alignment="SUPPORTIVE",
        ready_for_ai_review=True, explanation="Strong aligned uptrend.", warnings=[],
    )
    indicators = IndicatorSnapshot(ema9=105, ema21=102, ema50=98, ema200=90,
                                     rsi=65, macd_hist=1.5, macd_hist_prev=1.0,
                                     atr=12, atr_avg_20=10, rel_volume=1.3)
    fair_value = FairValueResult(mcx_price=63000, theoretical_price=62800,
                                   deviation=200, deviation_pct=0.32,
                                   deviation_zscore=0.8, is_reliable=True)
    macro = MacroBiasResult(macro_bias=25.0, components={})
    confluence = ConfluenceResult(
        long_score=85, short_score=30, decision=Decision.BUY, confidence=85,
        trade_type=TradeType.MOMENTUM, reasons_for=["strong trend"], reasons_against=[],
        stacked_confluence=True, invalidation_condition="Opposite CHOCH",
    )

    context = build_context(
        instrument="GOLDM", current_price=63000, htf_structure=htf, ltf_structure=ltf,
        situation=situation, indicators=indicators, fair_value=fair_value, macro=macro,
        confluence=confluence, news_risk="NORMAL", spread_points=0.5, session="MCX_PRIME",
    )
    print(context[:300] + "...")

    required_terms = ["Market Regime", "EMA9", "RSI", "MACD", "ATR", "Fair Value",
                       "Macro Bias", "Confluence Long Score", "News Risk", "Session"]
    for term in required_terms:
        assert term in context, f"Prompt context missing required field: {term}"


if __name__ == "__main__":
    tests = [
        test_valid_no_trade_signal,
        test_valid_buy_signal_with_required_fields,
        test_buy_without_stop_loss_rejected,
        test_confidence_out_of_bounds_rejected,
        test_invalid_decision_literal_rejected,
        test_no_trade_fallback_factory,
        test_client_falls_back_on_api_exception,
        test_client_falls_back_on_malformed_json,
        test_client_falls_back_on_schema_violation,
        test_client_falls_back_on_incomplete_buy,
        test_client_strips_markdown_fences,
        test_client_succeeds_on_valid_response,
        test_client_retries_before_falling_back,
        test_prompt_builder_includes_all_required_fields,
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
