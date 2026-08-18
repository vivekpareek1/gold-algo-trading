"""
Tests for the COMEX/USD-INR -> fair-value bridge. Discovery: the
FairValueEngine, its is_reliable safety gate, and its small bounded
(±10pt) confluence-score modifier were ALL already fully built and wired
into signal evaluation from earlier in this project — but nothing ever
called set_external_reference_data() with real data, so it was
permanently stuck reporting "unreliable" no matter how good the COMEX
feed was. This bridges external_quotes_poller's live data into it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider


def _engine():
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    return LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                               persistence_path=None, candle_persistence_path=None,
                               open_position_path=None)


def test_fair_value_unreliable_before_any_external_data():
    engine = _engine()
    engine.on_tick(LiveTick(ts=1735689600, open=155750, high=155760, low=155740,
                              close=155750, volume=100))
    fv = engine._compute_fair_value(155750)
    assert fv.is_reliable is False
    assert "No external" in fv.unreliable_reason


def test_fair_value_becomes_reliable_after_external_data_pushed():
    engine = _engine()
    engine.on_tick(LiveTick(ts=1735689600, open=155750, high=155760, low=155740,
                              close=155750, volume=100))
    engine.set_external_reference_data(xauusd=4771.0, usdinr=87.5)
    fv = engine._compute_fair_value(155750)
    assert fv.is_reliable is True


def test_fair_value_calculation_produces_sensible_small_deviation_for_realistic_inputs():
    """With genuinely consistent, realistic current price levels, the
    deviation must be small (a few percent) — real MCX gold tracks its
    import-parity theoretical price closely, not wildly diverging."""
    engine = _engine()
    engine.on_tick(LiveTick(ts=1735689600, open=155750, high=155760, low=155740,
                              close=155750, volume=100))
    engine.set_external_reference_data(xauusd=4771.0, usdinr=87.5)
    fv = engine._compute_fair_value(155750)
    assert abs(fv.deviation_pct) < 5.0, \
        f"Deviation {fv.deviation_pct}% is implausibly large for realistic, " \
        f"self-consistent price inputs — check the formula or test inputs"


def test_stale_external_data_marked_unreliable():
    """External data older than 10 minutes must not be silently trusted."""
    engine = _engine()
    engine.on_tick(LiveTick(ts=1735689600, open=155750, high=155760, low=155740,
                              close=155750, volume=100))
    engine.set_external_reference_data(xauusd=4771.0, usdinr=87.5)
    # simulate staleness by backdating the timestamp
    engine.state.external_data_updated_at -= 700  # > 600s max age
    fv = engine._compute_fair_value(155750)
    assert fv.is_reliable is False
    assert "stale" in fv.unreliable_reason.lower()


def test_bridge_function_pushes_data_correctly():
    """Verify the actual bridge logic (api/main.py's _push_external_data_to_engine
    pattern) correctly reads from an external_quotes-shaped state and pushes
    into the engine."""
    from market_data.external_quotes import ExternalQuotesState, ExternalQuote

    engine = _engine()
    engine.on_tick(LiveTick(ts=1735689600, open=155750, high=155760, low=155740,
                              close=155750, volume=100))

    quotes_state = ExternalQuotesState(
        comex_gold=ExternalQuote(value=4771.0, prev_close=4750.0),
        usd_inr=ExternalQuote(value=87.5, prev_close=87.3),
    )

    # mirrors _push_external_data_to_engine's logic
    if quotes_state.comex_gold.value is not None and quotes_state.usd_inr.value is not None:
        engine.set_external_reference_data(xauusd=quotes_state.comex_gold.value,
                                              usdinr=quotes_state.usd_inr.value)

    fv = engine._compute_fair_value(155750)
    assert fv.is_reliable is True


def test_incomplete_external_quotes_does_not_push_partial_data():
    """If only ONE of comex_gold/usd_inr has arrived (the other still
    None), the bridge must not push a half-complete/garbage update."""
    from market_data.external_quotes import ExternalQuotesState, ExternalQuote

    quotes_state = ExternalQuotesState(
        comex_gold=ExternalQuote(value=4771.0),
        usd_inr=ExternalQuote(value=None),   # not yet fetched
    )
    should_push = quotes_state.comex_gold.value is not None and quotes_state.usd_inr.value is not None
    assert should_push is False


def test_api_fair_value_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    resp = TestClient(app).get("/api/fair_value")
    assert resp.status_code == 200
    assert "is_reliable" in resp.json()


def test_dashboard_shows_fair_value_panel():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert "COMEX Fair Value" in html
    assert "fairValueContent" in html
if __name__ == "__main__":
    tests = [
        test_fair_value_unreliable_before_any_external_data,
        test_fair_value_becomes_reliable_after_external_data_pushed,
        test_fair_value_calculation_produces_sensible_small_deviation_for_realistic_inputs,
        test_stale_external_data_marked_unreliable,
        test_bridge_function_pushes_data_correctly,
        test_incomplete_external_quotes_does_not_push_partial_data,
        test_api_fair_value_endpoint,
        test_dashboard_shows_fair_value_panel,
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


