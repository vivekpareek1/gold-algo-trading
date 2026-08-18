"""
Tests for the EMA9/21/50 chart overlay — deliberately using the SAME
periods (9/21/50) the trading logic itself uses for confluence decisions,
not an arbitrary EMA20 the system never looks at.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _populate_candles(live_engine, n=60, base_ts=1735689600):
    live_engine.state.candle_history = []
    for i in range(n):
        live_engine.state.candle_history.append({
            "ts": base_ts + i * 300, "open": 155000 + i, "high": 155010 + i,
            "low": 154990 + i, "close": 155005 + i, "volume": 100 + i,
        })


def test_ema_endpoint_returns_correct_periods():
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    _populate_candles(live_engine)
    client = TestClient(app)
    resp = client.get("/api/candles/5M/ema")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["candles"]) == 60
    for c in body["candles"]:
        assert set(["ts", "close", "ema9", "ema21", "ema50"]).issubset(c.keys())


def test_ema_values_populate_progressively():
    """EMA9 should populate before EMA21, which populates before EMA50 —
    each needs its own period's worth of data first."""
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    _populate_candles(live_engine, n=60)
    client = TestClient(app)
    candles = client.get("/api/candles/5M/ema").json()["candles"]

    ema9_first_valid = next(i for i, c in enumerate(candles) if c["ema9"] is not None)
    ema21_first_valid = next(i for i, c in enumerate(candles) if c["ema21"] is not None)
    ema50_first_valid = next(i for i, c in enumerate(candles) if c["ema50"] is not None)
    assert ema9_first_valid <= ema21_first_valid <= ema50_first_valid


def test_ema_reflects_chronological_state_not_final_state():
    """
    THE core correctness requirement: EMA at candle 20 must reflect only
    what was knowable up to candle 20, not the FINAL/latest EMA state
    applied retroactively — otherwise every historical point would show
    an identical, wrong (too-smooth, look-ahead-biased) EMA value.
    """
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    _populate_candles(live_engine, n=60)
    client = TestClient(app)
    candles = client.get("/api/candles/5M/ema").json()["candles"]

    ema9_values = [c["ema9"] for c in candles if c["ema9"] is not None]
    # EMA9 values must generally increase across a monotonically rising
    # price series (not be flat/identical, which would indicate look-ahead)
    assert ema9_values[-1] > ema9_values[0], \
        "EMA must evolve chronologically across a trending price series, not be static"
    assert len(set(ema9_values)) > 5, "EMA values must vary candle-to-candle, not be one repeated value"


def test_ema_works_on_higher_timeframes():
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    _populate_candles(live_engine, n=120)
    client = TestClient(app)
    resp = client.get("/api/candles/1H/ema")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["candles"]) > 0


def test_invalid_timeframe_rejected_gracefully():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get("/api/candles/1M/ema")
    body = resp.json()
    assert body["candles"] == []
    assert "error" in body


def test_dashboard_includes_ema_line_series():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert "ema9Series" in html
    assert "ema21Series" in html
    assert "ema50Series" in html
    assert html.count("addLineSeries") == 3


def test_dashboard_structurally_valid_with_ema_overlay():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert html.count("<div") == html.count("</div>")
    assert html.count("<script") == html.count("</script>")


def test_ema_endpoint_handles_out_of_order_duplicate_candles():
    """THE core bug fix: candle_history could theoretically have an
    out-of-order or duplicate entry (restart/replay edge case). The EMA
    endpoint must sort+dedupe before replay, or (a) EMA values themselves
    become order-sensitive-wrong, and (b) the chart's OWN candle series
    could fail to render (lightweight-charts requires strictly ascending,
    unique times across ALL series sharing a chart)."""
    from fastapi.testclient import TestClient
    from api.main import app, live_engine

    live_engine.state.candle_history = []
    base_ts = 1735689600
    for i in range(60):
        live_engine.state.candle_history.append({
            "ts": base_ts + i * 300, "open": 155000 + i, "high": 155010 + i,
            "low": 154990 + i, "close": 155005 + i, "volume": 100 + i,
        })
    # inject a duplicate + an out-of-order entry
    live_engine.state.candle_history.append({"ts": base_ts + 30 * 300, "open": 155030,
                                                 "high": 155040, "low": 155020, "close": 155035, "volume": 130})
    live_engine.state.candle_history.insert(5, {"ts": base_ts + 59 * 300, "open": 155059,
                                                    "high": 155069, "low": 155049, "close": 155064, "volume": 159})

    client = TestClient(app)
    resp = client.get("/api/candles/5M/ema")
    data = resp.json()
    timestamps = [c["ts"] for c in data["candles"]]
    assert timestamps == sorted(timestamps), "EMA candle timestamps must be strictly ascending"
    assert len(timestamps) == len(set(timestamps)), "EMA candle timestamps must have no duplicates"
if __name__ == "__main__":
    tests = [
        test_ema_endpoint_returns_correct_periods,
        test_ema_values_populate_progressively,
        test_ema_reflects_chronological_state_not_final_state,
        test_ema_works_on_higher_timeframes,
        test_invalid_timeframe_rejected_gracefully,
        test_dashboard_includes_ema_line_series,
        test_dashboard_structurally_valid_with_ema_overlay,
        test_ema_endpoint_handles_out_of_order_duplicate_candles,
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


