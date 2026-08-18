"""
Tests for multi-timeframe chart support. 1M is deliberately NOT offered —
the live feed's base granularity is 5M, and 5M candles cannot be split
back into genuine 1M data (it doesn't exist); only aggregating UP to
coarser timeframes (15M/1H/4H) is valid.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _populate_candles(live_engine, n=100, base_ts=1735689600):
    for i in range(n):
        live_engine.state.candle_history.append({
            "ts": base_ts + i * 300, "open": 155000 + i, "high": 155010 + i,
            "low": 154990 + i, "close": 155005 + i, "volume": 100 + i,
        })


def test_5m_endpoint_returns_raw_candles():
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    live_engine.state.candle_history = []
    _populate_candles(live_engine, n=50)
    client = TestClient(app)
    resp = client.get("/api/candles/5M")
    assert resp.status_code == 200
    assert len(resp.json()["candles"]) == 50


def test_15m_aggregates_roughly_three_to_one():
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    live_engine.state.candle_history = []
    _populate_candles(live_engine, n=90)  # exactly 30 complete 15M buckets
    client = TestClient(app)
    resp = client.get("/api/candles/15M")
    candles = resp.json()["candles"]
    assert 25 <= len(candles) <= 30  # allowing for boundary-alignment effects


def test_1h_aggregates_roughly_twelve_to_one():
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    live_engine.state.candle_history = []
    _populate_candles(live_engine, n=120)  # 10 hours of 5M candles
    client = TestClient(app)
    resp = client.get("/api/candles/1H")
    candles = resp.json()["candles"]
    assert 7 <= len(candles) <= 10


def test_1m_rejected_with_clear_error():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get("/api/candles/1M")
    body = resp.json()
    assert body["candles"] == []
    assert "error" in body


def test_invalid_timeframe_rejected_gracefully():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get("/api/candles/GARBAGE")
    assert resp.status_code == 200  # doesn't crash
    assert resp.json()["candles"] == []


def test_resampled_ohlc_values_are_correct():
    """Verify actual OHLC aggregation correctness, not just candle counts —
    high must be the MAX across the bucket, low the MIN, open the first,
    close the last."""
    from fastapi.testclient import TestClient
    from api.main import app, live_engine
    live_engine.state.candle_history = []
    base_ts = 1735689600
    # 3 candles forming exactly one 15M bucket (aligned to :00)
    live_engine.state.candle_history = [
        {"ts": base_ts, "open": 100, "high": 105, "low": 98, "close": 102, "volume": 10},
        {"ts": base_ts + 300, "open": 102, "high": 110, "low": 101, "close": 108, "volume": 20},
        {"ts": base_ts + 600, "open": 108, "high": 109, "low": 95, "close": 103, "volume": 15},
        # a 4th candle starts the NEXT bucket, needed so the first bucket
        # is marked complete (not the trailing in-progress one)
        {"ts": base_ts + 900, "open": 103, "high": 104, "low": 100, "close": 101, "volume": 5},
    ]
    client = TestClient(app)
    resp = client.get("/api/candles/15M")
    candles = resp.json()["candles"]
    assert len(candles) == 1
    c = candles[0]
    assert c["open"] == 100    # first candle's open
    assert c["high"] == 110    # max across all 3
    assert c["low"] == 95      # min across all 3
    assert c["close"] == 103   # last candle's close
    assert c["volume"] == 45   # sum


def test_dashboard_has_timeframe_buttons():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert "switchTimeframe" in html
    for tf in ["5M", "15M", "1H", "4H"]:
        assert f"data-tf=\"{tf}\"" in html


def test_dashboard_structurally_valid_with_timeframe_ui():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert html.count("<div") == html.count("</div>")
    assert html.count("<button") == html.count("</button>")


if __name__ == "__main__":
    tests = [
        test_5m_endpoint_returns_raw_candles,
        test_15m_aggregates_roughly_three_to_one,
        test_1h_aggregates_roughly_twelve_to_one,
        test_1m_rejected_with_clear_error,
        test_invalid_timeframe_rejected_gracefully,
        test_resampled_ohlc_values_are_correct,
        test_dashboard_has_timeframe_buttons,
        test_dashboard_structurally_valid_with_timeframe_ui,
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
