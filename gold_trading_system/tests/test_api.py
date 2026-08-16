import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# MUST be set before importing api.main — that module instantiates a global
# live_engine at import time, and without this, test runs silently write
# real trade-history file artifacts into the repo directory (a real bug
# found during final pre-launch review: test pollution risking mixing test
# data with genuine live trading history if tests ever ran alongside it).
os.environ["TRADE_HISTORY_PATH"] = ""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    print(f"Health: {resp.status_code} {resp.json()}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "PAPER", "Must default to PAPER mode, never LIVE"
    assert data["broker_connected"] == False, "Angel One not connected — honestly reported"


def test_snapshot_endpoint_returns_valid_shape():
    resp = client.get("/api/snapshot")
    print(f"Snapshot: {resp.status_code} {resp.json()}")
    assert resp.status_code == 200
    data = resp.json()
    required_fields = {"instrument", "ltp", "regime_trend", "last_structure_event",
                        "has_open_position", "open_position", "trading_disabled",
                        "trades_taken_today", "total_trades_this_session"}
    assert required_fields.issubset(data.keys())
    assert data["instrument"] == "GOLDM"
    assert isinstance(data["ltp"], (int, float))


def test_signal_endpoint_returns_valid_decision():
    resp = client.get("/api/signal")
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] in ("BUY", "SELL", "NO_TRADE")
    assert 0 <= data["long_score"] <= 100
    assert 0 <= data["short_score"] <= 100


def test_performance_endpoint():
    resp = client.get("/api/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["trading_disabled"] in (True, False)
    assert "equity_inr" in data
    assert "total_trades_this_session" in data


def test_trades_endpoint_returns_list():
    resp = client.get("/api/trades")
    assert resp.status_code == 200
    data = resp.json()
    assert "trades" in data
    assert isinstance(data["trades"], list)


def test_websocket_connects_and_streams():
    with client.websocket_connect("/ws/live") as ws:
        message = ws.receive_json()
        print(f"WebSocket message: {message}")
        required_fields = {"ts", "ltp", "regime_trend", "has_open_position",
                            "risk_state", "total_trades_this_session"}
        assert required_fields.issubset(message.keys())


def test_snapshot_calls_are_repeatable_no_crash():
    """Simulate the dashboard polling repeatedly — must not degrade or crash."""
    for _ in range(10):
        resp = client.get("/api/snapshot")
        assert resp.status_code == 200


def test_repeated_snapshot_calls_actually_advance_the_session():
    """
    THE critical test: this is what the old stateless endpoints failed at.
    Repeated calls must advance a single persistent trading session (tick
    count increases, potentially opening/closing real positions) rather than
    recomputing an independent one-off answer each time.
    """
    from api.main import live_engine
    tick_before = live_engine.state.tick_count
    for _ in range(20):
        client.get("/api/snapshot")
    tick_after = live_engine.state.tick_count
    print(f"Tick count before: {tick_before}, after: {tick_after}")
    assert tick_after > tick_before, \
        "Repeated /api/snapshot calls must advance the persistent live session"
    assert tick_after - tick_before == 20, \
        "Each call should advance exactly one tick of the SAME session"


def test_performance_reflects_the_same_session_as_snapshot():
    """/api/performance and /api/snapshot must be reading the same underlying
    session state, not two independent computations."""
    from api.main import live_engine
    for _ in range(5):
        client.get("/api/snapshot")
    perf = client.get("/api/performance").json()
    assert perf["total_trades_this_session"] == len(live_engine.state.trade_log), \
        "Performance endpoint must reflect the actual live_engine trade log"


if __name__ == "__main__":
    tests = [
        test_health_endpoint,
        test_snapshot_endpoint_returns_valid_shape,
        test_signal_endpoint_returns_valid_decision,
        test_performance_endpoint,
        test_trades_endpoint_returns_list,
        test_websocket_connects_and_streams,
        test_snapshot_calls_are_repeatable_no_crash,
        test_repeated_snapshot_calls_actually_advance_the_session,
        test_performance_reflects_the_same_session_as_snapshot,
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
