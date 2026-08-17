import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import patch, MagicMock
from market_data.external_quotes import (
    parse_yahoo_chart_response, ExternalQuotesPoller, ExternalQuote
)


# ---------- parsing ----------

def test_parses_valid_response():
    raw = {
        "chart": {
            "result": [{
                "meta": {"regularMarketPrice": 87.45, "previousClose": 87.30}
            }]
        }
    }
    price, prev = parse_yahoo_chart_response(raw)
    assert price == 87.45
    assert prev == 87.30


def test_parses_response_with_chartPreviousClose_fallback():
    raw = {"chart": {"result": [{"meta": {"regularMarketPrice": 2650.5,
                                             "chartPreviousClose": 2645.0}}]}}
    price, prev = parse_yahoo_chart_response(raw)
    assert price == 2650.5
    assert prev == 2645.0


def test_missing_price_returns_none_not_crash():
    raw = {"chart": {"result": [{"meta": {}}]}}
    price, prev = parse_yahoo_chart_response(raw)
    assert price is None
    assert prev is None


def test_malformed_response_returns_none_not_crash():
    for bad in [{}, {"chart": {}}, {"chart": {"result": []}}, None, "not even a dict", 12345]:
        price, prev = parse_yahoo_chart_response(bad if bad is not None else {})
        assert price is None


def test_empty_result_list_handled():
    raw = {"chart": {"result": []}}
    price, prev = parse_yahoo_chart_response(raw)
    assert price is None


# ---------- poller behavior ----------

def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def test_successful_fetch_updates_state():
    poller = ExternalQuotesPoller()
    good_response = {"chart": {"result": [{"meta": {"regularMarketPrice": 87.5,
                                                        "previousClose": 87.2}}]}}
    with patch("market_data.external_quotes.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(200, good_response)
        poller.poll_once()

    assert poller.state.usd_inr.value == 87.5
    assert poller.state.usd_inr.last_error is None


def test_failed_fetch_preserves_last_good_value():
    """
    THE critical behavior: a transient failure must not wipe out the last
    known good price and blank the dashboard — it should keep showing the
    last real value while flagging that it's now stale.
    """
    poller = ExternalQuotesPoller()
    good_response = {"chart": {"result": [{"meta": {"regularMarketPrice": 87.5,
                                                        "previousClose": 87.2}}]}}
    with patch("market_data.external_quotes.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(200, good_response)
        poller.poll_once()
    assert poller.state.usd_inr.value == 87.5

    # now simulate every fetch failing
    with patch("market_data.external_quotes.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(500, {})
        poller.poll_once()

    assert poller.state.usd_inr.value == 87.5, \
        "A failed poll must NOT erase the last good value"
    assert poller.state.usd_inr.last_error is not None, \
        "But the error must be recorded so staleness is visible"


def test_one_ticker_failing_does_not_affect_others():
    """Each of the three quotes is fetched independently — one failing
    (e.g. Yahoo changes DXY's ticker format) must not take down the others."""
    poller = ExternalQuotesPoller()
    call_count = {"n": 0}

    def side_effect(url, headers=None, timeout=None):
        call_count["n"] += 1
        if "DX-Y.NYB" in url:
            return _mock_response(404, {})   # this one fails
        return _mock_response(200, {"chart": {"result": [{"meta": {
            "regularMarketPrice": 100.0, "previousClose": 99.0}}]}})

    with patch("market_data.external_quotes.requests") as mock_requests:
        mock_requests.get.side_effect = side_effect
        poller.poll_once()

    assert poller.state.usd_inr.value == 100.0
    assert poller.state.comex_gold.value == 100.0
    assert poller.state.dollar_index.value is None
    assert poller.state.dollar_index.last_error is not None


def test_network_exception_does_not_crash_poll_once():
    poller = ExternalQuotesPoller()
    with patch("market_data.external_quotes.requests") as mock_requests:
        mock_requests.get.side_effect = ConnectionError("network unreachable")
        try:
            poller.poll_once()
        except Exception as e:
            assert False, f"poll_once() must never raise, got {type(e).__name__}: {e}"
    assert poller.state.usd_inr.last_error is not None


def test_missing_requests_library_handled_gracefully():
    """If the requests library somehow isn't installed, must degrade
    gracefully rather than crashing the whole dashboard on import."""
    with patch("market_data.external_quotes.requests", None):
        poller = ExternalQuotesPoller()
        poller.poll_once()
    assert poller.state.usd_inr.last_error is not None
    assert poller.state.usd_inr.value is None


def test_correct_tickers_requested():
    poller = ExternalQuotesPoller()
    requested_urls = []

    def side_effect(url, headers=None, timeout=None):
        requested_urls.append(url)
        return _mock_response(200, {"chart": {"result": [{"meta": {
            "regularMarketPrice": 1.0, "previousClose": 1.0}}]}})

    with patch("market_data.external_quotes.requests") as mock_requests:
        mock_requests.get.side_effect = side_effect
        poller.poll_once()

    assert any("INR=X" in u for u in requested_urls)
    assert any("GC=F" in u for u in requested_urls)
    assert any("DX-Y.NYB" in u for u in requested_urls)


def test_api_endpoint_returns_all_three_quotes():
    from fastapi.testclient import TestClient
    from api.main import app
    resp = TestClient(app).get("/api/external_quotes")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"usd_inr", "comex_gold", "dollar_index"}
    for key in body:
        assert "value" in body[key]
        assert "stale" in body[key]


def test_dashboard_shows_market_context_strip():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert "Market Context" in html
    assert "refUsdInr" in html
    assert "refComexGold" in html
    assert "refDxy" in html
if __name__ == "__main__":
    tests = [
        test_parses_valid_response,
        test_parses_response_with_chartPreviousClose_fallback,
        test_missing_price_returns_none_not_crash,
        test_malformed_response_returns_none_not_crash,
        test_empty_result_list_handled,
        test_successful_fetch_updates_state,
        test_failed_fetch_preserves_last_good_value,
        test_one_ticker_failing_does_not_affect_others,
        test_network_exception_does_not_crash_poll_once,
        test_missing_requests_library_handled_gracefully,
        test_correct_tickers_requested,
        test_api_endpoint_returns_all_three_quotes,
        test_dashboard_shows_market_context_strip,
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


