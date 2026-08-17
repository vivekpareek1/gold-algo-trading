"""
Tests for the TradingView reference link panel. An embedded chart widget
was tried first but proved unreliable (silently fell back to the wrong
symbol/AAPL instead of MCX GOLDM, root cause unclear even after ruling
out the obvious login-based hypothesis). Replaced with a simple, reliable
link-out to the user's own TradingView account in a new tab — guaranteed
correct since it's literally the same interface they already use.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_tradingview_link_present():
    html = client.get("/").text
    assert "tradingview.com/chart" in html
    assert "GOLDM1" in html


def test_link_opens_new_tab():
    """Must not navigate away from the dashboard — open in a new tab."""
    html = client.get("/").text
    assert 'target="_blank"' in html
    assert 'rel="noopener"' in html


def test_no_embedded_widget_script_remains():
    """The unreliable embedded widget must be fully removed, not left
    alongside the link (would be confusing and wastes a script load)."""
    html = client.get("/").text
    assert "embed-widget-advanced-chart.js" not in html


def test_own_chart_still_present():
    """The system's own chart (which reflects real trading decisions)
    must still be there, unaffected by this change."""
    html = client.get("/").text
    assert 'id="chart"' in html
    assert "addCandlestickSeries" in html


def test_page_structurally_valid():
    html = client.get("/").text
    assert html.count("<div") == html.count("</div>")
    assert html.count("<script") == html.count("</script>")
    assert html.count("<a ") == html.count("</a>")


if __name__ == "__main__":
    tests = [
        test_tradingview_link_present,
        test_link_opens_new_tab,
        test_no_embedded_widget_script_remains,
        test_own_chart_still_present,
        test_page_structurally_valid,
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
