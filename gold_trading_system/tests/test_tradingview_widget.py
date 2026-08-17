"""
Tests for the TradingView reference chart panel — an ADDITIONAL, informational
panel only. It must never be confused with or replace the system's own chart,
which reflects the exact data trading decisions are based on.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_tradingview_widget_embedded():
    html = client.get("/").text
    assert "embed-widget-advanced-chart.js" in html
    assert "MCX:GOLDM1!" in html


def test_tradingview_panel_has_disclaimer():
    """Must be clearly labeled as a separate, general reference — never
    presented as reflecting this system's own trading data."""
    html = client.get("/").text
    assert "Independent reference only" in html
    assert "not" in html and "next-month contract" in html


def test_own_chart_still_present_alongside_tradingview():
    """The TradingView widget must be ADDITIONAL — the system's own chart
    (which reflects real trading decisions) must still be there."""
    html = client.get("/").text
    assert 'id="chart"' in html
    assert "addCandlestickSeries" in html
    assert "loadCandleHistory" in html


def test_page_structurally_valid_with_widget_added():
    html = client.get("/").text
    assert html.count("<div") == html.count("</div>")
    assert html.count("<script") == html.count("</script>")


if __name__ == "__main__":
    tests = [
        test_tradingview_widget_embedded,
        test_tradingview_panel_has_disclaimer,
        test_own_chart_still_present_alongside_tradingview,
        test_page_structurally_valid_with_widget_added,
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
