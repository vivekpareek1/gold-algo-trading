"""
Tests for the XAU/USD (international spot gold) embedded TradingView
widget. Unlike the earlier MCX:GOLDM1! embed attempt (which showed the
wrong symbol, likely due to login-gated MCX data), XAU/USD is a freely
available symbol requiring no login — should be more reliable, but this
is monitored, not assumed.
"""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_xau_widget_present():
    html = client.get("/").text
    assert "OANDA:XAUUSD" in html
    assert "embed-widget-advanced-chart.js" in html


def test_xau_widget_json_is_valid():
    """The exact class of bug that would silently break the widget —
    malformed JSON in the config block."""
    content = open(os.path.join(os.path.dirname(__file__), "..", "api", "main.py")).read()
    idx = content.index("OANDA:XAUUSD")
    start = content.rindex("{", 0, idx)
    depth = 0
    end = start
    for i, ch in enumerate(content[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    block = content[start:end]
    parsed = json.loads(block)   # raises if invalid
    assert parsed["symbol"] == "OANDA:XAUUSD"
    assert parsed["locale"] == "en"   # a valid, standard locale code


def test_xau_panel_has_disclaimer_distinguishing_from_mcx():
    """Must be clearly labeled as NOT the same as MCX GOLDM — different
    currency, exchange, and no duty/carry-cost — to avoid the person
    confusing this with the actual traded instrument."""
    html = client.get("/").text
    assert "NOT the same as MCX GOLDM" in html


def test_both_own_chart_and_tradingview_link_still_present():
    """The XAU widget is additional — must not replace the system's own
    chart or the GOLDM TradingView link-out."""
    html = client.get("/").text
    assert 'id="chart"' in html
    assert "Open GOLDM chart on TradingView" in html


def test_page_structurally_valid_with_xau_widget():
    html = client.get("/").text
    assert html.count("<div") == html.count("</div>")
    assert html.count("<script") == html.count("</script>")


if __name__ == "__main__":
    tests = [
        test_xau_widget_present,
        test_xau_widget_json_is_valid,
        test_xau_panel_has_disclaimer_distinguishing_from_mcx,
        test_both_own_chart_and_tradingview_link_still_present,
        test_page_structurally_valid_with_xau_widget,
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
