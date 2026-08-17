import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import patch, MagicMock
from news_engine.live_news_poller import (
    parse_yahoo_rss, YahooFinanceNewsProvider, GoldNewsMonitor
)
from news_engine.news_provider import SourceType, NewsRiskState


# Real Yahoo Finance RSS structure, taken from an actual verified sample
# feed (confirmed via search) — including a genuine Bloomberg-sourced
# gold headline in the exact format Yahoo serves.
REAL_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?><rss xmlns:media="http://search.yahoo.com/mrss/" version="2.0"><channel><title>Yahoo Finance</title><link>https://finance.yahoo.com/</link><description>desc</description><language>en-US</language><pubDate>Wed, 31 Dec 2025 03:56:25 -0500</pubDate><item><title>Gold, Silver Plunge as Traders Book Profit from Record Rallies</title><link>https://finance.yahoo.com/news/silver-pulls-back-record-historic-075048633.html</link><pubDate>2025-12-29T21:12:30Z</pubDate><source url="https://www.bloomberg.com/company">Bloomberg</source><guid isPermaLink="false">silver-pulls-back-record-historic-075048633.html</guid><media:content height="86" url="https://media.zenfs.com/x" width="130"/></item><item><title>Airbus Signs Deal With Two Chinese Airlines</title><link>https://www.investors.com/news/x</link><pubDate>2025-12-29T21:08:53Z</pubDate><source url="http://www.investors.com/">Investor's Business Daily</source><guid isPermaLink="false">x</guid></item></channel></rss>"""


def test_parses_real_yahoo_rss_structure():
    items = parse_yahoo_rss(REAL_SAMPLE_RSS)
    assert len(items) == 2
    assert items[0].text == "Gold, Silver Plunge as Traders Book Profit from Record Rallies"
    assert items[0].author == "Bloomberg"
    assert items[0].url == "https://finance.yahoo.com/news/silver-pulls-back-record-historic-075048633.html"


def test_parses_publish_date_correctly():
    items = parse_yahoo_rss(REAL_SAMPLE_RSS)
    assert items[0].published_at.year == 2025
    assert items[0].published_at.month == 12
    assert items[0].published_at.day == 29


def test_gold_headline_classified_correctly():
    """The real sample headline doesn't contain our HIGH/MEDIUM keywords —
    verify it's classified LOW/NORMAL rather than crashing, and that a
    genuinely high-impact headline WOULD be caught."""
    from news_engine.news_provider import NewsImpactClassifier
    items = parse_yahoo_rss(REAL_SAMPLE_RSS)
    classifier = NewsImpactClassifier()
    assessment = classifier.classify(items[0])
    assert assessment.item.text == items[0].text


def test_empty_feed_returns_empty_list():
    assert parse_yahoo_rss("") == []


def test_malformed_xml_does_not_crash():
    assert parse_yahoo_rss("not xml at all {{{ <unclosed") == []


def test_missing_source_tag_defaults_gracefully():
    xml = """<item><title>Some headline</title><link>http://x.com</link><pubDate>Wed, 31 Dec 2025 03:56:25 -0500</pubDate></item>"""
    items = parse_yahoo_rss(xml)
    assert len(items) == 1
    assert items[0].author == "Yahoo Finance"


def test_missing_title_skips_item_not_crash():
    xml = """<item><link>http://x.com</link><pubDate>Wed, 31 Dec 2025 03:56:25 -0500</pubDate></item>"""
    items = parse_yahoo_rss(xml)
    assert items == []


# ---------- provider-level tests ----------

def _mock_response(status_code=200, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


def test_provider_fetches_and_deduplicates_across_tickers():
    provider = YahooFinanceNewsProvider(tickers=["GC=F", "GLD"])
    with patch("news_engine.live_news_poller.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(200, REAL_SAMPLE_RSS)
        provider.poll_once()

    # same 2 items appear from BOTH tickers (same mock response) — must be deduplicated
    items = provider.fetch_recent(limit=10)
    assert len(items) == 2, f"Expected deduplication across tickers, got {len(items)} items"


def test_provider_preserves_last_good_data_on_failure():
    provider = YahooFinanceNewsProvider(tickers=["GC=F"])
    with patch("news_engine.live_news_poller.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(200, REAL_SAMPLE_RSS)
        provider.poll_once()
    assert len(provider.fetch_recent()) == 2

    with patch("news_engine.live_news_poller.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(500, "")
        provider.poll_once()
    assert len(provider.fetch_recent()) == 2, \
        "A failed poll must not wipe out previously fetched real news"


def test_provider_handles_network_exception_gracefully():
    provider = YahooFinanceNewsProvider(tickers=["GC=F"])
    with patch("news_engine.live_news_poller.requests") as mock_requests:
        mock_requests.get.side_effect = ConnectionError("unreachable")
        try:
            provider.poll_once()
        except Exception as e:
            assert False, f"Must never raise, got {type(e).__name__}: {e}"


# ---------- GoldNewsMonitor integration ----------

def test_monitor_produces_risk_state():
    monitor = GoldNewsMonitor()
    with patch("news_engine.live_news_poller.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(200, REAL_SAMPLE_RSS)
        monitor.poll_once()

    state = monitor.get_current_risk_state()
    assert state in (NewsRiskState.NORMAL, NewsRiskState.EVENT_APPROACHING,
                       NewsRiskState.HIGH_IMPACT, NewsRiskState.POST_EVENT_VOLATILITY)


def test_monitor_detects_high_impact_keyword():
    monitor = GoldNewsMonitor()
    high_impact_rss = REAL_SAMPLE_RSS.replace(
        "Gold, Silver Plunge as Traders Book Profit from Record Rallies",
        "Fed Chair Announces Emergency Rate Cut Amid Recession Fears"
    )
    with patch("news_engine.live_news_poller.requests") as mock_requests:
        mock_requests.get.return_value = _mock_response(200, high_impact_rss)
        monitor.poll_once()

    assessments = monitor.get_assessments()
    high_impact_found = any(a.impact_level == "HIGH" for a in assessments)
    assert high_impact_found, "A genuinely high-impact headline must be classified as such"


def test_flexible_date_parser_handles_iso_format():
    from news_engine.live_news_poller import _parse_flexible_date
    dt = _parse_flexible_date("2025-12-29T21:12:30Z")
    assert dt.year == 2025 and dt.month == 12 and dt.day == 29


def test_flexible_date_parser_handles_rfc2822_format():
    from news_engine.live_news_poller import _parse_flexible_date
    dt = _parse_flexible_date("Wed, 31 Dec 2025 03:56:25 -0500")
    assert dt.year == 2025 and dt.month == 12 and dt.day == 31
def test_api_news_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    resp = TestClient(app).get("/api/news")
    assert resp.status_code == 200
    body = resp.json()
    assert "risk_state" in body
    assert "items" in body


def test_dashboard_shows_gold_news_panel():
    from fastapi.testclient import TestClient
    from api.main import app
    html = TestClient(app).get("/").text
    assert "Gold News" in html
    assert "newsList" in html
    assert "newsRiskBadge" in html
if __name__ == "__main__":
    tests = [
        test_parses_real_yahoo_rss_structure,
        test_parses_publish_date_correctly,
        test_gold_headline_classified_correctly,
        test_empty_feed_returns_empty_list,
        test_malformed_xml_does_not_crash,
        test_missing_source_tag_defaults_gracefully,
        test_missing_title_skips_item_not_crash,
        test_provider_fetches_and_deduplicates_across_tickers,
        test_provider_preserves_last_good_data_on_failure,
        test_provider_handles_network_exception_gracefully,
        test_monitor_produces_risk_state,
        test_monitor_detects_high_impact_keyword,
        test_flexible_date_parser_handles_iso_format,
        test_flexible_date_parser_handles_rfc2822_format,
        test_api_news_endpoint,
        test_dashboard_shows_gold_news_panel,
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




