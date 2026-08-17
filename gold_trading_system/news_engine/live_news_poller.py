"""
Live Gold News Poller — real headlines via Yahoo Finance RSS (which
carries Bloomberg-sourced financial news), classified for gold-market
relevance and impact using the existing NewsImpactClassifier.

Same abstraction pattern as the broker/data layer: this is one concrete
NewsSourceProvider implementation; the interface (news_engine/news_provider.py)
was already built to support this without touching downstream code.

DISPLAY-ONLY for now — like the market-context quotes (USD/INR, COMEX
Gold, DXY), this does NOT feed into any trading decision by itself. The
classifier's risk-state output (NORMAL / EVENT_APPROACHING / HIGH_IMPACT)
is computed and exposed, but wiring it into the risk engine's actual
trade-gating logic is a deliberate separate step, not bundled in here —
that decision (whether news should be allowed to block trades) deserves
its own explicit review, not a silent side effect of adding a news feed.

NOTE: this sandbox's network allowlist does not include finance.yahoo.com,
so the live HTTP call could not be end-to-end tested here — only the RSS
parsing logic was verified against Yahoo's real, documented feed
structure (confirmed via search, including a real sample showing
Bloomberg-sourced gold headlines in exactly this format). Verify this
actually connects once deployed (see /api/news).
"""
import re
import time
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

try:
    import requests
except ImportError:
    requests = None

from news_engine.news_provider import (
    NewsSourceProvider, NewsItem, SourceType, NewsImpactClassifier, NewsRiskState
)


YAHOO_RSS_URL = "https://finance.yahoo.com/rss/headline?s={ticker}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _parse_flexible_date(date_str: str) -> datetime:
    """
    Yahoo's RSS feed has been observed using BOTH RFC 2822 dates (the
    formal RSS 2.0 spec format, e.g. channel-level pubDate) and ISO 8601
    (e.g. item-level pubDate: "2025-12-29T21:12:30Z") within the SAME
    feed. Tries ISO 8601 first (Yahoo's more common item-level format),
    falls back to RFC 2822, and only then gives up.
    """
    try:
        # handle the "Z" suffix explicitly — fromisoformat before Python 3.11
        # doesn't accept it directly
        iso_str = date_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    dt = parsedate_to_datetime(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

# Multiple tickers whose headlines plausibly move gold: the futures
# contract itself, the ETF, and silver (historically correlated).
GOLD_RELATED_TICKERS = ["GC=F", "GLD", "SI=F"]


def parse_yahoo_rss(xml_text: str) -> list[NewsItem]:
    """
    Minimal, dependency-free RSS 2.0 parser for Yahoo Finance's feed
    format. Deliberately not using a full XML library's namespace-strict
    parsing here, since Yahoo's feed mixes a media: namespace that isn't
    needed for our purposes — regex-based extraction of the handful of
    fields we actually need (title, link, pubDate, source) is more
    robust to minor formatting quirks than a strict parser that could
    reject the whole feed over one malformed tag.

    Returns an empty list (never raises) if the format doesn't match —
    a parsing failure must degrade this optional feature gracefully.
    """
    items = []
    try:
        item_blocks = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)
        for block in item_blocks:
            title_match = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
            link_match = re.search(r"<link>(.*?)</link>", block, re.DOTALL)
            pubdate_match = re.search(r"<pubDate>(.*?)</pubDate>", block, re.DOTALL)
            source_match = re.search(r'<source url="[^"]*">(.*?)</source>', block, re.DOTALL)

            if not title_match or not pubdate_match:
                continue

            title = title_match.group(1).strip()
            link = link_match.group(1).strip() if link_match else ""
            source_name = source_match.group(1).strip() if source_match else "Yahoo Finance"

            try:
                pubdate_str = pubdate_match.group(1).strip()
                published_at = _parse_flexible_date(pubdate_str)
            except (ValueError, TypeError):
                published_at = datetime.now(timezone.utc)

            items.append(NewsItem(
                source=SourceType.BLOOMBERG,  # Yahoo RSS aggregates Bloomberg + others;
                                                 # source_name captures the real attribution
                author=source_name, text=title, published_at=published_at, url=link,
            ))
    except Exception:
        return []
    return items


class YahooFinanceNewsProvider(NewsSourceProvider):
    """Real NewsSourceProvider implementation — fetches from Yahoo Finance
    RSS for each configured ticker, deduplicates, and returns the most
    recent items across all of them."""

    def __init__(self, tickers: list[str] = None):
        self.tickers = tickers or GOLD_RELATED_TICKERS
        self._cache: list[NewsItem] = []
        self._last_fetch_error: str | None = None

    def _fetch_ticker(self, ticker: str) -> list[NewsItem]:
        if requests is None:
            self._last_fetch_error = "requests library not installed"
            return []
        try:
            resp = requests.get(YAHOO_RSS_URL.format(ticker=ticker),
                                   headers={"User-Agent": USER_AGENT}, timeout=10)
            if resp.status_code != 200:
                self._last_fetch_error = f"HTTP {resp.status_code} for {ticker}"
                return []
            return parse_yahoo_rss(resp.text)
        except Exception as e:
            self._last_fetch_error = f"{type(e).__name__}: {e}"
            return []

    def poll_once(self):
        all_items = []
        seen_titles = set()
        for ticker in self.tickers:
            for item in self._fetch_ticker(ticker):
                if item.text not in seen_titles:
                    seen_titles.add(item.text)
                    all_items.append(item)
        if all_items:
            all_items.sort(key=lambda i: i.published_at, reverse=True)
            self._cache = all_items[:30]
            self._last_fetch_error = None
        # a failed poll (empty result) intentionally leaves _cache untouched —
        # same "don't wipe the last good data" principle as external_quotes

    def fetch_recent(self, limit: int = 20) -> list[NewsItem]:
        return self._cache[:limit]


class GoldNewsMonitor:
    """Background poller wrapping YahooFinanceNewsProvider + NewsImpactClassifier
    — periodically refreshes and classifies, exposes the current state for
    the API/dashboard to read."""

    def __init__(self, poll_interval_sec: int = 300):
        self.poll_interval_sec = poll_interval_sec
        self.provider = YahooFinanceNewsProvider()
        self.classifier = NewsImpactClassifier()
        self._stop_requested = False

    def poll_once(self):
        self.provider.poll_once()

    def get_assessments(self, limit: int = 10):
        items = self.provider.fetch_recent(limit=limit)
        return [self.classifier.classify(item) for item in items]

    def get_current_risk_state(self) -> NewsRiskState:
        assessments = self.get_assessments(limit=10)
        return self.classifier.aggregate_risk_state(assessments)

    def run_forever(self):
        while not self._stop_requested:
            try:
                self.poll_once()
            except Exception as e:
                print(f"Gold news poll error: {type(e).__name__}: {e}")
            time.sleep(self.poll_interval_sec)

    def stop(self):
        self._stop_requested = True

    def start_background_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t
