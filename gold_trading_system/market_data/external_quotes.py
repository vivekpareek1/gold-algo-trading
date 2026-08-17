"""
External Market Reference Quotes — USD/INR, COMEX Gold, Dollar Index.

These are DISPLAY-ONLY context for the dashboard (helps judge whether MCX
GOLDM moves are being driven by international gold, USD strength, or
INR weakness) — they do NOT feed into any trading decision. The strategy
trades purely on Angel One's own MCX feed; this is a separate, independent
reference poller.

Source: Yahoo Finance's public (undocumented) chart endpoint. This is a
best-effort reference feed, not a licensed data source — treat outages or
stale values as informational gaps, never as a reason to alter trading
behavior (that's why this is entirely decoupled from LiveTradingEngine).

NOTE: this sandbox's network allowlist does not include finance.yahoo.com,
so the live HTTP call could not be end-to-end tested here — only the
parsing logic was verified against Yahoo's documented response structure.
Verify this actually connects once deployed (see /api/external_quotes).
"""
import time
import threading
from dataclasses import dataclass, field

try:
    import requests
except ImportError:
    requests = None


YAHOO_TICKERS = {
    "usd_inr": "INR=X",
    "comex_gold": "GC=F",
    "dollar_index": "DX-Y.NYB",
}

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class ExternalQuote:
    value: float | None = None
    prev_close: float | None = None
    last_updated_at: float | None = None
    last_error: str | None = None


@dataclass
class ExternalQuotesState:
    usd_inr: ExternalQuote = field(default_factory=ExternalQuote)
    comex_gold: ExternalQuote = field(default_factory=ExternalQuote)
    dollar_index: ExternalQuote = field(default_factory=ExternalQuote)


def parse_yahoo_chart_response(raw_json: dict) -> tuple[float | None, float | None]:
    """
    Extracts (current_price, previous_close) from a Yahoo chart API response.
    Returns (None, None) if the structure doesn't match what's expected —
    never raises, since a format change on Yahoo's end must degrade this
    optional reference feature gracefully, not affect anything else.
    """
    try:
        result = raw_json["chart"]["result"][0]
        meta = result["meta"]
        price = meta.get("regularMarketPrice")
        prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")
        if price is None:
            return None, None
        return float(price), (float(prev_close) if prev_close is not None else None)
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None


class ExternalQuotesPoller:
    """Background poller — fetches all three quotes every poll_interval_sec
    seconds. Each quote fetch is independent: one failing must not affect
    the others, and a failure never crashes the poll loop itself."""

    def __init__(self, poll_interval_sec: int = 60):
        self.poll_interval_sec = poll_interval_sec
        self.state = ExternalQuotesState()
        self._stop_requested = False

    def _fetch_one(self, ticker: str) -> ExternalQuote:
        quote = ExternalQuote()
        if requests is None:
            quote.last_error = "requests library not installed"
            return quote
        try:
            resp = requests.get(
                YAHOO_CHART_URL.format(ticker=ticker),
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            if resp.status_code != 200:
                quote.last_error = f"HTTP {resp.status_code}"
                return quote
            price, prev_close = parse_yahoo_chart_response(resp.json())
            if price is None:
                quote.last_error = "Could not parse price from response"
                return quote
            quote.value = price
            quote.prev_close = prev_close
            quote.last_updated_at = time.time()
        except Exception as e:
            quote.last_error = f"{type(e).__name__}: {e}"
        return quote

    def poll_once(self):
        for field_name, ticker in YAHOO_TICKERS.items():
            result = self._fetch_one(ticker)
            existing = getattr(self.state, field_name)
            if result.value is not None:
                # a successful fetch replaces the stored quote entirely
                setattr(self.state, field_name, result)
            else:
                # a failed fetch must not wipe out the last GOOD value —
                # keep showing the last known price, just record the error
                # so staleness is visible rather than the dashboard going blank
                existing.last_error = result.last_error
                setattr(self.state, field_name, existing)

    def run_forever(self):
        while not self._stop_requested:
            try:
                self.poll_once()
            except Exception as e:
                print(f"External quotes poll error: {type(e).__name__}: {e}")
            time.sleep(self.poll_interval_sec)

    def stop(self):
        self._stop_requested = True

    def start_background_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t
