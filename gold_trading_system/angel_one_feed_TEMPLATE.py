"""
Angel One Live Feed Handler — runs on your AWS server, NOT in this sandbox
(this file cannot be run or tested here; smartapi.angelone.in is unreachable
from this environment by network policy, verified in an earlier session).

Connects Angel One's SmartWebSocketV2 for live GOLDM ticks, aggregates them
into 5-minute candles via TickAggregator (tested, deterministic), and feeds
each completed candle into LiveTradingEngine.on_tick() — the same tested
engine used throughout backtesting.

SETUP (on your AWS server):
    pip install smartapi-python pyotp --break-system-packages
    (fastapi, uvicorn should already be installed alongside the API layer)

RUN:
    python3 angel_one_live_feed.py

This is designed to run ALONGSIDE the FastAPI app (api/main.py) — start
both, and the dashboard's /api/snapshot and /ws/live will reflect real
market data instead of the simulated feed, because they read from the
same live_engine instance this script feeds.

IMPORTANT: regenerate your API key, secret, and TOTP secret before using
this — any credentials that were ever typed into a chat session must be
treated as compromised, per the earlier security note.
"""
import time
import threading
from datetime import datetime, timezone

from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp

from market_data.tick_aggregator import TickAggregator, Tick
from market_data.contract_selector import select_next_month_contract
from execution.live_trading_engine import LiveTradingEngine, LiveTick

# ============ CREDENTIALS — fill these in ============
API_KEY = "YOUR_API_KEY_HERE"
CLIENT_CODE = "YOUR_CLIENT_CODE_HERE"
PIN = "YOUR_PIN_HERE"
TOTP_SECRET = "YOUR_TOTP_SECRET_HERE"
# =======================================================

EXCHANGE_TYPE = 5   # MCX, per Angel One's WebSocket exchange type codes
# CRITICAL: subscription mode MUST carry volume.
#   mode 1 = LTP        -> last_traded_price ONLY, NO volume field
#   mode 2 = Quote      -> includes volume_trade_for_the_day  <-- required
#   mode 3 = Snap Quote -> Quote + market depth (more bandwidth than needed)
# This was originally set to 1, which silently fed volume=0 into every
# candle. Verified impact on real 2-year MCX data: the strategy collapses
# from 1921 trades / +0.262R / PF 1.69 to 69 trades / -0.378R / PF 0.37,
# because every volume-confirmation check in the signal engine and the
# momentum-health check degrade to a constant. Do not lower this to 1.
SUBSCRIPTION_MODE = 2
SYMBOL_TOKEN = None   # resolved in AngelOneLiveFeed.connect()
CANDLE_INTERVAL_MINUTES = 5


class AngelOneLiveFeed:
    def __init__(self, live_engine: LiveTradingEngine, on_candle_callback=None):
        self.live_engine = live_engine
        self.aggregator = TickAggregator(interval_minutes=CANDLE_INTERVAL_MINUTES)
        self.on_candle_callback = on_candle_callback  # optional: notify dashboard/DB
        self.smart_api = None
        self.ws = None
        self._auth_token = None
        self._feed_token = None
        self._volume_warning_shown = False
        self._stop_requested = False
        self._reconnect_count = 0

    def connect(self):
        self.smart_api = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_SECRET).now()
        session = self.smart_api.generateSession(CLIENT_CODE, PIN, totp)
        if not session.get("status"):
            raise RuntimeError(f"Login failed: {session}")

        self._auth_token = session["data"]["jwtToken"]
        self._feed_token = self.smart_api.getfeedToken()
        print("Logged in, feed token acquired.")

        global SYMBOL_TOKEN
        if SYMBOL_TOKEN is None:
            result = self.smart_api.searchScrip(exchange="MCX", searchscrip="GOLDM")
            if not result.get("status") or not result.get("data"):
                raise RuntimeError(f"Could not fetch GOLDM contracts: {result}")
            contract = select_next_month_contract(result["data"])
            SYMBOL_TOKEN = contract.symboltoken
            print(f"Selected next-month contract: {contract.tradingsymbol} "
                  f"(token {contract.symboltoken}, expires {contract.expiry_date.date()})")

        self.ws = SmartWebSocketV2(self._auth_token, API_KEY, CLIENT_CODE, self._feed_token)
        self.ws.on_open = self._on_open
        self.ws.on_data = self._on_data
        self.ws.on_error = self._on_error
        self.ws.on_close = self._on_close

    def _on_open(self, wsapp):
        print("WebSocket opened — subscribing to GOLDM...")
        # LOCKED DESIGN DECISION: SYMBOL_TOKEN is the NEXT-month contract,
        # resolved at connect() via select_next_month_contract() — never the
        # nearest expiry (declining liquidity + rollover/delivery risk), and
        # never hardcoded, so it stays correct as contracts roll over.
        token_list = [{"exchangeType": EXCHANGE_TYPE, "tokens": [SYMBOL_TOKEN]}]
        self.ws.subscribe("goldm_feed", SUBSCRIPTION_MODE, token_list)

    def _on_data(self, wsapp, message):
        try:
            ltp = float(message.get("last_traded_price", 0)) / 100.0  # Angel One sends paise
            volume = float(message.get("volume_trade_for_the_day", 0))
            ts = int(message.get("exchange_timestamp", time.time() * 1000)) // 1000

            tick = Tick(ts=ts, ltp=ltp, volume=volume)

            # Push the raw LTP on EVERY tick, not just on candle close. The
            # engine's full pipeline still only runs per completed candle,
            # but without this the dashboard's "live" price was stale by up
            # to a full 5-minute bar between candle closes.
            self.live_engine.update_live_price(ltp=ltp, ts=ts)

            completed = self.aggregator.add_tick(tick)

            # Loud, once-only warning if the feed is delivering no volume at
            # all — that silently guts every volume-confirmation check in the
            # strategy rather than failing visibly.
            if self.aggregator.volume_feed_looks_broken and not self._volume_warning_shown:
                self._volume_warning_shown = True
                print("\n*** WARNING: no volume in any tick so far. The feed is "
                      "probably subscribed in LTP mode (1) instead of Quote mode (2). "
                      "Volume-confirmation checks will be meaningless and the strategy "
                      "will behave very differently from its backtest. ***\n")

            if completed is not None:
                live_tick = LiveTick(ts=completed.ts, open=completed.open, high=completed.high,
                                       low=completed.low, close=completed.close,
                                       volume=completed.volume)
                snapshot = self.live_engine.on_tick(live_tick)
                print(f"[{datetime.fromtimestamp(completed.ts, tz=timezone.utc)}] "
                      f"Candle closed: O={completed.open} H={completed.high} "
                      f"L={completed.low} C={completed.close} V={completed.volume} | "
                      f"regime={snapshot['regime_trend']} "
                      f"open_position={snapshot['has_open_position']}")
                if self.on_candle_callback:
                    self.on_candle_callback(completed, snapshot)
        except Exception as e:
            print(f"Error processing tick: {type(e).__name__}: {e}")
            # a malformed tick must not crash the feed — log and continue

    def _on_error(self, wsapp, error):
        print(f"WebSocket error: {error}")

    def _on_close(self, wsapp):
        print("WebSocket closed.")

    def run_forever(self, max_backoff_sec: int = 300, rate_limit_cooldown_sec: int = 300):
        """
        Blocking call — run this in a dedicated thread if alongside FastAPI.

        Reconnects automatically. Without this, a single network blip ended
        the feed permanently: ws.connect() would return, the thread would
        exit, and nothing anywhere would notice — the dashboard kept showing
        a green LIVE badge over frozen prices. For a system meant to run
        unattended for weeks, a dead feed must never look like a quiet market.

        BUGFIX (real incident): a rate-limit ban ("Access denied because of
        exceeding access rate") took ~3.5 hours to clear while backoff was
        capped at 60s — meaning hundreds of retries hammered the same
        rate-limited endpoint the whole time. Many APIs count repeated
        failed attempts within their rate-limit window, so aggressive
        retrying can itself PROLONG a ban rather than waiting it out. Now:
        max_backoff_sec raised to 5 minutes, AND a rate-limit-specific error
        jumps straight to a full cooldown rather than the normal doubling
        progression — gentler on the API when it's explicitly telling us
        to back off.
        """
        backoff = 1
        while not self._stop_requested:
            try:
                self.connect()
                self._reconnect_count += 1 if self._reconnect_count or backoff > 1 else 0
                backoff = 1   # a successful connect resets the backoff
                self.ws.connect()   # blocks until the socket closes
            except Exception as e:
                error_text = f"{type(e).__name__}: {e}"
                print(f"Feed connection error: {error_text}")
                if "exceeding access rate" in error_text.lower() or "access denied" in error_text.lower():
                    backoff = rate_limit_cooldown_sec
                    print(f"Rate-limit error detected — jumping straight to a "
                          f"{rate_limit_cooldown_sec}s cooldown instead of the normal "
                          f"backoff progression, to avoid extending the ban.")

            if self._stop_requested:
                break

            print(f"Feed disconnected — reconnecting in {backoff}s "
                  f"(reconnect #{self._reconnect_count + 1})...")
            time.sleep(backoff)
            self._reconnect_count += 1
            backoff = min(backoff * 2, max_backoff_sec)

    def stop(self):
        """Request a clean shutdown of the reconnect loop."""
        self._stop_requested = True
        try:
            if self.ws:
                self.ws.close_connection()
        except Exception:
            pass


def run_in_background_thread(live_engine: LiveTradingEngine) -> AngelOneLiveFeed:
    """Call this from api/main.py's startup to run the feed alongside uvicorn
    without blocking the API server."""
    feed = AngelOneLiveFeed(live_engine)
    thread = threading.Thread(target=feed.run_forever, daemon=True)
    thread.start()
    return feed


if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Fill in your credentials at the top of this script first.")
    else:
        from config.settings import Settings
        from execution.broker_adapters.paper_provider import PaperBrokerProvider

        settings = Settings()
        broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
        broker.connect()
        engine = LiveTradingEngine(settings, broker, symbol="GOLDM")

        feed = AngelOneLiveFeed(engine)
        print("Connecting to Angel One live feed for GOLDM...")
        feed.run_forever()
