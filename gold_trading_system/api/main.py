"""
API layer — connects the backend engines to the dashboard.

Automatically uses the REAL Angel One live feed if angel_one_feed.py is
present in this same folder (it lives only on the server, gitignored,
holds credentials — never committed). Falls back to a simulated feed
otherwise, so this file works out of the box for anyone cloning the repo
without needing real credentials just to explore the dashboard.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
import random
import time
import threading
import os
from datetime import datetime, timezone, timedelta

from config.settings import Settings
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from market_data.external_quotes import ExternalQuotesPoller
from news_engine.live_news_poller import GoldNewsMonitor

app = FastAPI(title="Gold Trading System API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- shared, persistent live trading session (single instrument, v1) ----------
settings = Settings()
broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
broker.connect()
# Persistence path is configurable so test imports of this module (which
# instantiate this SAME global live_engine at import time) never write to
# the real trade history file — see tests/test_api.py, which sets this env
# var to a throwaway temp path before importing. An empty string disables
# persistence entirely (used by tests that want zero disk side effects).
_persistence_env = os.environ.get("TRADE_HISTORY_PATH", "trade_history.jsonl")
_persistence_path = _persistence_env if _persistence_env else None
_candle_persistence_env = os.environ.get("CANDLE_HISTORY_PATH", "candle_history.jsonl")
_candle_persistence_path = _candle_persistence_env if _candle_persistence_env else None
_open_position_env = os.environ.get("OPEN_POSITION_PATH", "open_position.json")
_open_position_path = _open_position_env if _open_position_env else None
live_engine = LiveTradingEngine(settings, broker, symbol="GOLDM", persistence_path=_persistence_path,
                                   candle_persistence_path=_candle_persistence_path,
                                   open_position_path=_open_position_path)

# BUGFIX (process/deploy issue, not code logic): repeated confusion over
# whether a deploy actually took effect — the dashboard would show stale
# behavior with no visible way to confirm from a screenshot whether new
# code was actually running. This string changes with every deploy, shown
# prominently in the footer, so it is now immediately, unambiguously
# checkable from a screenshot rather than inferred from subtle UI details.
BUILD_VERSION = "2026-08-18-fed-treasury-v17"

_last_price = 63000.0
_tick_count = 0

# Try to load the real Angel One feed handler. This file is gitignored and
# only ever exists on the deployment server, with real credentials filled
# in — importing it here never exposes anything to the git-tracked repo.
LIVE_FEED_ACTIVE = False

# External reference quotes — independent of the trading feed, polled every
# 60 seconds. Display-only.
external_quotes_poller = ExternalQuotesPoller(poll_interval_sec=60)

# Gold news monitor — display-only context, same pattern as external
# quotes above. Polled less frequently (news doesn't need 60s freshness
# like a price feed) to be gentle on Yahoo's free RSS endpoint.
gold_news_monitor = GoldNewsMonitor(poll_interval_sec=300)
try:
    from angel_one_feed import AngelOneLiveFeed
    _angel_feed = AngelOneLiveFeed(live_engine)
    LIVE_FEED_ACTIVE = True
except ImportError:
    _angel_feed = None


@app.on_event("startup")
def start_feed():
    global LIVE_FEED_ACTIVE
    if _angel_feed is not None:
        thread = threading.Thread(target=_angel_feed.run_forever, daemon=True)
        thread.start()
        print("Started REAL Angel One live feed in background thread.")
    else:
        print("angel_one_feed.py not found — running on SIMULATED data. "
              "Place a filled-in angel_one_feed.py in this folder for real live trading.")

    # External reference quotes (USD/INR, COMEX Gold, Dollar Index) — entirely
    # separate from the trading feed above. Display-only context, never feeds
    # into any trading decision.
    external_quotes_poller.start_background_thread()
    print("Started external reference quotes poller (USD/INR, COMEX Gold, DXY).")

    gold_news_monitor.start_background_thread()
    print("Started gold news monitor (Yahoo Finance RSS — Bloomberg-sourced headlines).")


def _get_market_session_status() -> dict:
    """
    MCX gold trading session: ~9:00 AM to 11:30/11:55 PM IST, Mon-Fri.
    This is informational for the dashboard (session badge, countdown) —
    it does NOT gate trading logic, which is the risk engine's job.
    """
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))  # IST
    is_weekday = now.weekday() < 5
    session_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    session_end = now.replace(hour=23, minute=30, second=0, microsecond=0)
    is_open = is_weekday and session_start <= now <= session_end

    if is_open:
        return {"status": "OPEN", "label": "Market Open", "local_time": now.strftime("%H:%M:%S IST")}
    return {"status": "CLOSED", "label": "Market Closed", "local_time": now.strftime("%H:%M:%S IST")}


class HealthResponse(BaseModel):
    status: str
    mode: str
    broker_connected: bool
    data_feed: str
    build_version: str
    # True when the live feed has delivered no volume at all — the signature
    # of subscribing in a mode that carries none. Surfaced here so the
    # dashboard can show it; a silent detector nobody reads is no safety net.
    volume_feed_broken: bool = False
    # Seconds since a tick last ARRIVED (wall clock), and whether that is
    # long enough to consider the feed dead rather than the market quiet.
    # Without this a dropped socket looked identical to a calm session.
    seconds_since_last_tick: float | None = None
    feed_stale: bool = False


class SnapshotResponse(BaseModel):
    instrument: str
    ltp: float
    day_open_price: float | None
    regime_trend: str
    last_structure_event: str
    has_open_position: bool
    open_position: dict | None
    trading_disabled: bool
    trades_taken_today: int
    total_trades_this_session: int
    session_status: str
    session_label: str
    local_time: str


class SignalResponse(BaseModel):
    decision: str
    long_score: int
    short_score: int
    confidence: int


class PerformanceResponse(BaseModel):
    trades_taken: int
    max_trades: int
    net_pnl: float
    total_trades_this_session: int
    current_streak: int
    lot_multiplier: float
    scaleup_recommended: bool
    trading_disabled: bool
    equity_inr: float


def _simulate_next_tick() -> LiveTick:
    """v1 placeholder feed — replace with the real Angel One WebSocket handler
    later; live_engine.on_tick() only depends on the LiveTick shape, not the
    source, so swapping this out is the only change needed for real data."""
    global _last_price, _tick_count
    _tick_count += 1
    drift = random.uniform(-1.5, 2.0)
    noise = random.uniform(5, 20)
    _last_price += drift
    high = _last_price + random.uniform(0, noise)
    low = _last_price - random.uniform(0, noise)
    close = _last_price + random.uniform(-noise / 2, noise / 2)
    volume = 1000 + random.uniform(-200, 500)
    # BUGFIX: this used to pass ts=_tick_count (1, 2, 3...). LiveTradingEngine
    # converts ts via datetime.fromtimestamp() to detect calendar day
    # boundaries, so sequential integers pinned every tick to 1970-01-01 —
    # trades_taken_today never reset and the post-disable cooldown never
    # fired, silently disabling the daily risk controls in live paper
    # trading. Real epoch seconds are required.
    return LiveTick(ts=int(time.time()), open=_last_price, high=max(high, _last_price, close),
                      low=min(low, _last_price, close), close=close, volume=max(volume, 1))


# A live feed that has gone quiet for longer than this during an open
# session is treated as dead, not merely idle. GOLDM trades often enough
# that multiple minutes of true silence is not a normal market state.
FEED_STALE_AFTER_SEC = 180


@app.get("/health", response_model=HealthResponse)
def health():
    volume_broken = False
    if _angel_feed is not None and getattr(_angel_feed, "aggregator", None) is not None:
        volume_broken = _angel_feed.aggregator.volume_feed_looks_broken

    since = live_engine.seconds_since_last_tick() if LIVE_FEED_ACTIVE else None
    session = _get_market_session_status()
    # Only meaningful while the market is actually open — silence overnight
    # or at the weekend is expected, not a fault.
    stale = bool(
        LIVE_FEED_ACTIVE
        and session["status"] == "OPEN"
        and (since is None or since > FEED_STALE_AFTER_SEC)
    )

    return HealthResponse(
        status="ok", mode=settings.mode, broker_connected=LIVE_FEED_ACTIVE,
        data_feed="ANGEL_ONE_LIVE" if LIVE_FEED_ACTIVE else "PAPER_SIMULATED",
        volume_feed_broken=volume_broken,
        seconds_since_last_tick=round(since, 1) if since is not None else None,
        feed_stale=stale, build_version=BUILD_VERSION,
    )


def _get_or_advance_snapshot() -> dict:
    """
    In LIVE mode: the Angel One feed thread is already calling on_tick()
    continuously in the background — this just reads the latest snapshot,
    never injects a fake tick on top of real data.
    In SIMULATED mode: each call generates and advances one simulated tick,
    same as before (this is what makes the demo self-driving without a
    real feed).
    """
    if LIVE_FEED_ACTIVE:
        snap = dict(live_engine.state.last_snapshot) if live_engine.state.last_snapshot else {
            "ts": 0, "ltp": 0.0, "regime_trend": "RANGE", "last_structure_event": "NONE",
            "has_open_position": False, "open_position": None,
            "risk_state": {"trading_disabled": False, "trades_taken_today": 0,
                            "consecutive_losses": 0, "lot_multiplier": 1.0},
            "total_trades_this_session": 0,
        }
        # The snapshot itself is only rebuilt when a candle closes (up to 5
        # minutes apart). Overlay the latest raw tick price so the dashboard
        # shows a current price rather than a stale bar close. Analysis
        # fields are deliberately NOT overlaid — those must stay tied to the
        # last completed candle.
        if live_engine.state.last_tick_price is not None:
            snap["ltp"] = live_engine.state.last_tick_price
            snap["ts"] = live_engine.state.last_tick_ts or snap.get("ts", 0)
        return snap
    tick = _simulate_next_tick()
    return live_engine.on_tick(tick)


@app.get("/api/snapshot", response_model=SnapshotResponse)
def get_snapshot():
    snap = _get_or_advance_snapshot()
    session = _get_market_session_status()
    return SnapshotResponse(
        instrument=settings.instrument.symbol, ltp=round(snap["ltp"], 2),
        day_open_price=round(snap["day_open_price"], 2) if snap.get("day_open_price") else None,
        regime_trend=snap["regime_trend"], last_structure_event=snap["last_structure_event"],
        has_open_position=snap["has_open_position"], open_position=snap["open_position"],
        trading_disabled=snap["risk_state"]["trading_disabled"],
        trades_taken_today=snap["risk_state"]["trades_taken_today"],
        total_trades_this_session=snap["total_trades_this_session"],
        session_status=session["status"], session_label=session["label"],
        local_time=session["local_time"],
    )


@app.get("/api/signal", response_model=SignalResponse)
def get_signal():
    """Latest evaluated signal from the live session's own log. In simulated
    mode, forces one tick if the log is empty so there's something to show."""
    if not live_engine.state.signal_log and not LIVE_FEED_ACTIVE:
        tick = _simulate_next_tick()
        live_engine.on_tick(tick)
    if not live_engine.state.signal_log:
        return SignalResponse(decision="NO_TRADE", long_score=0, short_score=0, confidence=0)
    latest = live_engine.state.signal_log[-1]
    return SignalResponse(
        decision=latest["decision"], long_score=latest["long_score"],
        short_score=latest["short_score"], confidence=0,
    )


@app.get("/api/performance", response_model=PerformanceResponse)
def get_performance():
    st = live_engine.risk_engine.state
    balance = broker.get_balance()
    return PerformanceResponse(
        trades_taken=st.trades_taken_today, max_trades=settings.risk.max_trades_per_day,
        # BUGFIX: this used to show risk_engine's R-multiple-based daily_pnl_inr,
        # an approximation used internally for de-risking decisions. It gave
        # a DIFFERENT number than each trade row's real, brokerage-adjusted
        # net P&L — showing the real, charges-inclusive total here instead.
        net_pnl=round(live_engine.state.real_daily_net_pnl_inr, 2),
        total_trades_this_session=len(live_engine.state.trade_log),
        current_streak=max(st.consecutive_wins, st.consecutive_losses),
        lot_multiplier=st.current_lot_multiplier, scaleup_recommended=st.scaleup_recommended,
        trading_disabled=st.trading_disabled, equity_inr=round(balance.equity_inr, 2),
    )


@app.get("/api/trades")
def get_trades():
    """Full trade history for this live session — the thing the old
    endpoints never actually produced, since they never opened real trades."""
    return {"trades": live_engine.state.trade_log}


@app.get("/api/candles")
def get_candles():
    """Recent OHLCV history for the chart — bounded, see LiveEngineState."""
    return {"candles": live_engine.state.candle_history}


@app.get("/api/daily_pnl")
def get_daily_pnl():
    """Real, brokerage-adjusted P&L grouped by trading day, most recent
    first — reads the full persisted trade history, not just the bounded
    in-memory list, so this stays accurate across a multi-week run."""
    return {"days": live_engine.get_daily_pnl_history()}


def _quote_to_dict(q):
    return {
        "value": q.value, "prev_close": q.prev_close,
        "change": (round(q.value - q.prev_close, 4)
                    if q.value is not None and q.prev_close is not None else None),
        "last_updated_at": q.last_updated_at, "last_error": q.last_error,
        "stale": q.last_error is not None,
    }


@app.get("/api/external_quotes")
def get_external_quotes():
    """USD/INR, COMEX Gold, Dollar Index, US 10Y Treasury Yield — reference
    context only, entirely independent of the trading feed and decisions."""
    s = external_quotes_poller.state
    return {
        "usd_inr": _quote_to_dict(s.usd_inr),
        "comex_gold": _quote_to_dict(s.comex_gold),
        "dollar_index": _quote_to_dict(s.dollar_index),
        "us_10y_treasury": _quote_to_dict(s.us_10y_treasury),
    }


@app.get("/api/news")
def get_gold_news():
    """Recent gold-relevant headlines (Bloomberg-sourced via Yahoo Finance
    RSS), classified for impact. Display-only — does not gate trading."""
    assessments = gold_news_monitor.get_assessments(limit=10)
    return {
        "risk_state": gold_news_monitor.get_current_risk_state().value,
        "items": [
            {
                "text": a.item.text, "author": a.item.author,
                "published_at": a.item.published_at.isoformat(),
                "url": a.item.url, "impact_level": a.impact_level,
                "matched_keywords": a.matched_keywords,
            }
            for a in assessments
        ],
    }


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Pushes a fresh snapshot every ~2 seconds. In LIVE mode this just reads
    the live_engine state the Angel One feed thread is already updating —
    never injects a simulated tick on top of real data."""
    await websocket.accept()
    try:
        while True:
            snap = _get_or_advance_snapshot()
            await websocket.send_json(snap)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Simple self-refreshing HTML dashboard — no separate frontend build
    or hosting needed, just open http://<server-ip>:<port>/ in a browser.
    The build version is substituted server-side (not via JS) so it's
    visible even if something else client-side is broken — the whole point
    is to make "is my latest code actually running" checkable at a glance."""
    return _DASHBOARD_HTML.replace("__BUILD_VERSION__", BUILD_VERSION)


_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>GOLDM — Live Paper Trading</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&%23129395;</text></svg>">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #08090B; --bg-alt: #0C0E11;
    --panel: #13161A; --panel-2: #171B20; --panel-border: #24282F;
    --gold: #C9A227; --gold-soft: #E0BE4F; --gold-dim: #6B5A18;
    --bull: #3FA796; --bull-soft: rgba(63,167,150,0.14);
    --bear: #C24F42; --bear-soft: rgba(194,79,66,0.14);
    --text: #EDEFF2; --text-dim: #9198A3; --text-faint: #565C68;
    --mono: 'JetBrains Mono', 'SF Mono', 'Courier New', monospace;
    --sans: 'Inter', -apple-system, 'Segoe UI', sans-serif;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    background:
      radial-gradient(ellipse 1200px 600px at 50% -10%, rgba(201,162,39,0.05), transparent),
      var(--bg);
    color: var(--text); margin: 0; padding: 0; font-family: var(--sans);
    min-height: 100vh;
  }
  ::selection { background: rgba(201,162,39,0.25); }
  .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 18px 22px 40px; }

  /* ---------- header ---------- */
  header {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    padding: 14px 0 18px; margin-bottom: 18px; border-bottom: 1px solid var(--panel-border);
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand-mark {
    width: 30px; height: 30px; border-radius: 7px;
    background: linear-gradient(135deg, var(--gold-soft), var(--gold-dim));
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 13px; color: #0B0D0F; font-family: var(--mono);
  }
  .brand-text .symbol { color: var(--gold-soft); font-weight: 700; font-size: 17px; letter-spacing: 0.02em; line-height: 1.1; }
  .brand-text .sub { color: var(--text-faint); font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; }

  .price-block { display: flex; align-items: baseline; gap: 10px; margin-left: 6px; }
  .price { font-size: 28px; font-weight: 700; letter-spacing: -0.01em; }
  .change { font-size: 13px; font-weight: 600; padding: 2px 7px; border-radius: 5px; }
  .change.bull { color: var(--bull); background: var(--bull-soft); }
  .change.bear { color: var(--bear); background: var(--bear-soft); }
  .change.flat { color: var(--text-faint); }

  .header-right { margin-left: auto; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .pill {
    display: flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
    border: 1px solid var(--panel-border); background: var(--panel);
  }
  .pill .dot { width: 6px; height: 6px; border-radius: 50%; }
  .pill.session-open .dot { background: var(--bull); box-shadow: 0 0 8px var(--bull); animation: blink 2s infinite; }
  .pill.session-closed .dot { background: var(--text-faint); }
  .pill.session-open { color: var(--bull); border-color: rgba(63,167,150,0.3); }
  .pill.feed-live { color: var(--bull); border-color: rgba(63,167,150,0.3); }
  .pill.feed-live .dot { background: var(--bull); animation: blink 1.4s infinite; }
  .pill.feed-sim { color: var(--gold); border-color: rgba(201,162,39,0.3); }
  .pill.feed-sim .dot { background: var(--gold); }
  .clock { color: var(--text-dim); font-size: 12px; }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.25; } }

  /* ---------- layout ---------- */
  .layout { display: grid; grid-template-columns: 2.1fr 1fr; gap: 16px; align-items: start; }
  @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } }

  .panel {
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 10px;
    overflow: hidden;
  }
  .panel-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-bottom: 1px solid var(--panel-border);
  }
  .panel-head h3 {
    margin: 0; font-size: 11px; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--text-faint); font-weight: 700; display: flex; align-items: center; gap: 7px;
  }
  .panel-head h3::before { content: ''; width: 3px; height: 12px; background: var(--gold-dim); border-radius: 2px; }
  .panel-body { padding: 16px; }

  #chart { height: 340px; }
  #volume { height: 70px; margin-top: 2px; }
  .chart-empty {
    height: 410px; display: flex; align-items: center; justify-content: center;
    color: var(--text-faint); font-size: 13px; text-align: center; flex-direction: column; gap: 10px;
  }
  .chart-empty .icon { font-size: 22px; opacity: 0.4; }

  .stack { display: flex; flex-direction: column; gap: 14px; }
  .row {
    display: flex; justify-content: space-between; align-items: center; padding: 8px 0;
    border-bottom: 1px solid var(--panel-border); font-size: 13px;
  }
  .row:last-child { border-bottom: none; }
  .row .k { color: var(--text-dim); }
  .row .v { font-weight: 600; }
  .bull { color: var(--bull); } .bear { color: var(--bear); } .gold { color: var(--gold-soft); }
  .empty-note { color: var(--text-faint); font-size: 12.5px; padding: 8px 0; }

  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat {
    background: var(--panel-2); border: 1px solid var(--panel-border); border-radius: 8px;
    padding: 11px 12px;
  }
  .stat .label { font-size: 10px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
  .stat .value { font-size: 17px; font-weight: 700; }

  .trade-item {
    display: flex; align-items: center; gap: 10px; padding: 10px 0;
    border-bottom: 1px solid var(--panel-border); font-size: 13px;
  }
  .trade-item-full:last-child .trade-item { border-bottom: none; }
  .trade-sub {
    font-size: 11px; color: var(--text-faint); padding: 0 0 10px 0;
    border-bottom: 1px solid var(--panel-border);
  }
  .trade-item-full:last-child .trade-sub { border-bottom: none; }

  .day-row {
    display: grid; grid-template-columns: 100px 1fr 90px 90px; align-items: center;
    gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--panel-border); font-size: 13px;
  }
  .day-row:last-child { border-bottom: none; }
  .day-row .date { color: var(--text-dim); }
  .day-row .day-stats { color: var(--text-faint); font-size: 11.5px; }
  .day-row .day-net { text-align: right; font-weight: 700; }
  .day-row .day-charges { text-align: right; color: var(--text-faint); font-size: 11px; }

  .news-item { padding: 10px 0; border-bottom: 1px solid var(--panel-border); font-size: 13px; }
  .news-item:last-child { border-bottom: none; }
  .news-item a { color: var(--text); text-decoration: none; }
  .news-item a:hover { color: var(--gold-soft); }
  .news-meta { font-size: 10.5px; color: var(--text-faint); margin-top: 3px; display: flex; gap: 8px; align-items: center; }
  .news-impact { padding: 1px 6px; border-radius: 3px; font-weight: 700; text-transform: uppercase; font-size: 9.5px; }
  .news-impact.HIGH { background: var(--bear-soft); color: var(--bear); }
  .news-impact.MEDIUM { background: rgba(201,162,39,0.14); color: var(--gold-soft); }
  .news-impact.LOW { background: var(--panel-2); color: var(--text-faint); }
  .trade-badge {
    font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; letter-spacing: 0.03em;
  }
  .trade-badge.long { background: var(--bull-soft); color: var(--bull); }
  .trade-badge.short { background: var(--bear-soft); color: var(--bear); }
  .trade-detail { flex: 1; color: var(--text-dim); }
  .trade-r { font-weight: 700; }
  .trade-reason { font-size: 10.5px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.03em; }

  footer {
    margin-top: 20px; padding-top: 14px; border-top: 1px solid var(--panel-border);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    font-size: 11px; color: var(--text-faint);
  }
  footer .dot-live { width: 6px; height: 6px; border-radius: 50%; background: var(--bull); display: inline-block; margin-right: 5px; animation: blink 2s infinite; }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-mark">Au</div>
      <div class="brand-text">
        <div class="symbol mono">GOLDM</div>
        <div class="sub">MCX · Next-Month Futures</div>
      </div>
    </div>

    <div class="price-block">
      <span id="ltp" class="mono price">--</span>
      <span id="change" class="mono change flat">--</span>
    </div>

    <div class="header-right">
      <span id="clock" class="mono clock">--:--:--</span>
      <span id="sessionPill" class="pill session-closed"><span class="dot"></span><span id="sessionLabel">--</span></span>
      <span id="feedBadge" class="pill feed-sim"><span class="dot"></span>Loading</span>
    </div>
  </header>

  <div class="panel" style="margin-bottom:16px; padding:12px 16px;">
    <div style="display:flex; gap:28px; flex-wrap:wrap; align-items:center; font-size:12.5px;">
      <span style="color:var(--text-faint); font-size:10.5px; text-transform:uppercase; letter-spacing:0.06em;">Market Context</span>
      <span class="mono">USD/INR <span id="refUsdInr" style="font-weight:700;">--</span></span>
      <span class="mono">COMEX Gold <span id="refComexGold" style="font-weight:700;">--</span></span>
      <span class="mono">Dollar Index <span id="refDxy" style="font-weight:700;">--</span></span>
      <span class="mono">US 10Y Yield <span id="refTreasury" style="font-weight:700;">--</span></span>
      <span id="refStaleNote" style="color:var(--text-faint); font-size:11px; display:none;">(some values may be stale)</span>
    </div>
  </div>

  <div id="staleWarning" style="display:none; background:rgba(194,79,66,0.16); border:1px solid rgba(194,79,66,0.5); color:var(--bear); padding:11px 14px; border-radius:8px; margin-bottom:14px; font-size:12.5px; font-weight:700;">
    ⛔ FEED STALE — no ticks received recently while the market is open. Trading has effectively stopped. Check the feed process and network.
  </div>

  <div id="volumeWarning" style="display:none; background:rgba(194,79,66,0.12); border:1px solid rgba(194,79,66,0.4); color:var(--bear); padding:11px 14px; border-radius:8px; margin-bottom:14px; font-size:12.5px; font-weight:600;">
    ⚠ Feed is delivering no volume. Volume-confirmation checks are meaningless in this state and the strategy will not behave like its backtest. Check the WebSocket subscription mode (needs Quote mode, not LTP).
  </div>

  <div class="layout">
    <!-- left column: chart -->
    <div class="panel">
      <div class="panel-head">
        <h3>Price Action · 5M</h3>
        <span id="trendTag" class="mono" style="font-size:11px; color:var(--text-faint);">--</span>
      </div>
      <div class="panel-body" style="padding: 10px;">
        <div id="chartEmpty" class="chart-empty">
          <div class="icon">◇</div>
          <div>Waiting for the first candle<br>Chart renders once real data starts flowing.</div>
        </div>
        <div id="chartWrap" style="display:none;">
          <div id="chart"></div>
          <div id="volume"></div>
        </div>
      </div>
    </div>

    <!-- right column: stacked cards -->
    <div class="stack">
      <div class="panel">
        <div class="panel-head"><h3>Situation</h3></div>
        <div class="panel-body">
          <div class="row"><span class="k">Trend</span><span id="regime" class="v mono">--</span></div>
          <div class="row"><span class="k">Last Structure Event</span><span id="event" class="v mono">--</span></div>
          <div class="row"><span class="k">Open Position</span><span id="hasPos" class="v mono">--</span></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head"><h3>Open Position</h3></div>
        <div class="panel-body">
          <div id="posDetails"><div class="empty-note">No open position</div></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head"><h3>Risk &amp; Performance</h3></div>
        <div class="panel-body">
          <div class="stat-grid">
            <div class="stat">
              <div class="label">Equity</div>
              <div id="equity" class="value mono gold">--</div>
            </div>
            <div class="stat">
              <div class="label">Trades Today</div>
              <div id="tradesToday" class="value mono">--</div>
            </div>
            <div class="stat">
              <div class="label">Session Total</div>
              <div id="totalTrades" class="value mono">--</div>
            </div>
            <div class="stat">
              <div class="label">Trading Disabled</div>
              <div id="disabled" class="value mono">--</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="panel-head"><h3>Recent Trades</h3></div>
    <div class="panel-body">
      <div id="tradesList"><div class="empty-note">No trades yet this session.</div></div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="panel-head"><h3>Daily P&amp;L (Real, After Charges)</h3></div>
    <div class="panel-body">
      <div id="dailyPnlList"><div class="empty-note">No completed trading days yet.</div></div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="panel-head">
      <h3>TradingView Reference Chart</h3>
    </div>
    <div class="panel-body">
      <div style="padding:4px 2px 14px; font-size:11.5px; color:var(--text-faint); line-height:1.5;">
        Independent reference only — an embedded widget proved unreliable (kept showing the
        wrong symbol), so this opens your own real TradingView chart in a new tab instead —
        exactly what you already use, no embedding issues. Trading decisions in this dashboard
        are based on this system's own Angel One feed, not TradingView's.
      </div>
      <a href="https://www.tradingview.com/chart/?symbol=MCX%3AGOLDM1%21" target="_blank" rel="noopener"
         style="display:inline-flex; align-items:center; gap:8px; padding:10px 18px; border-radius:8px;
                background: var(--panel-2); border: 1px solid var(--panel-border); color: var(--gold-soft);
                text-decoration:none; font-size:13px; font-weight:600;">
        Open GOLDM chart on TradingView ↗
      </a>
    </div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="panel-head">
      <h3>Gold News</h3>
      <span id="newsRiskBadge" class="mono" style="font-size:11px; color:var(--text-faint);">--</span>
    </div>
    <div class="panel-body">
      <div style="padding:0 0 10px; font-size:11px; color:var(--text-faint);">
        Bloomberg-sourced headlines via Yahoo Finance — display-only, does not gate trading decisions.
      </div>
      <div id="newsList"><div class="empty-note">Loading news...</div></div>
    </div>
  </div>

  <footer>
    <span><span class="dot-live"></span>Auto-refresh: 5s · Build: __BUILD_VERSION__</span>
    <span id="status">Connecting...</span>
  </footer>

</div>

<script>
function fmt(n) { return typeof n === 'number' ? n.toLocaleString('en-IN', {maximumFractionDigits:2}) : n; }

let chart = null, volumeChart = null, candleSeries = null, volumeSeries = null;
let lastClose = null, chartInitialized = false;

function tickClock() {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
  document.getElementById('clock').textContent = ist.toLocaleTimeString('en-IN', { hour12: false }) + ' IST';
}
tickClock();
setInterval(tickClock, 1000);

function initChart() {
  document.getElementById('chartEmpty').style.display = 'none';
  document.getElementById('chartWrap').style.display = 'block';

  // BUGFIX: createChart() was never given an explicit width. Without it,
  // lightweight-charts measures the container at the instant this code
  // runs — which can be BEFORE the browser has finished resolving the
  // CSS grid/flex layout around it, especially on first paint. The result:
  // the chart initializes at 0 (or wrong) width, so the price axis (which
  // has its own fixed-width area) still renders, but the actual candle-
  // drawing canvas collapses and nothing visible gets drawn. Fixed by
  // reading the real, already-laid-out container width, and keeping it in
  // sync on resize.
  const chartContainer = document.getElementById('chart');
  const volumeContainer = document.getElementById('volume');

  chart = LightweightCharts.createChart(chartContainer, {
    width: chartContainer.clientWidth,
    height: 340,
    layout: { background: { color: 'transparent' }, textColor: '#9198A3', fontFamily: 'JetBrains Mono, monospace' },
    grid: { vertLines: { color: '#1C2026' }, horzLines: { color: '#1C2026' } },
    timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#24282F' },
    rightPriceScale: { borderColor: '#24282F' },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: '#C9A227', width: 1, style: 2 }, horzLine: { color: '#C9A227', width: 1, style: 2 } },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#3FA796', downColor: '#C24F42',
    borderUpColor: '#3FA796', borderDownColor: '#C24F42',
    wickUpColor: '#3FA796', wickDownColor: '#C24F42',
  });

  volumeChart = LightweightCharts.createChart(volumeContainer, {
    width: volumeContainer.clientWidth,
    height: 70,
    layout: { background: { color: 'transparent' }, textColor: '#565C68', fontFamily: 'JetBrains Mono, monospace' },
    grid: { vertLines: { color: 'transparent' }, horzLines: { color: 'transparent' } },
    timeScale: { visible: false },
    rightPriceScale: { borderColor: '#24282F' },
  });
  volumeSeries = volumeChart.addHistogramSeries({ color: '#3FA796', priceFormat: { type: 'volume' } });

  chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (range) volumeChart.timeScale().setVisibleLogicalRange(range);
  });

  // Keep both charts sized correctly as the viewport changes (rotation,
  // resize, or a layout shift after fonts/webfonts finish loading) —
  // without this, the SAME collapse could recur on any relayout.
  window.addEventListener('resize', () => {
    if (chart && chartContainer.clientWidth > 0) {
      chart.applyOptions({ width: chartContainer.clientWidth });
    }
    if (volumeChart && volumeContainer.clientWidth > 0) {
      volumeChart.applyOptions({ width: volumeContainer.clientWidth });
    }
  });

  chartInitialized = true;
}

async function loadCandleHistory() {
  const resp = await fetch('/api/candles');
  const data = await resp.json();
  if (data.candles && data.candles.length > 0) {
    if (!chartInitialized) initChart();

    // Defensive: lightweight-charts REQUIRES strictly ascending, unique
    // timestamps in setData() — any violation throws, and that exception
    // used to be swallowed into a generic "Connection error" status with
    // no indication the chart itself was the cause. Sort + dedupe here so
    // a single overlapping tick (e.g. around a feed reconnect) can't
    // silently blank the whole chart, and surface a specific error if
    // setData still fails for any other reason.
    const seen = new Set();
    const sorted = [...data.candles]
      .sort((a, b) => a.ts - b.ts)
      .filter(c => {
        if (seen.has(c.ts)) return false;
        seen.add(c.ts);
        return true;
      });

    const candles = sorted.map(c => ({ time: c.ts, open: c.open, high: c.high, low: c.low, close: c.close }));
    const vols = sorted.map(c => ({
      time: c.ts, value: c.volume,
      color: c.close >= c.open ? 'rgba(63,167,150,0.55)' : 'rgba(194,79,66,0.55)',
    }));

    try {
      candleSeries.setData(candles);
      volumeSeries.setData(vols);
      lastClose = candles[candles.length - 1].close;
      chart.timeScale().fitContent();
      volumeChart.timeScale().fitContent();
    } catch (e) {
      console.error('Chart render error:', e, 'candle count:', candles.length);
      document.getElementById('status').textContent = 'Chart render error: ' + e.message;
    }
  }
}

async function loadDailyPnl() {
  try {
    const resp = await fetch('/api/daily_pnl');
    const data = await resp.json();
    const days = data.days || [];
    const el = document.getElementById('dailyPnlList');
    if (days.length === 0) {
      el.innerHTML = '<div class="empty-note">No completed trading days yet.</div>';
      return;
    }
    el.innerHTML = days.map(d => `
      <div class="day-row">
        <span class="date mono">${d.date}</span>
        <span class="day-stats">${d.trade_count} trade${d.trade_count!==1?'s':''} · ${d.wins}W-${d.losses}L</span>
        <span class="day-charges mono">-₹${fmt(d.total_charges_inr)}</span>
        <span class="day-net mono ${d.net_pnl_inr>=0?'bull':'bear'}">${d.net_pnl_inr>=0?'+':''}₹${fmt(d.net_pnl_inr)}</span>
      </div>`
    ).join('');
  } catch (e) {
    console.error('Daily P&L load error:', e);
  }
}

async function loadExternalQuotes() {
  try {
    const resp = await fetch('/api/external_quotes');
    const data = await resp.json();
    let anyStale = false;

    function render(elId, q, decimals, suffix) {
      const el = document.getElementById(elId);
      if (q.value === null) {
        el.textContent = '--';
        el.style.color = 'var(--text-faint)';
        return;
      }
      suffix = suffix || '';
      const changeStr = q.change !== null
        ? ' ' + (q.change >= 0 ? '▲' : '▼') + Math.abs(q.change).toFixed(decimals) + suffix
        : '';
      el.textContent = q.value.toFixed(decimals) + suffix + changeStr;
      el.style.color = q.stale ? 'var(--text-faint)' : (q.change >= 0 ? 'var(--bull)' : 'var(--bear)');
      if (q.stale) anyStale = true;
    }

    render('refUsdInr', data.usd_inr, 2);
    render('refComexGold', data.comex_gold, 1);
    render('refDxy', data.dollar_index, 2);
    render('refTreasury', data.us_10y_treasury, 2, '%');

    document.getElementById('refStaleNote').style.display = anyStale ? 'inline' : 'none';
  } catch (e) {
    console.error('External quotes load error:', e);
  }
}

async function loadGoldNews() {
  try {
    const resp = await fetch('/api/news');
    const data = await resp.json();
    const badge = document.getElementById('newsRiskBadge');
    const riskColors = {
      'NORMAL': 'var(--text-faint)', 'EVENT_APPROACHING': 'var(--gold-soft)',
      'HIGH_IMPACT': 'var(--bear)', 'POST_EVENT_VOLATILITY': 'var(--gold-soft)',
    };
    badge.textContent = data.risk_state;
    badge.style.color = riskColors[data.risk_state] || 'var(--text-faint)';

    const listEl = document.getElementById('newsList');
    if (!data.items || data.items.length === 0) {
      listEl.innerHTML = '<div class="empty-note">No recent news loaded yet.</div>';
      return;
    }
    listEl.innerHTML = data.items.slice(0, 8).map(item => {
      const time = new Date(item.published_at).toLocaleString('en-IN', {
        day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
      });
      const titleHtml = item.url
        ? `<a href="${item.url}" target="_blank" rel="noopener">${item.text}</a>`
        : item.text;
      return `<div class="news-item">
        <div>${titleHtml}</div>
        <div class="news-meta">
          <span class="news-impact ${item.impact_level}">${item.impact_level}</span>
          <span>${item.author}</span><span>·</span><span>${time}</span>
        </div>
      </div>`;
    }).join('');
  } catch (e) {
    console.error('News load error:', e);
  }
}

async function refresh() {
  try {
    const snapResp = await fetch('/api/snapshot');
    const snap = await snapResp.json();

    // BUGFIX: this used to compare against `lastClose` — the close of the
    // most recently loaded 5-minute candle — which flips sign on ordinary
    // tick noise every few minutes regardless of the day's real trend.
    // A user could see red/down here while the actual session was up
    // hundreds of points. Now compares against the DAY'S OPEN, matching
    // what every real trading platform means by "change".
    document.getElementById('ltp').textContent = snap.ltp ? '₹' + fmt(snap.ltp) : '--';
    const changeEl = document.getElementById('change');
    if (snap.day_open_price && snap.ltp) {
      const diff = snap.ltp - snap.day_open_price;
      const pct = (diff / snap.day_open_price) * 100;
      changeEl.textContent = (diff >= 0 ? '▲ ' : '▼ ') + Math.abs(diff).toFixed(2) +
        ' (' + (diff >= 0 ? '+' : '') + pct.toFixed(2) + '%)';
      changeEl.className = 'mono change ' + (diff > 0 ? 'bull' : diff < 0 ? 'bear' : 'flat');
    }

    document.getElementById('regime').textContent = snap.regime_trend;
    document.getElementById('trendTag').textContent = snap.regime_trend + ' · ' + snap.last_structure_event;
    document.getElementById('event').textContent = snap.last_structure_event;
    document.getElementById('hasPos').textContent = snap.has_open_position ? 'YES' : 'NO';
    document.getElementById('disabled').innerHTML = snap.trading_disabled
      ? '<span class="bear">YES ⚠️</span>' : '<span class="bull">No</span>';
    document.getElementById('tradesToday').textContent = snap.trades_taken_today;
    document.getElementById('totalTrades').textContent = snap.total_trades_this_session;

    const sessionPill = document.getElementById('sessionPill');
    const sessionLabel = document.getElementById('sessionLabel');
    sessionLabel.textContent = snap.session_label + ' · ' + snap.local_time;
    sessionPill.className = 'pill ' + (snap.session_status === 'OPEN' ? 'session-open' : 'session-closed');

    if (snap.open_position) {
      const p = snap.open_position;
      document.getElementById('posDetails').innerHTML =
        '<div class="row"><span class="k">Direction</span><span class="v mono ' + (p.direction==='LONG'?'bull':'bear') + '">' + p.direction + '</span></div>' +
        '<div class="row"><span class="k">Entry</span><span class="v mono">₹' + fmt(p.entry_price) + '</span></div>' +
        '<div class="row"><span class="k">Current Stop</span><span class="v mono">₹' + fmt(p.current_stop) + '</span></div>' +
        '<div class="row"><span class="k">State</span><span class="v mono">' + p.state + '</span></div>' +
        '<div class="row"><span class="k">Qty Remaining</span><span class="v mono">' + p.quantity_remaining_pct + '%</span></div>';
    } else {
      document.getElementById('posDetails').innerHTML = '<div class="empty-note">No open position</div>';
    }

    const perfResp = await fetch('/api/performance');
    const perf = await perfResp.json();
    document.getElementById('equity').textContent = '₹' + fmt(perf.equity_inr);

    const healthResp = await fetch('/health');
    const health = await healthResp.json();
    const badge = document.getElementById('feedBadge');
    if (health.data_feed === 'ANGEL_ONE_LIVE' && health.feed_stale) {
      // never show a reassuring green LIVE badge over a dead feed
      badge.innerHTML = '<span class="dot"></span>Feed stale';
      badge.className = 'pill feed-sim';
    } else if (health.data_feed === 'ANGEL_ONE_LIVE') {
      const age = health.seconds_since_last_tick;
      badge.innerHTML = '<span class="dot"></span>Live — Angel One' +
        (age !== null && age !== undefined ? ' · ' + Math.round(age) + 's' : '');
      badge.className = 'pill feed-live';
    } else {
      badge.innerHTML = '<span class="dot"></span>Simulated';
      badge.className = 'pill feed-sim';
    }
    document.getElementById('volumeWarning').style.display = health.volume_feed_broken ? 'block' : 'none';
    document.getElementById('staleWarning').style.display = health.feed_stale ? 'block' : 'none';

    const tradesResp = await fetch('/api/trades');
    const tradesData = await tradesResp.json();
    const trades = tradesData.trades.slice(-10).reverse();
    if (trades.length === 0) {
      document.getElementById('tradesList').innerHTML = '<div class="empty-note">No trades yet this session.</div>';
    } else {
      document.getElementById('tradesList').innerHTML = trades.map(t => {
        const hasCharges = typeof t.net_pnl_inr === 'number';
        const netPnl = hasCharges ? t.net_pnl_inr : null;
        const pnlLine = hasCharges
          ? `<span class="trade-r mono ${netPnl>=0?'bull':'bear'}">${netPnl>=0?'+':''}₹${fmt(netPnl)}</span>`
          : `<span class="trade-r mono ${t.r_multiple>=0?'bull':'bear'}">${t.r_multiple>=0?'+':''}${t.r_multiple.toFixed(2)}R</span>`;
        const subLine = hasCharges
          ? `<div class="trade-sub">Gross ₹${fmt(t.gross_pnl_inr)} · Charges ₹${fmt(t.total_charges_inr)} · ${t.lots} lot${t.lots>1?'s':''} · ${t.r_multiple.toFixed(2)}R</div>`
          : '';
        return `
        <div class="trade-item-full">
          <div class="trade-item">
            <span class="trade-badge ${t.direction==='LONG'?'long':'short'}">${t.direction}</span>
            <span class="trade-detail mono">₹${fmt(t.entry_price)} → ₹${fmt(t.exit_price)}</span>
            <span class="trade-reason">${t.exit_reason}</span>
            ${pnlLine}
          </div>
          ${subLine}
        </div>`;
      }).join('');
    }

    await loadCandleHistory();
    await loadDailyPnl();
    await loadExternalQuotes();
    await loadGoldNews();

    document.getElementById('status').textContent = 'Last updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('status').textContent = 'Connection error: ' + e;
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
