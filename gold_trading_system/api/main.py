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

from config.settings import Settings
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from execution.live_trading_engine import LiveTradingEngine, LiveTick

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
live_engine = LiveTradingEngine(settings, broker, symbol="GOLDM")

_last_price = 63000.0
_tick_count = 0

# Try to load the real Angel One feed handler. This file is gitignored and
# only ever exists on the deployment server, with real credentials filled
# in — importing it here never exposes anything to the git-tracked repo.
LIVE_FEED_ACTIVE = False
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


class HealthResponse(BaseModel):
    status: str
    mode: str
    broker_connected: bool
    data_feed: str


class SnapshotResponse(BaseModel):
    instrument: str
    ltp: float
    regime_trend: str
    last_structure_event: str
    has_open_position: bool
    open_position: dict | None
    trading_disabled: bool
    trades_taken_today: int
    total_trades_this_session: int


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


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok", mode=settings.mode, broker_connected=LIVE_FEED_ACTIVE,
        data_feed="ANGEL_ONE_LIVE" if LIVE_FEED_ACTIVE else "PAPER_SIMULATED",
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
        return live_engine.state.last_snapshot or {
            "ts": 0, "ltp": 0.0, "regime_trend": "RANGE", "last_structure_event": "NONE",
            "has_open_position": False, "open_position": None,
            "risk_state": {"trading_disabled": False, "trades_taken_today": 0,
                            "consecutive_losses": 0, "lot_multiplier": 1.0},
            "total_trades_this_session": 0,
        }
    tick = _simulate_next_tick()
    return live_engine.on_tick(tick)


@app.get("/api/snapshot", response_model=SnapshotResponse)
def get_snapshot():
    snap = _get_or_advance_snapshot()
    return SnapshotResponse(
        instrument=settings.instrument.symbol, ltp=round(snap["ltp"], 2),
        regime_trend=snap["regime_trend"], last_structure_event=snap["last_structure_event"],
        has_open_position=snap["has_open_position"], open_position=snap["open_position"],
        trading_disabled=snap["risk_state"]["trading_disabled"],
        trades_taken_today=snap["risk_state"]["trades_taken_today"],
        total_trades_this_session=snap["total_trades_this_session"],
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
        net_pnl=round(st.daily_pnl_inr, 2),
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
    or hosting needed, just open http://<server-ip>:<port>/ in a browser."""
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Gold Trading System — Live</title>
<meta charset="utf-8">
<style>
  body { background:#0B0D0F; color:#E7E9EC; font-family: 'Segoe UI', Arial, sans-serif; margin:0; padding:20px; }
  .mono { font-family: 'Courier New', monospace; }
  .grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:16px; margin-top:16px; }
  .card { background:#14171A; border:1px solid #22262B; border-radius:8px; padding:16px; }
  .card h3 { margin:0 0 12px 0; font-size:12px; letter-spacing:0.08em; text-transform:uppercase; color:#8A93A3; }
  .big { font-size:28px; font-weight:700; }
  .row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #22262B; font-size:14px; }
  .bull { color:#3FA796; } .bear { color:#B8483C; } .gold { color:#C9A227; }
  .badge { display:inline-block; padding:3px 8px; border-radius:4px; font-size:11px; text-transform:uppercase; }
  .badge.live { background:rgba(63,167,150,0.15); color:#3FA796; }
  .badge.sim { background:rgba(201,162,39,0.15); color:#C9A227; }
  #status { font-size:12px; color:#565D69; margin-top:8px; }
</style>
</head>
<body>
  <div style="display:flex; align-items:center; gap:12px;">
    <span class="gold mono" style="font-size:22px; font-weight:700;">GOLDM</span>
    <span id="ltp" class="mono big">--</span>
    <span id="feedBadge" class="badge sim">Loading...</span>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Situation</h3>
      <div class="row"><span>Trend</span><span id="regime" class="mono">--</span></div>
      <div class="row"><span>Last Event</span><span id="event" class="mono">--</span></div>
      <div class="row"><span>Open Position</span><span id="hasPos" class="mono">--</span></div>
    </div>
    <div class="card">
      <h3>Open Position</h3>
      <div id="posDetails">No open position</div>
    </div>
    <div class="card">
      <h3>Risk State</h3>
      <div class="row"><span>Trading Disabled</span><span id="disabled" class="mono">--</span></div>
      <div class="row"><span>Trades Today</span><span id="tradesToday" class="mono">--</span></div>
      <div class="row"><span>Total This Session</span><span id="totalTrades" class="mono">--</span></div>
      <div class="row"><span>Equity</span><span id="equity" class="mono">--</span></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Recent Trades</h3>
    <div id="tradesList">Loading...</div>
  </div>

  <div id="status">Connecting...</div>

<script>
function fmt(n) { return typeof n === 'number' ? n.toLocaleString('en-IN', {maximumFractionDigits:2}) : n; }

async function refresh() {
  try {
    const snapResp = await fetch('/api/snapshot');
    const snap = await snapResp.json();
    document.getElementById('ltp').textContent = '₹' + fmt(snap.ltp);
    document.getElementById('regime').textContent = snap.regime_trend;
    document.getElementById('event').textContent = snap.last_structure_event;
    document.getElementById('hasPos').textContent = snap.has_open_position ? 'YES' : 'NO';
    document.getElementById('disabled').textContent = snap.trading_disabled ? 'YES ⚠️' : 'No';
    document.getElementById('tradesToday').textContent = snap.trades_taken_today;
    document.getElementById('totalTrades').textContent = snap.total_trades_this_session;

    if (snap.open_position) {
      const p = snap.open_position;
      document.getElementById('posDetails').innerHTML =
        `<div class="row"><span>Direction</span><span class="mono ${p.direction==='LONG'?'bull':'bear'}">${p.direction}</span></div>
         <div class="row"><span>Entry</span><span class="mono">₹${fmt(p.entry_price)}</span></div>
         <div class="row"><span>Current Stop</span><span class="mono">₹${fmt(p.current_stop)}</span></div>
         <div class="row"><span>State</span><span class="mono">${p.state}</span></div>
         <div class="row"><span>Qty Remaining</span><span class="mono">${p.quantity_remaining_pct}%</span></div>`;
    } else {
      document.getElementById('posDetails').textContent = 'No open position';
    }

    const perfResp = await fetch('/api/performance');
    const perf = await perfResp.json();
    document.getElementById('equity').textContent = '₹' + fmt(perf.equity_inr);

    const healthResp = await fetch('/health');
    const health = await healthResp.json();
    const badge = document.getElementById('feedBadge');
    if (health.data_feed === 'ANGEL_ONE_LIVE') {
      badge.textContent = 'LIVE — Angel One';
      badge.className = 'badge live';
    } else {
      badge.textContent = 'SIMULATED DATA';
      badge.className = 'badge sim';
    }

    const tradesResp = await fetch('/api/trades');
    const tradesData = await tradesResp.json();
    const trades = tradesData.trades.slice(-10).reverse();
    if (trades.length === 0) {
      document.getElementById('tradesList').textContent = 'No trades yet this session.';
    } else {
      document.getElementById('tradesList').innerHTML = trades.map(t =>
        `<div class="row"><span>${t.direction} @ ₹${fmt(t.entry_price)} → ₹${fmt(t.exit_price)}</span>
         <span class="mono ${t.r_multiple>=0?'bull':'bear'}">${t.r_multiple>=0?'+':''}${t.r_multiple.toFixed(2)}R (${t.exit_reason})</span></div>`
      ).join('');
    }

    document.getElementById('status').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('status').textContent = 'Connection error: ' + e;
  }
}

refresh();
setInterval(refresh, 5000);  // refresh every 5 seconds
</script>
</body>
</html>
"""
