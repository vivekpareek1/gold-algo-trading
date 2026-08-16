"""
API layer — connects the backend engines to the dashboard.
Runs entirely on the paper broker / simulated feed; no Angel One credentials
needed here. When live integration happens later, only the market_data feed
source changes — these endpoints and their response shapes stay the same.

Uses LiveTradingEngine for actual continuous paper trading: state persists
across requests/ticks, positions are genuinely opened/managed/closed, not
just displayed. (Earlier versions of this file only computed read-only
snapshots and never actually traded — fixed as part of pre-paper-trading
validation.)
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import random
import time

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
        status="ok", mode=settings.mode, broker_connected=False,
        data_feed="PAPER_SIMULATED",
    )


@app.get("/api/snapshot", response_model=SnapshotResponse)
def get_snapshot():
    tick = _simulate_next_tick()
    snap = live_engine.on_tick(tick)
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
    """Latest evaluated signal from the live session's own log — does NOT
    force a new tick, since signal evaluation only happens when no position
    is open (see LiveTradingEngine._evaluate_new_entry)."""
    if not live_engine.state.signal_log:
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
    """Pushes a fresh snapshot every ~2 seconds — replace polling interval /
    source with the real Angel One WebSocket tick handler later. Each tick
    genuinely advances the persistent live_engine session (opens/manages/
    closes real paper positions), not just a display refresh."""
    await websocket.accept()
    try:
        while True:
            tick = _simulate_next_tick()
            snap = live_engine.on_tick(tick)
            await websocket.send_json(snap)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
