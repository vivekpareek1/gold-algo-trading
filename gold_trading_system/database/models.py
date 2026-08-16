"""
SQLAlchemy models — matches the Sprint 1 schema design.
Every trade decision (including NO_TRADEs and vetoed signals) is logged;
you cannot analyse why the system passed on setups if you only store
executed trades.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class Instrument(Base):
    __tablename__ = "instrument"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, unique=True)   # e.g. GOLDM
    exchange = Column(String, nullable=False)               # MCX
    lot_size_grams = Column(Integer, nullable=False)
    point_value_inr = Column(Float, nullable=False)
    tick_size = Column(Float, nullable=False)
    expiry_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)


class MarketCandle(Base):
    __tablename__ = "market_candle"
    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey("instrument.id"), nullable=False)
    timeframe = Column(String, nullable=False)   # "5M", "15M", "1H", "4H"
    ts = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    oi = Column(Float, nullable=True)
    is_complete = Column(Boolean, default=True)  # NEVER act on incomplete candles

    __table_args__ = (
        UniqueConstraint("instrument_id", "timeframe", "ts", name="uq_candle"),
    )


class MacroSnapshot(Base):
    __tablename__ = "macro_snapshot"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False, default=utcnow)
    xauusd = Column(Float)
    usdinr = Column(Float)
    dxy = Column(Float)
    us10y_nominal = Column(Float)
    us10y_real = Column(Float)
    crude = Column(Float)
    source = Column(String)
    is_stale = Column(Boolean, default=False)


class FairValueSnapshot(Base):
    __tablename__ = "fair_value_snapshot"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False, default=utcnow)
    mcx_price = Column(Float)
    theoretical_price = Column(Float)
    deviation = Column(Float)
    deviation_pct = Column(Float)
    deviation_zscore = Column(Float, nullable=True)
    move_classification = Column(String)  # METAL_DRIVEN / RUPEE_DRIVEN / AMPLIFIED / CONFLICTED / FLAT
    is_reliable = Column(Boolean, default=True)
    unreliable_reason = Column(String, nullable=True)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshot"
    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey("instrument.id"), nullable=False)
    ts = Column(DateTime, nullable=False, default=utcnow)
    timeframe = Column(String, nullable=False)
    ema9 = Column(Float)
    ema21 = Column(Float)
    ema50 = Column(Float)
    ema200 = Column(Float)
    rsi = Column(Float)
    macd_line = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bb_upper = Column(Float)
    bb_mid = Column(Float)
    bb_lower = Column(Float)
    atr = Column(Float)
    vwap = Column(Float)
    rel_volume = Column(Float)
    regime = Column(String)          # MarketRegime value
    structure_state = Column(JSON)   # trend, last_event, active fvgs etc, serialized


class StrategyModel(Base):
    __tablename__ = "strategy"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    versions = relationship("StrategyVersion", back_populates="strategy")


class StrategyVersion(Base):
    __tablename__ = "strategy_version"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategy.id"), nullable=False)
    version = Column(String, nullable=False)
    config_json = Column(JSON, nullable=False)   # snapshot of config/settings.py at this version
    created_at = Column(DateTime, default=utcnow)
    backtest_run_id = Column(Integer, ForeignKey("backtest_run.id"), nullable=True)
    is_approved = Column(Boolean, default=False)  # never auto-approved — human gate
    approved_by = Column(String, nullable=True)

    strategy = relationship("StrategyModel", back_populates="versions")


class Signal(Base):
    __tablename__ = "signal"
    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey("instrument.id"), nullable=False)
    ts = Column(DateTime, nullable=False, default=utcnow)
    strategy_version_id = Column(Integer, ForeignKey("strategy_version.id"), nullable=True)

    long_score = Column(Integer)
    short_score = Column(Integer)
    macro_bias = Column(Float)

    decision = Column(String)        # BUY / SELL / NO_TRADE
    confidence = Column(Integer)
    entry_zone_low = Column(Float, nullable=True)
    entry_zone_high = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target_1 = Column(Float, nullable=True)
    target_2 = Column(Float, nullable=True)
    target_3 = Column(Float, nullable=True)
    risk_reward = Column(Float, nullable=True)
    position_size = Column(Integer, nullable=True)

    trade_type = Column(String, nullable=True)   # BREAKOUT/PULLBACK/MOMENTUM/REVERSAL/RANGE
    trailing_method = Column(String, nullable=True)
    reasons_for = Column(JSON, nullable=True)
    reasons_against = Column(JSON, nullable=True)
    invalidation_condition = Column(Text, nullable=True)
    ai_explanation = Column(Text, nullable=True)

    # CRITICAL: log rejections too — cannot learn from only executed trades
    was_vetoed = Column(Boolean, default=False)
    veto_reason = Column(String, nullable=True)


class Trade(Base):
    __tablename__ = "trade"
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("signal.id"), nullable=True)
    instrument_id = Column(Integer, ForeignKey("instrument.id"), nullable=False)
    mode = Column(String, nullable=False)  # RESEARCH / PAPER / LIVE

    entry_ts = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=True)   # lots
    stop_loss = Column(Float, nullable=True)

    exit_ts = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String, nullable=True)

    pnl = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    fees = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)

    max_favourable_excursion = Column(Float, nullable=True)  # for tuning stops/targets from real data
    max_adverse_excursion = Column(Float, nullable=True)

    post_trade_grade = Column(String, nullable=True)  # GOOD_WIN/BAD_WIN/GOOD_LOSS/BAD_LOSS


class TradeStateEvent(Base):
    __tablename__ = "trade_state_event"
    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey("trade.id"), nullable=False)
    ts = Column(DateTime, default=utcnow)
    from_state = Column(String)
    to_state = Column(String)
    trigger_reason = Column(String)
    price_at_event = Column(Float)


class TrailingStopEvent(Base):
    __tablename__ = "trailing_stop_event"
    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey("trade.id"), nullable=False)
    ts = Column(DateTime, default=utcnow)
    old_stop = Column(Float)
    new_stop = Column(Float)
    method_used = Column(String)   # ATR/EMA9/EMA21/EMA50/STRUCTURE/HYBRID
    reason = Column(String)
    price_at_event = Column(Float)


class OrderRecord(Base):
    __tablename__ = "order_record"
    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey("trade.id"), nullable=True)
    client_order_id = Column(String, nullable=False, unique=True)  # idempotency guard
    broker_order_id = Column(String, nullable=True)
    side = Column(String)
    qty = Column(Integer)
    price = Column(Float, nullable=True)
    order_type = Column(String)
    status = Column(String)
    ts = Column(DateTime, default=utcnow)
    raw_response = Column(JSON, nullable=True)


class RiskEvent(Base):
    __tablename__ = "risk_event"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=utcnow)
    event_type = Column(String)
    severity = Column(String)
    description = Column(Text)
    action_taken = Column(String)
    trade_id = Column(Integer, ForeignKey("trade.id"), nullable=True)
    is_resolved = Column(Boolean, default=False)


class DailyPerformance(Base):
    __tablename__ = "daily_performance"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, unique=True)
    trades_taken = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    gross_pnl = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    r_sum = Column(Float, default=0.0)
    risk_limit_hit = Column(Boolean, default=False)
    trading_disabled = Column(Boolean, default=False)
    current_lot_multiplier = Column(Float, default=1.0)
    streak_count = Column(Integer, default=0)   # positive = win streak, negative = loss streak


class NewsEvent(Base):
    __tablename__ = "news_event"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False)
    event_name = Column(String)
    country = Column(String)
    impact_level = Column(String)   # LOW/MEDIUM/HIGH
    actual = Column(Float, nullable=True)
    forecast = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    affects_gold = Column(Boolean, default=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=utcnow)
    actor = Column(String)
    action = Column(String)
    entity_type = Column(String)
    entity_id = Column(Integer, nullable=True)
    before_json = Column(JSON, nullable=True)
    after_json = Column(JSON, nullable=True)


class BacktestRun(Base):
    __tablename__ = "backtest_run"
    id = Column(Integer, primary_key=True)
    strategy_version_id = Column(Integer, ForeignKey("strategy_version.id"), nullable=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    instrument_id = Column(Integer, ForeignKey("instrument.id"))
    total_trades = Column(Integer)
    win_rate = Column(Float)
    expectancy = Column(Float)
    profit_factor = Column(Float)
    sharpe = Column(Float, nullable=True)
    sortino = Column(Float, nullable=True)
    max_drawdown = Column(Float)
    avg_r = Column(Float)
    consecutive_losses = Column(Integer)
    config_snapshot_json = Column(JSON)
    created_at = Column(DateTime, default=utcnow)
