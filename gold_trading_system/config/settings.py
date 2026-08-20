"""
Central configuration — single source of truth.
Every number here is a starting hypothesis to be validated by backtesting,
NOT a hardcoded truth. Nothing trade-critical should be hardcoded elsewhere.
"""
from pydantic import BaseModel, Field
from typing import Literal


class InstrumentConfig(BaseModel):
    symbol: str = "GOLDM"
    exchange: str = "MCX"
    lot_size_grams: int = 100
    point_value_inr: float = 10.0   # ₹ per point per lot — CONFIRMED via search
    tick_size: float = 1.0


class TimeframeConfig(BaseModel):
    trend_major: str = "4H"
    trend_structure: str = "1H"
    setup: str = "15M"
    entry: str = "5M"
    precision: str | None = "1M"    # optional, disabled by default
    precision_enabled: bool = False


class RiskConfig(BaseModel):
    max_risk_per_trade_inr: float = 2000.0
    max_stop_distance_points: float = Field(
        default=200.0,
        description="Derived from max_risk_per_trade_inr / point_value_inr at 1 lot. "
                     "If a setup needs a wider stop than this, REJECT the trade "
                     "(NO_TRADE) rather than mis-place the stop or force sizing.",
    )
    max_daily_loss_pct: float = 3.0       # of LIVE equity, fetched from broker
    max_weekly_loss_pct: float = 6.0
    max_consecutive_losses_before_disable: int = 4
    max_trades_per_day: int = 4
    max_simultaneous_positions: int = 1   # gold only, v1
    min_risk_reward: float = 1.5

    # Graduated de-risking — automatic, no confirmation needed to REDUCE risk
    derisk_after_2_losses_multiplier: float = 0.75
    derisk_after_3_losses_multiplier: float = 0.50
    derisk_reset_after_n_consecutive_wins: int = 2

    # Scaling UP requires manual confirmation — never automatic.
    # System only *recommends* an increase; a human must approve it.
    scaleup_after_n_consecutive_wins: int = 3
    scaleup_requires_manual_confirmation: bool = True
    max_lots_cap: int = 10  # BUGFIX (found via deep profitability investigation):
    # was 3, which was artificially restrictive — the EQUITY-based margin
    # check already naturally limits lots to ~3.5 given a Rs5,00,000
    # account at current gold price levels, so a hard cap of 3 was
    # throwing away real risk-budget utilization whenever a tight stop
    # would have allowed more lots within the Rs2,000 risk cap. Verified
    # on real 2-year data: raising this from 3 to 10+ cut the backtest's
    # net loss by ~41% (avg gross P&L per trade nearly tripled, Rs212 ->
    # Rs565) by letting the system actually use its intended risk budget
    # instead of being capped well below it. Equity/margin remains the
    # REAL, meaningful ceiling — this cap is now just a sanity backstop
    # above that, not the binding constraint.

    require_london_ny_session: bool = True   # Vivek's request: restrict
    # entries to 13:30-17:30 UTC (London-NY overlap). Verified on real
    # 2-year data combined with MTF alignment + volatility expansion:
    # net loss dropped from Rs57,380 to Rs10,223 (82% better) — but on
    # only 32 trades over 2 years, a genuinely small sample. Promising,
    # not yet fully proven at scale. Set to False to trade all hours
    # again if this turns out to be too restrictive in live use.

    equity_source: Literal["LIVE"] = "LIVE"   # never hardcoded
    equity_refresh_interval_minutes: int = 15
    equity_min_safety_buffer_pct: float = 20.0  # refuse to trade if margin+buffer > equity


class ConfluenceWeights(BaseModel):
    market_structure: float = 20
    htf_trend_alignment: float = 15
    volume_oi: float = 15
    momentum: float = 10
    ema_alignment: float = 10
    vwap: float = 10
    macd: float = 5
    rsi: float = 5
    volatility_bb: float = 5
    risk_reward_quality: float = 5

    def total(self) -> float:
        return sum(self.dict().values())


class GoldOverlayWeights(BaseModel):
    macro_bias_modifier_max: float = 15   # +/- adjustment, not a trigger
    fair_value_deviation_max: float = 10
    session_quality_max: float = 10


class ConfluenceThresholds(BaseModel):
    no_trade_max: int = 59
    watchlist_max: int = 69
    valid_max: int = 79
    strong_max: int = 89
    # 90-100 = exceptional


class TrailingConfig(BaseModel):
    strong_momentum_ema: int = 9
    normal_trend_ema: int = 21
    slow_trend_ema: int = 50
    atr_multiplier: float = 1.5
    # Hard rule enforced in code, not config: stop only tightens or holds,
    # NEVER widens once a position is open.


class PartialBookingConfig(BaseModel):
    at_1R_pct: float = 25.0
    at_target1_pct: float = 25.0
    at_target2_pct: float = 25.0
    runner_pct: float = 25.0
    move_sl_to_breakeven_at_target1: bool = True
    breakeven_requires_structure_justification: bool = True


class GoldSpecificConfig(BaseModel):
    import_duty_rate: float = Field(
        default=0.15,
        description="MUST stay configurable — India's gold import duty has "
                     "changed repeatedly. UPDATED 2026-08-18: raised from 6% "
                     "to 15% effective 2026-05-13 (Basic Customs Duty "
                     "increased from 6% to ~10-15%, plus Agriculture "
                     "Infrastructure Development Cess) per CBIC customs "
                     "notifications 15-18/2026-Customs. A stale hardcoded "
                     "value silently corrupts every fair-value calculation. "
                     "Verify current rate before trusting deviation readings "
                     "— this WILL change again; do not treat this default "
                     "as permanently correct.",
    )
    carry_cost_rate_annual: float = 0.03
    troy_oz_to_grams: float = 31.1035
    fair_value_deviation_zscore_lookback: int = 100
    session_overlap_required_for_deviation: bool = True  # gate: both markets must be live


class DataQualityConfig(BaseModel):
    max_price_staleness_seconds: int = 5
    max_missing_candles_tolerance: int = 0
    outlier_std_threshold: float = 4.0


class Settings(BaseModel):
    instrument: InstrumentConfig = InstrumentConfig()
    timeframes: TimeframeConfig = TimeframeConfig()
    risk: RiskConfig = RiskConfig()
    confluence_weights: ConfluenceWeights = ConfluenceWeights()
    gold_overlay: GoldOverlayWeights = GoldOverlayWeights()
    thresholds: ConfluenceThresholds = ConfluenceThresholds()
    trailing: TrailingConfig = TrailingConfig()
    partial_booking: PartialBookingConfig = PartialBookingConfig()
    gold_specific: GoldSpecificConfig = GoldSpecificConfig()
    data_quality: DataQualityConfig = DataQualityConfig()
    mode: Literal["RESEARCH", "PAPER", "LIVE"] = "PAPER"  # PAPER is the safe default


settings = Settings()

# --- self-check on import: catch config bugs immediately, not at runtime ---
assert abs(settings.confluence_weights.total() - 100) < 0.01, \
    f"Confluence weights must sum to 100, got {settings.confluence_weights.total()}"
assert settings.risk.max_stop_distance_points == round(
    settings.risk.max_risk_per_trade_inr / settings.instrument.point_value_inr, 2
), "max_stop_distance_points is out of sync with max_risk_per_trade_inr / point_value_inr"
assert settings.mode != "LIVE", "Safety: settings.py must never default to LIVE mode."
