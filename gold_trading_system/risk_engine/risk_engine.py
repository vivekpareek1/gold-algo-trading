"""
Risk Engine — FINAL VETO AUTHORITY.
AI can recommend. Risk engine can reject. Execution can only run approved trades.
Never allowed to be overridden by AI confidence.
"""
from dataclasses import dataclass
from enum import Enum


class VetoReason(str, Enum):
    NONE = "NONE"
    STOP_TOO_WIDE = "STOP_TOO_WIDE"
    DAILY_LOSS_LIMIT_HIT = "DAILY_LOSS_LIMIT_HIT"
    WEEKLY_LOSS_LIMIT_HIT = "WEEKLY_LOSS_LIMIT_HIT"
    MAX_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
    MAX_TRADES_TODAY = "MAX_TRADES_TODAY"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    RISK_REWARD_TOO_LOW = "RISK_REWARD_TOO_LOW"
    INSUFFICIENT_EQUITY = "INSUFFICIENT_EQUITY"
    POSITION_SIZE_ROUNDS_TO_ZERO = "POSITION_SIZE_ROUNDS_TO_ZERO"
    STALE_DATA = "STALE_DATA"


@dataclass
class DailyRiskState:
    trades_taken_today: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    daily_pnl_inr: float = 0.0
    weekly_pnl_inr: float = 0.0
    trading_disabled: bool = False
    current_lot_multiplier: float = 1.0
    scaleup_recommended: bool = False   # never auto-applied; needs manual confirm


@dataclass
class PositionSizeResult:
    approved: bool
    lots: int
    risk_amount_inr: float
    veto_reason: VetoReason = VetoReason.NONE
    notes: str = ""


class RiskEngine:
    def __init__(self, config, daily_state: DailyRiskState):
        self.config = config
        self.state = daily_state
        # BUGFIX: was a hardcoded 0, which made the POSITION_ALREADY_OPEN veto
        # permanently unreachable (1 <= 0 is always False). Tracked explicitly
        # and updated by the caller as positions open/close.
        self._open_positions = 0

    def check_hard_limits(self, live_equity_inr: float, data_is_stale: bool,
                           position_already_open: bool) -> VetoReason:
        """Fast fail-checks that run BEFORE any position sizing math."""
        if data_is_stale:
            return VetoReason.STALE_DATA
        if self.state.trading_disabled:
            return VetoReason.MAX_CONSECUTIVE_LOSSES
        if self.state.trades_taken_today >= self.config.risk.max_trades_per_day:
            return VetoReason.MAX_TRADES_TODAY
        if self._open_positions_count() >= self.config.risk.max_simultaneous_positions:
            return VetoReason.POSITION_ALREADY_OPEN
        if position_already_open and self.config.risk.max_simultaneous_positions <= 1:
            return VetoReason.POSITION_ALREADY_OPEN

        daily_loss_limit = live_equity_inr * (self.config.risk.max_daily_loss_pct / 100)
        if self.state.daily_pnl_inr <= -daily_loss_limit:
            return VetoReason.DAILY_LOSS_LIMIT_HIT

        weekly_loss_limit = live_equity_inr * (self.config.risk.max_weekly_loss_pct / 100)
        if self.state.weekly_pnl_inr <= -weekly_loss_limit:
            return VetoReason.WEEKLY_LOSS_LIMIT_HIT

        return VetoReason.NONE

    def _open_positions_count(self) -> int:
        return self._open_positions

    def register_position_opened(self):
        self._open_positions += 1

    def register_position_closed(self):
        self._open_positions = max(0, self._open_positions - 1)

    def calculate_position_size(self, entry_price: float, stop_price: float,
                                  live_equity_inr: float, risk_reward: float) -> PositionSizeResult:
        """
        Position size is calculated BEFORE entry, and is NEVER increased
        because AI confidence is high. Confidence affects whether to trade,
        never how much to risk.
        """
        risk_per_unit_points = abs(entry_price - stop_price)  # in points, GOLDM quoted per 10g
        point_value = self.config.instrument.point_value_inr

        # Hard cap: reject rather than mis-place the stop
        if risk_per_unit_points > self.config.risk.max_stop_distance_points:
            return PositionSizeResult(
                approved=False, lots=0, risk_amount_inr=0,
                veto_reason=VetoReason.STOP_TOO_WIDE,
                notes=f"Required stop distance {risk_per_unit_points:.1f}pts exceeds "
                      f"max {self.config.risk.max_stop_distance_points}pts. "
                      f"NOT tightening the stop artificially — rejecting the trade instead."
            )

        if risk_reward < self.config.risk.min_risk_reward:
            return PositionSizeResult(
                approved=False, lots=0, risk_amount_inr=0,
                veto_reason=VetoReason.RISK_REWARD_TOO_LOW,
                notes=f"R:R {risk_reward:.2f} below minimum {self.config.risk.min_risk_reward}"
            )

        allowed_risk = (
            self.config.risk.max_risk_per_trade_inr * self.state.current_lot_multiplier
        )
        risk_per_lot_inr = risk_per_unit_points * point_value

        if risk_per_lot_inr <= 0:
            return PositionSizeResult(
                approved=False, lots=0, risk_amount_inr=0,
                veto_reason=VetoReason.POSITION_SIZE_ROUNDS_TO_ZERO,
                notes="Zero or negative risk-per-lot — entry equals stop, invalid setup."
            )

        raw_lots = allowed_risk / risk_per_lot_inr
        lots = int(raw_lots)  # floor — never round up to force a trade
        lots = min(lots, self.config.risk.max_lots_cap)

        if lots < 1:
            return PositionSizeResult(
                approved=False, lots=0, risk_amount_inr=0,
                veto_reason=VetoReason.POSITION_SIZE_ROUNDS_TO_ZERO,
                notes=f"Computed size rounds to 0 lots (raw={raw_lots:.3f}). "
                      f"NOT rounding up to force a trade."
            )

        # equity safety check — refuse if margin+buffer exceeds live equity
        margin_estimate = self._estimate_margin(lots, entry_price=entry_price)
        safety_buffer = margin_estimate * (self.config.risk.equity_min_safety_buffer_pct / 100)
        if margin_estimate + safety_buffer > live_equity_inr:
            return PositionSizeResult(
                approved=False, lots=0, risk_amount_inr=0,
                veto_reason=VetoReason.INSUFFICIENT_EQUITY,
                notes=f"Margin ~₹{margin_estimate:.0f} + buffer exceeds live equity ₹{live_equity_inr:.0f}"
            )

        actual_risk_inr = lots * risk_per_lot_inr
        return PositionSizeResult(
            approved=True, lots=lots, risk_amount_inr=actual_risk_inr,
            notes=f"{lots} lot(s), risking ₹{actual_risk_inr:.0f} "
                  f"(multiplier={self.state.current_lot_multiplier})"
        )

    def _estimate_margin(self, lots: int, entry_price: float | None = None) -> float:
        """
        BUGFIX: was a hardcoded ₹65,000/lot, calibrated to gold's price level
        when this was first written (~₹70,000). Real 2-year MCX data showed
        gold move to ~₹152,000, at which point the hardcoded figure had
        silently drifted to ~4.3% of contract value instead of the ~9%
        it represented originally — understating true margin by roughly
        half. Now computed as a percentage of actual contract value, so it
        tracks price automatically. Falls back to the old flat estimate
        only if no entry_price is available (keeps the signature safe for
        any caller that hasn't been updated to pass it).
        """
        if entry_price is None or entry_price <= 0:
            return 65000.0 * lots  # conservative fallback, not the primary path
        contract_value = entry_price * 10  # GOLDM: quoted per 10g, 100g contract
        margin_pct = 0.09  # ~9% is representative of MCX gold futures margin; verify against
                             # the exchange's current SPAN+exposure margin before live trading
        return contract_value * margin_pct * lots

    # ---------- graduated de-risking (automatic, no confirmation needed) ----------
    def record_trade_result(self, pnl_inr: float):
        self.state.trades_taken_today += 1
        self.state.daily_pnl_inr += pnl_inr
        self.state.weekly_pnl_inr += pnl_inr

        if pnl_inr < 0:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
        else:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0

        self._apply_derisking()
        self._check_scaleup_eligibility()

    def _apply_derisking(self):
        """Automatic risk REDUCTION — no human approval needed to get safer."""
        cl = self.state.consecutive_losses
        if cl >= self.config.risk.max_consecutive_losses_before_disable:
            self.state.trading_disabled = True
            return
        if cl >= 3:
            self.state.current_lot_multiplier = self.config.risk.derisk_after_3_losses_multiplier
        elif cl >= 2:
            self.state.current_lot_multiplier = self.config.risk.derisk_after_2_losses_multiplier
        elif self.state.consecutive_wins >= self.config.risk.derisk_reset_after_n_consecutive_wins:
            self.state.current_lot_multiplier = 1.0  # restore base after recovery

    def _check_scaleup_eligibility(self):
        """Only RECOMMENDS scaling up. Never auto-applies — requires manual confirm."""
        if (self.state.consecutive_wins >= self.config.risk.scaleup_after_n_consecutive_wins
                and self.state.current_lot_multiplier >= 1.0):
            self.state.scaleup_recommended = True
        else:
            self.state.scaleup_recommended = False

    def confirm_scaleup(self, new_multiplier: float):
        """Explicit human action required to actually increase risk."""
        capped = min(new_multiplier, self.config.risk.max_lots_cap)
        self.state.current_lot_multiplier = capped
        self.state.scaleup_recommended = False

    def manual_reset(self):
        """
        Explicit reset after MAX_CONSECUTIVE_LOSSES disables trading. Per
        spec: 'Manual administrator reset required' — this method exists so
        that action is a deliberate, auditable call, never automatic in live
        or paper trading. A backtest replay may call this on a cooldown to
        simulate a realistic operator reviewing and resuming the next day —
        that simulation choice belongs in the backtest runner, not here.
        """
        self.state.trading_disabled = False
        self.state.consecutive_losses = 0
        self.state.current_lot_multiplier = 1.0
