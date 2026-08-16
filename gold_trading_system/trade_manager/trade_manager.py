"""
Trade Manager — the state machine + trade management logic (Sprint 1 §13-16).
Ties together: partial profit booking, momentum-aware trailing, and
immediate exit on structure break or momentum decay.

Hard rule enforced throughout: stop only tightens or holds, NEVER widens.
"""
from dataclasses import dataclass, field
from enum import Enum


class TradeState(str, Enum):
    NEW_TRADE = "NEW_TRADE"
    RISK_ACTIVE = "RISK_ACTIVE"
    PROFITABLE = "PROFITABLE"
    TARGET_1_REACHED = "TARGET_1_REACHED"
    BREAKEVEN_PROTECTED = "BREAKEVEN_PROTECTED"
    TARGET_2_REACHED = "TARGET_2_REACHED"
    TRAILING_RUNNER = "TRAILING_RUNNER"
    EXITED = "EXITED"


class ExitReason(str, Enum):
    NONE = "NONE"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    ALL_TARGETS_HIT = "ALL_TARGETS_HIT"
    MOMENTUM_DECAY = "MOMENTUM_DECAY"
    STRUCTURE_BREAK = "STRUCTURE_BREAK"
    MANUAL = "MANUAL"


class TrailMethod(str, Enum):
    EMA9 = "EMA9"
    EMA21 = "EMA21"
    EMA50 = "EMA50"
    ATR = "ATR"
    STRUCTURE = "STRUCTURE"


@dataclass
class StateTransition:
    from_state: TradeState
    to_state: TradeState
    trigger_reason: str
    price_at_event: float


@dataclass
class TrailUpdate:
    old_stop: float
    new_stop: float
    method_used: str
    reason: str


@dataclass
class TradeManagerState:
    trade_state: TradeState = TradeState.NEW_TRADE
    direction: str = "LONG"          # LONG or SHORT
    entry_price: float = 0.0
    original_stop: float = 0.0
    current_stop: float = 0.0
    original_risk_points: float = 0.0

    quantity_remaining_pct: float = 100.0
    booked_at_1R: bool = False
    booked_at_t1: bool = False
    booked_at_t2: bool = False

    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0

    state_history: list = field(default_factory=list)
    trail_history: list = field(default_factory=list)
    exit_reason: ExitReason = ExitReason.NONE

    # BUGFIX: track R actually banked on each partial so final trade R reflects
    # the blended result, not just where the residual runner happened to exit.
    # Each entry is (pct_of_position_closed, r_multiple_at_that_close).
    realized_legs: list = field(default_factory=list)


class TradeManager:
    def __init__(self, config, state: TradeManagerState, min_stop_cushion_atr_mult: float = 0.25):
        self.config = config
        self.state = state
        # minimum distance (in ATR multiples) a trailing stop must stay away
        # from current price — prevents placing a stop at/through the market
        self.min_stop_cushion_atr_mult = min_stop_cushion_atr_mult

    # ---------- state transitions ----------
    def _transition(self, new_state: TradeState, reason: str, price: float):
        if new_state == self.state.trade_state:
            return
        self.state.state_history.append(StateTransition(
            from_state=self.state.trade_state, to_state=new_state,
            trigger_reason=reason, price_at_event=price,
        ))
        self.state.trade_state = new_state

    def _r_multiple(self, current_price: float) -> float:
        if self.state.original_risk_points <= 0:
            return 0.0
        move = (current_price - self.state.entry_price if self.state.direction == "LONG"
                else self.state.entry_price - current_price)
        return move / self.state.original_risk_points

    def blended_r_multiple(self, exit_price: float) -> float:
        """
        BUGFIX: the true R of the trade is the position-weighted blend of every
        partial booked along the way PLUS the residual runner at exit_price.
        Using only _r_multiple(exit_price) reports a runner that round-tripped
        to breakeven as 0.0R even when half the position was banked at +1R/+2R.
        """
        total = 0.0
        for pct, r_at_leg in self.state.realized_legs:
            total += (pct / 100.0) * r_at_leg
        residual_pct = max(0.0, self.state.quantity_remaining_pct)
        total += (residual_pct / 100.0) * self._r_multiple(exit_price)
        return total

    # ---------- partial profit booking ----------
    def check_partial_booking(self, current_price: float) -> list[str]:
        """Returns list of booking actions taken this update, for logging."""
        actions = []
        if self.state.trade_state == TradeState.EXITED:
            return actions

        r = self._r_multiple(current_price)

        if r >= 1.0 and not self.state.booked_at_1R:
            self.state.booked_at_1R = True
            self.state.quantity_remaining_pct -= self.config.partial_booking.at_1R_pct
            self.state.realized_legs.append(
                (self.config.partial_booking.at_1R_pct, self._r_multiple(current_price)))
            self._transition(TradeState.PROFITABLE, "Reached +1R", current_price)
            actions.append(f"Booked {self.config.partial_booking.at_1R_pct}% at +1R")

        target_hit = (
            current_price >= self.state.target_1 if self.state.direction == "LONG"
            else current_price <= self.state.target_1
        )
        if target_hit and not self.state.booked_at_t1:
            self.state.booked_at_t1 = True
            self.state.quantity_remaining_pct -= self.config.partial_booking.at_target1_pct
            self.state.realized_legs.append(
                (self.config.partial_booking.at_target1_pct, self._r_multiple(current_price)))
            self._transition(TradeState.TARGET_1_REACHED, "Reached Target 1", current_price)
            actions.append(f"Booked {self.config.partial_booking.at_target1_pct}% at Target 1")

            if self.config.partial_booking.move_sl_to_breakeven_at_target1:
                self._move_to_breakeven(current_price)

        target2_hit = (
            current_price >= self.state.target_2 if self.state.direction == "LONG"
            else current_price <= self.state.target_2
        )
        if target2_hit and not self.state.booked_at_t2 and self.state.booked_at_t1:
            self.state.booked_at_t2 = True
            self.state.quantity_remaining_pct -= self.config.partial_booking.at_target2_pct
            self.state.realized_legs.append(
                (self.config.partial_booking.at_target2_pct, self._r_multiple(current_price)))
            self._transition(TradeState.TARGET_2_REACHED, "Reached Target 2", current_price)
            self._transition(TradeState.TRAILING_RUNNER, "Runner active", current_price)
            actions.append(f"Booked {self.config.partial_booking.at_target2_pct}% at Target 2, "
                            f"runner now trailing")

        return actions

    def _move_to_breakeven(self, current_price: float):
        """Only moves stop TOWARD reduced risk — enforces the never-widen rule."""
        new_stop = self.state.entry_price
        if self._is_stop_improvement(new_stop):
            self._apply_stop_update(new_stop, TrailMethod.STRUCTURE.value,
                                       "Moved to breakeven after Target 1")
            self._transition(TradeState.BREAKEVEN_PROTECTED, "Breakeven set", current_price)

    # ---------- trailing ----------
    def _is_stop_improvement(self, new_stop: float) -> bool:
        """
        HARD RULE: stop only tightens or holds, never widens original risk.
        For LONG: new stop must be higher than current. For SHORT: lower.
        """
        if self.state.direction == "LONG":
            return new_stop > self.state.current_stop
        return new_stop < self.state.current_stop

    def _apply_stop_update(self, new_stop: float, method: str, reason: str):
        if not self._is_stop_improvement(new_stop):
            return  # silently reject — the hard rule is non-negotiable, not a warning
        old_stop = self.state.current_stop
        self.state.current_stop = new_stop
        self.state.trail_history.append(TrailUpdate(
            old_stop=old_stop, new_stop=new_stop, method_used=method, reason=reason,
        ))

    def update_trailing_stop(self, current_price: float, ema9: float, ema21: float,
                              ema50: float, atr: float, momentum_health: str,
                              structure_broke_against: bool) -> TrailUpdate | None:
        """
        Called every candle on an open runner. Selects trail method by regime,
        tightens on momentum decay, exits immediately on structure break —
        never widens the stop under any circumstance.
        """
        if self.state.trade_state == TradeState.EXITED:
            return None

        if structure_broke_against:
            self.close_trade(current_price, ExitReason.STRUCTURE_BREAK)
            return None

        if momentum_health == "DEAD":
            self.close_trade(current_price, ExitReason.MOMENTUM_DECAY)
            return None

        # select trail method by momentum regime
        if momentum_health == "STRONG":
            candidate_stop, method = ema9, TrailMethod.EMA9.value
        elif momentum_health == "WEAKENING":
            # tighten: use the closer/tighter of EMA21 or an ATR-based stop
            atr_stop = (current_price - atr * self.config.trailing.atr_multiplier
                        if self.state.direction == "LONG"
                        else current_price + atr * self.config.trailing.atr_multiplier)
            if self.state.direction == "LONG":
                candidate_stop = max(ema21, atr_stop)
            else:
                candidate_stop = min(ema21, atr_stop)
            method = TrailMethod.ATR.value
        else:
            candidate_stop, method = ema50, TrailMethod.EMA50.value

        # BUGFIX: a trailing stop must NEVER sit beyond the current price.
        # A reference line (e.g. EMA9) can be above price during a pullback;
        # using it raw would place the stop above market for a LONG, which a
        # real broker would reject and which silently self-triggers in backtest.
        # Clamp to a small ATR-based cushion on the correct side of price.
        cushion = atr * self.min_stop_cushion_atr_mult
        if self.state.direction == "LONG":
            max_allowed_stop = current_price - cushion
            candidate_stop = min(candidate_stop, max_allowed_stop)
        else:
            min_allowed_stop = current_price + cushion
            candidate_stop = max(candidate_stop, min_allowed_stop)

        before = self.state.current_stop
        self._apply_stop_update(candidate_stop, method,
                                   f"Momentum={momentum_health}, trailing via {method}")
        if self.state.current_stop != before:
            return self.state.trail_history[-1]
        return None

    # ---------- exit ----------
    def check_stop_hit(self, current_price: float) -> bool:
        hit = (
            current_price <= self.state.current_stop if self.state.direction == "LONG"
            else current_price >= self.state.current_stop
        )
        if hit:
            self.close_trade(current_price, ExitReason.STOP_LOSS_HIT)
        return hit

    def check_stop_hit_intrabar(self, high: float, low: float, close: float) -> bool:
        """
        BUGFIX: checking only the close misses candles that traded THROUGH the
        stop intrabar and recovered — which systematically understates losses.
        A real stop order fills when price touches it, not at the candle close,
        so the fill is recorded at the stop price itself (conservative; ignores
        slippage/gaps, which the paper broker models separately).
        """
        if self.state.direction == "LONG":
            hit = low <= self.state.current_stop
        else:
            hit = high >= self.state.current_stop

        if hit:
            # gap protection: if the candle OPENED beyond the stop, a real fill
            # would be at/near the open, not the stop price — take the worse of
            # the two rather than optimistically assuming a clean stop fill
            fill_price = self.state.current_stop
            if self.state.direction == "LONG" and high < self.state.current_stop:
                fill_price = high   # entire candle gapped below the stop
            elif self.state.direction == "SHORT" and low > self.state.current_stop:
                fill_price = low    # entire candle gapped above the stop
            self.close_trade(fill_price, ExitReason.STOP_LOSS_HIT)
            return True
        return False

    def close_trade(self, exit_price: float, reason: ExitReason):
        self.state.exit_reason = reason
        self._transition(TradeState.EXITED, reason.value, exit_price)
