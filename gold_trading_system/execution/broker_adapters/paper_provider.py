"""
Paper Trading Provider — the default execution mode (Sprint 1, Section 26).
Simulates realistic commission, fees, slippage, and spread so paper results
are meaningful, not an idealized best-case. This is what everything gets
built and tested against until real Angel One credentials are wired in.
"""
import time
import uuid

from execution.broker_adapters.base import (
    BrokerProvider, Quote, OrderRequest, OrderResult, OrderSide, OrderStatus,
    Position, AccountBalance
)


class PaperBrokerProvider(BrokerProvider):
    def __init__(self, starting_equity_inr: float = 500_000.0,
                 slippage_points: float = 1.0,
                 commission_per_lot_inr: float = 40.0,
                 spread_points: float = 0.5,
                 point_value_inr: float = 10.0):
        self._connected = False
        self._equity = starting_equity_inr
        self._margin_used = 0.0
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}   # keyed by client_order_id — idempotency
        self._quotes: dict[str, Quote] = {}          # test/simulation feeds this
        self.slippage_points = slippage_points
        self.commission_per_lot_inr = commission_per_lot_inr
        self.spread_points = spread_points
        self.point_value_inr = point_value_inr   # GOLDM: ₹10 per point per lot

    # ---------- test/simulation helper, not part of the broker interface ----------
    def set_quote(self, symbol: str, ltp: float, volume: float = 1000.0,
                   stale: bool = False):
        """Feeds simulated market data in — a real adapter would get this from a WebSocket."""
        self._quotes[symbol] = Quote(
            symbol=symbol, ltp=ltp,
            bid=ltp - self.spread_points / 2,
            ask=ltp + self.spread_points / 2,
            volume=volume, timestamp=time.time(), is_stale=stale,
        )

    # ---------- BrokerProvider interface ----------
    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def get_quote(self, symbol: str) -> Quote:
        if symbol not in self._quotes:
            raise ValueError(f"No simulated quote set for {symbol} — call set_quote() first")
        return self._quotes[symbol]

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_orders(self) -> list[OrderResult]:
        return list(self._orders.values())

    def place_order(self, order: OrderRequest) -> OrderResult:
        # idempotency guard — mirrors the DB unique constraint on client_order_id.
        # A real broker retry after a network timeout must NEVER double-fill.
        if order.client_order_id in self._orders:
            existing = self._orders[order.client_order_id]
            return OrderResult(
                client_order_id=order.client_order_id,
                broker_order_id=existing.broker_order_id,
                status=existing.status,
                filled_price=existing.filled_price,
                filled_quantity=existing.filled_quantity,
                message="Duplicate client_order_id — returning original result, not re-executing.",
            )

        quote = self._quotes.get(order.symbol)
        if quote is None or quote.is_stale:
            result = OrderResult(
                client_order_id=order.client_order_id, broker_order_id=None,
                status=OrderStatus.REJECTED,
                message="No live/fresh quote available — order rejected (fail-safe).",
            )
            self._orders[order.client_order_id] = result
            return result

        # simulate slippage: fills worse than mid-price, in the direction that hurts you
        if order.side == OrderSide.BUY:
            fill_price = quote.ask + self.slippage_points
        else:
            fill_price = quote.bid - self.slippage_points

        broker_order_id = f"PAPER-{uuid.uuid4().hex[:10]}"
        result = OrderResult(
            client_order_id=order.client_order_id,
            broker_order_id=broker_order_id,
            status=OrderStatus.FILLED,
            filled_price=fill_price,
            filled_quantity=order.quantity,
            message="Paper fill simulated with slippage + spread.",
        )
        self._orders[order.client_order_id] = result
        self._apply_fill_to_position(order, fill_price)
        return result

    def _apply_fill_to_position(self, order: OrderRequest, fill_price: float):
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        existing = self._positions.get(order.symbol)

        commission = self.commission_per_lot_inr * order.quantity
        self._equity -= commission

        if existing is None:
            self._positions[order.symbol] = Position(
                symbol=order.symbol, quantity=signed_qty,
                avg_price=fill_price, unrealized_pnl=0.0,
            )
        else:
            new_qty = existing.quantity + signed_qty
            # BUGFIX: realized P&L must hit equity, not just commissions.
            # Any fill that reduces or flips the position closes some quantity,
            # and that closed portion's profit/loss is realized here.
            closing_qty = 0
            if existing.quantity > 0 and signed_qty < 0:
                closing_qty = min(existing.quantity, -signed_qty)
            elif existing.quantity < 0 and signed_qty > 0:
                closing_qty = min(-existing.quantity, signed_qty)

            if closing_qty > 0:
                direction = 1 if existing.quantity > 0 else -1
                points = (fill_price - existing.avg_price) * direction
                realized_pnl = points * self.point_value_inr * closing_qty
                self._equity += realized_pnl

            if new_qty == 0:
                del self._positions[order.symbol]
            else:
                existing.quantity = new_qty
                # only reset avg price when the position flips direction;
                # adding to an existing position should not discard the old basis
                if (existing.quantity > 0) != (new_qty > 0):
                    existing.avg_price = fill_price

    def modify_order(self, client_order_id: str, new_price: float | None = None,
                      new_quantity: int | None = None) -> OrderResult:
        if client_order_id not in self._orders:
            return OrderResult(client_order_id=client_order_id, broker_order_id=None,
                                status=OrderStatus.REJECTED, message="Order not found.")
        existing = self._orders[client_order_id]
        if existing.status == OrderStatus.FILLED:
            return OrderResult(client_order_id=client_order_id,
                                broker_order_id=existing.broker_order_id,
                                status=existing.status,
                                message="Cannot modify a filled order.")
        return existing

    def cancel_order(self, client_order_id: str) -> OrderResult:
        if client_order_id not in self._orders:
            return OrderResult(client_order_id=client_order_id, broker_order_id=None,
                                status=OrderStatus.REJECTED, message="Order not found.")
        existing = self._orders[client_order_id]
        if existing.status == OrderStatus.FILLED:
            return OrderResult(client_order_id=client_order_id,
                                broker_order_id=existing.broker_order_id,
                                status=existing.status, message="Cannot cancel a filled order.")
        existing.status = OrderStatus.CANCELLED
        return existing

    def get_order_status(self, client_order_id: str) -> OrderStatus:
        if client_order_id not in self._orders:
            raise ValueError(f"Unknown client_order_id: {client_order_id}")
        return self._orders[client_order_id].status

    def get_balance(self) -> AccountBalance:
        return AccountBalance(
            equity_inr=self._equity,
            margin_used_inr=self._margin_used,
            margin_available_inr=self._equity - self._margin_used,
        )
