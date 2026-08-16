"""
Broker abstraction layer (Sprint 1, Section 24).
Strategy modules NEVER talk to a broker directly — everything goes through
this interface. This is what lets us build and test the entire system now,
and drop in real Angel One credentials later without touching anything else.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PLACED = "PLACED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Quote:
    symbol: str
    ltp: float           # last traded price
    bid: float
    ask: float
    volume: float
    timestamp: float
    is_stale: bool = False


@dataclass
class OrderRequest:
    client_order_id: str   # idempotency key — caller generates a unique one per intent
    symbol: str
    side: OrderSide
    quantity: int           # in lots
    order_type: str = "MARKET"
    price: float | None = None


@dataclass
class OrderResult:
    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    filled_price: float | None = None
    filled_quantity: int = 0
    message: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int    # signed: positive = long, negative = short
    avg_price: float
    unrealized_pnl: float


@dataclass
class AccountBalance:
    equity_inr: float
    margin_used_inr: float
    margin_available_inr: float


class BrokerProvider(ABC):
    """Every broker adapter (Angel One, Zerodha, mock/paper) implements this."""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_orders(self) -> list[OrderResult]: ...

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def modify_order(self, client_order_id: str, new_price: float | None = None,
                      new_quantity: int | None = None) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> OrderResult: ...

    @abstractmethod
    def get_order_status(self, client_order_id: str) -> OrderStatus: ...

    @abstractmethod
    def get_balance(self) -> AccountBalance: ...
