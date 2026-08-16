import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from execution.broker_adapters.paper_provider import PaperBrokerProvider
from execution.broker_adapters.base import OrderRequest, OrderSide, OrderStatus


def fresh_broker():
    b = PaperBrokerProvider(starting_equity_inr=500_000.0)
    b.connect()
    return b


def test_connect_and_balance():
    b = fresh_broker()
    bal = b.get_balance()
    assert bal.equity_inr == 500_000.0
    assert bal.margin_available_inr == 500_000.0


def test_order_rejected_without_quote():
    """Must fail-safe: no quote set at all -> reject, not silently execute."""
    b = fresh_broker()
    order = OrderRequest(client_order_id="TEST-1", symbol="GOLDM",
                          side=OrderSide.BUY, quantity=1)
    result = b.place_order(order)
    assert result.status == OrderStatus.REJECTED
    print(f"No-quote rejection: {result.message}")


def test_order_rejected_on_stale_quote():
    b = fresh_broker()
    b.set_quote("GOLDM", ltp=63000, stale=True)
    order = OrderRequest(client_order_id="TEST-2", symbol="GOLDM",
                          side=OrderSide.BUY, quantity=1)
    result = b.place_order(order)
    assert result.status == OrderStatus.REJECTED
    assert "stale" in result.message.lower() or "fresh" in result.message.lower()


def test_normal_fill_applies_slippage():
    b = fresh_broker()
    b.set_quote("GOLDM", ltp=63000)  # spread 0.5, so ask=63000.25
    order = OrderRequest(client_order_id="TEST-3", symbol="GOLDM",
                          side=OrderSide.BUY, quantity=1)
    result = b.place_order(order)
    print(f"Fill: {result.filled_price} (raw ltp was 63000)")
    assert result.status == OrderStatus.FILLED
    # BUY should fill worse than raw ltp (ask + slippage)
    assert result.filled_price > 63000, \
        f"BUY fill should be worse (higher) than raw price due to slippage+spread, got {result.filled_price}"


def test_duplicate_client_order_id_does_not_double_fill():
    """THE critical safety test — network retry must never cause a double trade."""
    b = fresh_broker()
    b.set_quote("GOLDM", ltp=63000)
    order = OrderRequest(client_order_id="DUPTEST-1", symbol="GOLDM",
                          side=OrderSide.BUY, quantity=1)

    result1 = b.place_order(order)
    result2 = b.place_order(order)  # simulate a retry with the SAME client_order_id

    print(f"First fill: {result1.filled_price}, Second call message: {result2.message}")
    assert result1.broker_order_id == result2.broker_order_id, \
        "Retry with same client_order_id must return the SAME broker order, not create a new one"

    positions = b.get_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 1, \
        f"Expected exactly 1 lot after duplicate retry, got {positions[0].quantity} — DOUBLE FILL BUG"


def test_position_updates_correctly_on_fill():
    b = fresh_broker()
    b.set_quote("GOLDM", ltp=63000)
    order = OrderRequest(client_order_id="TEST-4", symbol="GOLDM",
                          side=OrderSide.BUY, quantity=2)
    b.place_order(order)
    positions = b.get_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 2


def test_commission_deducted_from_equity():
    b = fresh_broker()
    starting = b.get_balance().equity_inr
    b.set_quote("GOLDM", ltp=63000)
    order = OrderRequest(client_order_id="TEST-5", symbol="GOLDM",
                          side=OrderSide.BUY, quantity=1)
    b.place_order(order)
    after = b.get_balance().equity_inr
    assert after < starting, "Commission should reduce equity after a fill"
    print(f"Equity before: {starting}, after: {after}, commission deducted: {starting - after}")


def test_cannot_cancel_filled_order():
    b = fresh_broker()
    b.set_quote("GOLDM", ltp=63000)
    order = OrderRequest(client_order_id="TEST-6", symbol="GOLDM",
                          side=OrderSide.BUY, quantity=1)
    b.place_order(order)
    result = b.cancel_order("TEST-6")
    assert result.status == OrderStatus.FILLED, "Filled order must not be cancellable"


def test_sell_fills_worse_direction():
    """SELL should fill at bid - slippage (worse for the seller), not better."""
    b = fresh_broker()
    b.set_quote("GOLDM", ltp=63000)
    order = OrderRequest(client_order_id="TEST-7", symbol="GOLDM",
                          side=OrderSide.SELL, quantity=1)
    result = b.place_order(order)
    assert result.filled_price < 63000, \
        f"SELL fill should be worse (lower) than raw price, got {result.filled_price}"


if __name__ == "__main__":
    tests = [
        test_connect_and_balance,
        test_order_rejected_without_quote,
        test_order_rejected_on_stale_quote,
        test_normal_fill_applies_slippage,
        test_duplicate_client_order_id_does_not_double_fill,
        test_position_updates_correctly_on_fill,
        test_commission_deducted_from_equity,
        test_cannot_cancel_filled_order,
        test_sell_fills_worse_direction,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__} -> {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
