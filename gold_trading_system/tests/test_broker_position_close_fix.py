"""
Tests for a serious real accounting bug found while manually verifying a
live trade's equity: exits NEVER called broker.place_order() — only
entries did. This meant the broker's own position bookkeeping never
actually closed, risking corrupted quantity/avg_price tracking on the
NEXT entry, and equity never properly reflected a real close.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from tests.test_backtest_runner import make_synthetic_trending_candles


def _engine():
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(Settings(), broker, symbol="GOLDM",
                                 persistence_path=None, candle_persistence_path=None,
                                 open_position_path=None)
    return engine, broker


def test_broker_position_is_flat_after_every_close():
    """THE critical check: after ANY trade closes, the broker's own
    position book must show zero open quantity — not a lingering phantom
    position that could corrupt the next entry's accounting."""
    engine, broker = _engine()
    candles = make_synthetic_trending_candles(n=2000, drift=3.0, noise=10.0, seed=17)
    for c in candles:
        engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))

    print(f"Trades closed: {len(engine.state.trade_log)}")
    if engine.state.trade_log:
        assert engine.state.open_trade_manager is None or len(broker.get_positions()) <= 1, \
            "At most one position (the currently open one, if any) should exist"
        if engine.state.open_trade_manager is None:
            assert broker.get_positions() == [], \
                "With no trade open, the broker must show ZERO open positions — " \
                "a leftover phantom position here is exactly the corruption risk found"


def test_multiple_sequential_trades_dont_corrupt_broker_quantity():
    """
    Open, close, open again, close again — the SECOND entry's quantity
    must be exactly what THAT trade opened with, not accumulated/corrupted
    by a phantom leftover from the first trade.
    """
    engine, broker = _engine()
    candles = make_synthetic_trending_candles(n=3000, drift=3.0, noise=10.0, seed=23)
    quantities_seen = []
    for c in candles:
        engine.on_tick(LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low,
                                  close=c.close, volume=c.volume))
        positions = broker.get_positions()
        if positions:
            quantities_seen.append(abs(positions[0].quantity))

    if len(engine.state.trade_log) >= 2:
        # every observed open-position quantity must match SOME actual
        # trade's lot count — never an accumulated/doubled value
        actual_lot_counts = {t["lots"] for t in engine.state.trade_log}
        for q in quantities_seen:
            assert q in actual_lot_counts, \
                f"Observed broker position quantity {q} doesn't match any real " \
                f"trade's lot count {actual_lot_counts} — possible corruption"


def test_exit_places_a_real_closing_order():
    """Directly verify a closing order is actually placed — not just that
    the end-state happens to look flat by coincidence."""
    from trade_manager.trade_manager import TradeManager, TradeManagerState
    from execution.broker_adapters.base import OrderRequest, OrderSide

    engine, broker = _engine()
    base_ts = 1735689600
    engine.on_tick(LiveTick(ts=base_ts, open=155000, high=155010, low=154990,
                              close=155000, volume=100))

    # manually open a position via a real order, exactly like a genuine entry would
    order = OrderRequest(client_order_id="TEST-ENTRY", symbol="GOLDM",
                            side=OrderSide.BUY, quantity=1)
    fill = broker.place_order(order)
    assert len(broker.get_positions()) == 1

    tm_state = TradeManagerState(direction="LONG", entry_price=fill.filled_price,
                                    original_stop=fill.filled_price - 200,
                                    current_stop=fill.filled_price - 200,
                                    original_risk_points=200.0,
                                    target_1=fill.filled_price + 100,
                                    target_2=fill.filled_price + 200,
                                    target_3=fill.filled_price + 300)
    engine.state.open_trade_manager = TradeManager(engine.config, tm_state)
    engine.state.open_trade_lots = 1
    engine.risk_engine.register_position_opened()

    orders_before = len(broker.get_orders())

    # force a stop-loss hit
    engine.on_tick(LiveTick(ts=base_ts + 300, open=154900, high=154910,
                              low=154750, close=154800, volume=100))

    orders_after = len(broker.get_orders())
    assert orders_after > orders_before, \
        "A real closing order must be placed with the broker when a trade exits"
    assert broker.get_positions() == [], \
        "The position must be fully closed in the broker's own books"


if __name__ == "__main__":
    tests = [
        test_broker_position_is_flat_after_every_close,
        test_multiple_sequential_trades_dont_corrupt_broker_quantity,
        test_exit_places_a_real_closing_order,
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
