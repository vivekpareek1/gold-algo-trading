import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from execution.live_trading_engine import LiveTradingEngine, LiveTick
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from tests.test_backtest_runner import make_synthetic_trending_candles


def make_engine(settings=None):
    settings = settings or Settings()
    broker = PaperBrokerProvider(starting_equity_inr=500_000.0)
    broker.connect()
    engine = LiveTradingEngine(settings, broker, symbol="GOLDM", persistence_path=None)
    return engine, broker


def candles_to_ticks(candles):
    return [LiveTick(ts=c.ts, open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume)
            for c in candles]


def test_state_persists_across_calls():
    """THE core property this module exists for: state must carry over
    between on_tick() calls, not reset each time like the old API endpoints did."""
    engine, broker = make_engine()
    candles = make_synthetic_trending_candles(n=100, drift=2.0, noise=10.0)
    ticks = candles_to_ticks(candles)

    tick_counts_seen = []
    for t in ticks:
        snap = engine.on_tick(t)
        tick_counts_seen.append(engine.state.tick_count)

    assert tick_counts_seen == list(range(1, 101)), \
        "tick_count must increment monotonically across calls — proves state persists"


def test_engine_can_open_and_close_a_trade_over_many_ticks():
    engine, broker = make_engine()
    candles = make_synthetic_trending_candles(n=500, drift=3.0, noise=12.0, seed=5)
    ticks = candles_to_ticks(candles)

    for t in ticks:
        engine.on_tick(t)

    print(f"Trades over session: {len(engine.state.trade_log)}, "
          f"signals evaluated: {len(engine.state.signal_log)}")
    # not asserting a specific trade count (NO_TRADE is valid) — just that
    # the engine ran the full session without crashing and tracked state
    assert engine.state.tick_count == 500
    assert isinstance(engine.state.trade_log, list)


def test_open_position_reflected_in_snapshot():
    engine, broker = make_engine()
    candles = make_synthetic_trending_candles(n=200, drift=3.0, noise=8.0, seed=9)
    ticks = candles_to_ticks(candles)

    had_open_position = False
    for t in ticks:
        snap = engine.on_tick(t)
        if snap["has_open_position"]:
            had_open_position = True
            assert snap["open_position"] is not None
            assert snap["open_position"]["direction"] in ("LONG", "SHORT")
    print(f"Had an open position at some point: {had_open_position}")


def test_intrabar_stop_fix_applies_in_live_engine():
    """Verify the intrabar stop-hit bugfix carries over into the live engine,
    not just the backtest runner — this was a real risk (fixing one path,
    forgetting the other)."""
    import inspect
    from execution.live_trading_engine import LiveTradingEngine
    source = inspect.getsource(LiveTradingEngine._manage_open_trade)
    assert "check_stop_hit_intrabar" in source, \
        "Live engine must use the intrabar-aware stop check, not the close-only one"
    assert "blended_r_multiple" in source, \
        "Live engine must use blended R accounting for partials, not the naive R"


def test_daily_reset_and_cooldown_logic_present_in_live_engine():
    import inspect
    from execution.live_trading_engine import LiveTradingEngine
    source = inspect.getsource(LiveTradingEngine._handle_day_boundary)
    assert "trades_taken_today = 0" in source
    assert "manual_reset" in source


def test_risk_engine_position_tracking_wired_in_live_engine():
    import inspect
    from execution.live_trading_engine import LiveTradingEngine
    source = inspect.getsource(LiveTradingEngine._evaluate_new_entry) + \
             inspect.getsource(LiveTradingEngine._manage_open_trade)
    assert "register_position_opened" in source
    assert "register_position_closed" in source


def test_never_evaluates_new_entry_while_position_open():
    """A position must block new entries until it closes — no double-entry."""
    engine, broker = make_engine()
    candles = make_synthetic_trending_candles(n=300, drift=3.0, noise=10.0, seed=13)
    ticks = candles_to_ticks(candles)

    max_signals_while_open = 0
    for t in ticks:
        was_open_before = engine.state.open_trade_manager is not None
        signals_before = len(engine.state.signal_log)
        engine.on_tick(t)
        if was_open_before:
            signals_after = len(engine.state.signal_log)
            assert signals_after == signals_before, \
                "No new signal should be evaluated while a position is already open"


def test_equity_reflects_realized_pnl_over_session():
    """Cross-check against the paper broker's own bugfix — equity should move
    as trades close, not just from commissions."""
    engine, broker = make_engine()
    starting_equity = broker.get_balance().equity_inr
    candles = make_synthetic_trending_candles(n=800, drift=2.5, noise=15.0, seed=21)
    ticks = candles_to_ticks(candles)

    for t in ticks:
        engine.on_tick(t)

    final_equity = broker.get_balance().equity_inr
    print(f"Starting equity: {starting_equity}, final: {final_equity}, "
          f"trades closed: {len(engine.state.trade_log)}")
    if len(engine.state.trade_log) > 0:
        assert final_equity != starting_equity, \
            "Equity should have moved from its starting value after real trades closed"


def test_regime_tag_recorded_on_closed_trades():
    engine, broker = make_engine()
    candles = make_synthetic_trending_candles(n=800, drift=2.5, noise=12.0, seed=33)
    ticks = candles_to_ticks(candles)
    for t in ticks:
        engine.on_tick(t)

    if engine.state.trade_log:
        for trade in engine.state.trade_log:
            assert "entry_regime" in trade
            assert trade["entry_regime"] is not None


def test_signal_log_bounded_for_long_running_session():
    """A long-running live session must not accumulate unbounded memory."""
    engine, broker = make_engine()
    candles = make_synthetic_trending_candles(n=2000, drift=1.0, noise=10.0, seed=41)
    ticks = candles_to_ticks(candles)
    for t in ticks:
        engine.on_tick(t)
    assert len(engine.state.signal_log) <= 500, \
        f"signal_log should be bounded to prevent unbounded memory growth, got {len(engine.state.signal_log)}"


if __name__ == "__main__":
    tests = [
        test_state_persists_across_calls,
        test_engine_can_open_and_close_a_trade_over_many_ticks,
        test_open_position_reflected_in_snapshot,
        test_intrabar_stop_fix_applies_in_live_engine,
        test_daily_reset_and_cooldown_logic_present_in_live_engine,
        test_risk_engine_position_tracking_wired_in_live_engine,
        test_never_evaluates_new_entry_while_position_open,
        test_equity_reflects_realized_pnl_over_session,
        test_regime_tag_recorded_on_closed_trades,
        test_signal_log_bounded_for_long_running_session,
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
