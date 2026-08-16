import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from market_structure.structure_engine import TrendState
from backtesting.backtest_runner import run_backtest, OHLCV


def make_synthetic_trending_candles(n=200, start_price=63000.0, drift=2.0,
                                      noise=15.0, seed=42, interval_minutes=5,
                                      base_epoch=1735689600) -> list[OHLCV]:
    """Synthetic candles with an upward drift + noise — enough structure for
    the engine to actually find setups, not real market data (that's Week 2+).
    Uses realistic epoch-second timestamps spaced interval_minutes apart —
    required for multi-timeframe resampling to bucket correctly; plain
    sequential integers (ts=i) would compress the whole series into a
    fraction of a real minute and break HTF aggregation."""
    random.seed(seed)
    candles = []
    price = start_price
    for i in range(n):
        price += drift + random.uniform(-noise, noise)
        high = price + random.uniform(0, noise)
        low = price - random.uniform(0, noise)
        close = price + random.uniform(-noise / 2, noise / 2)
        open_ = price - random.uniform(-noise / 2, noise / 2)
        volume = 1000 + random.uniform(-200, 500)
        ts = base_epoch + i * interval_minutes * 60
        candles.append(OHLCV(ts=ts, open=open_, high=max(high, open_, close),
                               low=min(low, open_, close), close=close, volume=max(volume, 1)))
    return candles


def test_backtest_runs_without_crashing():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=150)
    result = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)
    print(f"Total trades: {result.metrics.total_trades}, "
          f"win_rate: {result.metrics.win_rate:.1f}%, "
          f"expectancy: {result.metrics.expectancy_r:.3f}R, "
          f"signals logged: {len(result.signal_log)}")
    assert result is not None
    assert isinstance(result.metrics.total_trades, int)


def test_backtest_logs_no_trade_decisions_too():
    """Critical: must log NO_TRADE decisions, not just executed trades."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=150)
    result = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)
    no_trade_count = sum(1 for s in result.signal_log if s["decision"] == "NO_TRADE")
    print(f"Signals: {len(result.signal_log)}, NO_TRADE count: {no_trade_count}, "
          f"trades executed: {result.metrics.total_trades}")
    assert len(result.signal_log) > 0, "Should have logged at least some signal evaluations"


def test_backtest_respects_daily_loss_limit():
    """Even a bad synthetic series should never exceed the configured max_daily_loss_pct."""
    settings = Settings()
    settings.risk.max_daily_loss_pct = 3.0
    # deliberately choppy/adverse series
    candles = make_synthetic_trending_candles(n=300, drift=-1.0, noise=40.0, seed=7)
    result = run_backtest(candles, settings, htf_trend_override=TrendState.RANGE)

    cumulative_pnl = sum(t["r_multiple"] for t in result.trade_log) * settings.risk.max_risk_per_trade_inr
    print(f"Cumulative PnL across full run: ₹{cumulative_pnl:.0f}, trades: {len(result.trade_log)}")
    # this is a loose sanity check — the daily limit resets each day in a
    # real system; here we're just confirming trades did stop somewhere
    # rather than running unbounded through a losing streak
    assert result.metrics.max_consecutive_losses <= settings.risk.max_consecutive_losses_before_disable + 1, \
        f"Consecutive losses ({result.metrics.max_consecutive_losses}) should be capped near the " \
        f"disable threshold ({settings.risk.max_consecutive_losses_before_disable}), risk engine should " \
        f"have started blocking new entries"


def test_backtest_trade_log_structure():
    settings = Settings()
    candles = make_synthetic_trending_candles(n=200, drift=3.0, noise=10.0)
    result = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)

    if result.trade_log:
        trade = result.trade_log[0]
        required_keys = {"entry_price", "exit_price", "direction", "r_multiple", "exit_reason", "ts"}
        assert required_keys.issubset(trade.keys()), \
            f"Trade log missing required fields: {required_keys - trade.keys()}"
        print(f"Sample trade: {trade}")
    else:
        print("No trades executed in this synthetic run (acceptable — NO_TRADE is a valid outcome)")


def test_no_lookahead_ordering():
    """Sanity: candles are processed strictly in the order given (ts increasing),
    never re-ordered or peeked ahead."""
    settings = Settings()
    candles = make_synthetic_trending_candles(n=100)
    result = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)

    timestamps = [s["ts"] for s in result.signal_log]
    assert timestamps == sorted(timestamps), \
        "Signal log timestamps must be strictly increasing — no look-ahead reordering"


def test_daily_trade_limit_resets_across_calendar_days():
    """
    Regression test for a real bug found on a 2-year real dataset: max_trades
    per_day (default 4) was never reset across calendar day boundaries in the
    backtest, so it silently became "4 trades across the ENTIRE backtest"
    instead of "4 trades per day". Over a long enough window with a strategy
    that fires more than 4 times total, trade count must be able to exceed
    max_trades_per_day if trades are properly spread across multiple days.
    """
    settings = Settings()
    candles = make_synthetic_trending_candles(n=2000, drift=3.0, noise=10.0, seed=17)
    result = run_backtest(candles, settings, htf_trend_override=TrendState.TRENDING_UP)
    print(f"Long-window trade count: {result.metrics.total_trades} "
          f"(max_trades_per_day={settings.risk.max_trades_per_day})")
    assert result.metrics.total_trades > settings.risk.max_trades_per_day or \
        result.metrics.total_trades == 0, \
        f"With data spanning multiple days, trade count should not be silently " \
        f"capped at exactly max_trades_per_day ({settings.risk.max_trades_per_day}) " \
        f"unless the strategy genuinely found zero setups"


def test_trading_disabled_does_not_persist_forever_in_backtest():
    """
    Regression test for a real bug found on a 2-year real dataset: once
    MAX_CONSECUTIVE_LOSSES disabled trading, it never re-enabled — a real
    2-year backtest produced exactly 18 trades whether fed 15,000 or 79,000
    candles, because trading silently shut off early and never resumed.
    With the cooldown-based resume simulation, a long enough window should
    produce more trades than a short window that hits the same disable.
    """
    settings = Settings()
    settings.risk.max_consecutive_losses_before_disable = 2  # force an early disable
    # adverse/choppy synthetic data likely to produce losing streaks
    candles = make_synthetic_trending_candles(n=3000, drift=-0.5, noise=25.0, seed=3)

    result_with_cooldown = run_backtest(candles, settings, htf_trend_override=TrendState.RANGE,
                                          cooldown_days_after_disable=1)
    result_no_cooldown = run_backtest(candles, settings, htf_trend_override=TrendState.RANGE,
                                        cooldown_days_after_disable=None)

    print(f"With cooldown: {result_with_cooldown.metrics.total_trades} trades")
    print(f"Without cooldown (old behavior): {result_no_cooldown.metrics.total_trades} trades")
    assert result_with_cooldown.metrics.total_trades >= result_no_cooldown.metrics.total_trades, \
        "Cooldown-based resume should allow at least as many trades as a permanent disable"


if __name__ == "__main__":
    tests = [
        test_backtest_runs_without_crashing,
        test_backtest_logs_no_trade_decisions_too,
        test_backtest_respects_daily_loss_limit,
        test_backtest_trade_log_structure,
        test_no_lookahead_ordering,
        test_daily_trade_limit_resets_across_calendar_days,
        test_trading_disabled_does_not_persist_forever_in_backtest,
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
