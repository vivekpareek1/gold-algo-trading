"""
Regression test for the final review's structural finding: momentum
classification logic used to be independently copy-pasted into
backtest_runner.py and live_trading_engine.py. Identical at the time, but
two copies of the same decision is exactly the kind of thing that silently
diverges after a fix lands in one place and not the other.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from indicators.incremental import momentum_health_from_indicator_result


def _ind(macd_hist, macd_hist_prev, rel_volume):
    return {"macd_hist": macd_hist, "macd_hist_prev": macd_hist_prev, "rel_volume": rel_volume}


def test_both_modules_use_the_same_shared_function_object():
    """The real regression guard: both call sites must resolve to the exact
    same function, not two coincidentally-identical copies."""
    import backtesting.backtest_runner as bt
    import execution.live_trading_engine as live

    assert bt.momentum_health_from_indicator_result is live.momentum_health_from_indicator_result, \
        "backtest_runner and live_trading_engine must import the SAME function object, " \
        "not independent copies that could silently diverge"


def test_strong_momentum():
    result = momentum_health_from_indicator_result(_ind(2.0, 1.0, 1.5))
    assert result == "STRONG"


def test_dead_momentum():
    result = momentum_health_from_indicator_result(_ind(0.5, 1.0, 0.5))
    assert result == "DEAD"


def test_weakening_momentum():
    result = momentum_health_from_indicator_result(_ind(2.0, 1.0, 0.5))
    assert result == "WEAKENING"


def test_no_backtest_runner_local_duplicate_remains():
    """Guards against the OLD duplicate function quietly being reintroduced."""
    import backtesting.backtest_runner as bt
    assert not hasattr(bt, "_momentum_from_indicators"), \
        "The old local duplicate must be fully removed, not left dead alongside the shared one"


def test_no_live_engine_local_duplicate_remains():
    from execution.live_trading_engine import LiveTradingEngine
    assert not hasattr(LiveTradingEngine, "_momentum_from_indicators"), \
        "The old local duplicate method must be fully removed"


if __name__ == "__main__":
    tests = [
        test_both_modules_use_the_same_shared_function_object,
        test_strong_momentum,
        test_dead_momentum,
        test_weakening_momentum,
        test_no_backtest_runner_local_duplicate_remains,
        test_no_live_engine_local_duplicate_remains,
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
