"""
Strategy Regime Analytics (Sprint 1 §31).
"A strategy that performs only in one regime must not be automatically
used in all regimes." Segments trade_log entries (from backtest_runner,
each tagged with its entry_regime) into independent per-regime metrics,
so a strong aggregate number can't hide a strategy that only works in
one market condition.
"""
from dataclasses import dataclass

from backtesting.metrics import compute_metrics, BacktestMetrics


@dataclass
class RegimeBreakdown:
    per_regime_metrics: dict   # regime name -> BacktestMetrics
    best_regime: str | None
    worst_regime: str | None
    regime_consistency_warning: str | None   # set if performance is concentrated in one regime


def analyze_by_regime(trade_log: list[dict], min_trades_for_confidence: int = 10) -> RegimeBreakdown:
    """
    trade_log: list of dicts as produced by backtest_runner, each with
    'r_multiple' and 'entry_regime' keys. Trades with a missing/None regime
    tag are grouped under 'UNKNOWN' rather than silently dropped.
    """
    buckets: dict[str, list[float]] = {}
    for t in trade_log:
        regime = t.get("entry_regime") or "UNKNOWN"
        buckets.setdefault(regime, []).append(t["r_multiple"])

    per_regime: dict[str, BacktestMetrics] = {
        regime: compute_metrics(r_list) for regime, r_list in buckets.items()
    }

    # only compare regimes with enough trades to be meaningful
    confident_regimes = {
        r: m for r, m in per_regime.items() if m.total_trades >= min_trades_for_confidence
    }

    best_regime = None
    worst_regime = None
    if confident_regimes:
        best_regime = max(confident_regimes, key=lambda r: confident_regimes[r].expectancy_r)
        worst_regime = min(confident_regimes, key=lambda r: confident_regimes[r].expectancy_r)

    warning = None
    if len(confident_regimes) >= 2:
        total_trades = sum(m.total_trades for m in confident_regimes.values())
        best_share = confident_regimes[best_regime].total_trades / total_trades
        # flag if one regime dominates trade count AND the OTHER regimes
        # are net negative — i.e. the aggregate positive number is being
        # carried entirely by one regime, masking failure elsewhere
        other_regimes_negative = all(
            m.expectancy_r <= 0 for r, m in confident_regimes.items() if r != best_regime
        )
        if other_regimes_negative and len(confident_regimes) >= 2:
            warning = (
                f"Positive performance is concentrated in '{best_regime}' "
                f"(expectancy {confident_regimes[best_regime].expectancy_r:.3f}R). "
                f"Every other regime with enough trades to judge shows non-positive "
                f"expectancy — this strategy may only actually work in one market "
                f"condition, not universally as the aggregate backtest number implies."
            )

    return RegimeBreakdown(
        per_regime_metrics=per_regime,
        best_regime=best_regime,
        worst_regime=worst_regime,
        regime_consistency_warning=warning,
    )
