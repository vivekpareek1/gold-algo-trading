"""
Full Validation Pipeline — Steps 2-5 of the rollout guide, as one command.

Usage:
    python3 run_full_validation.py path/to/mcx_gold_candles.csv

Expected CSV columns: timestamp, open, high, low, close, volume
(timestamp format: "YYYY-MM-DD HH:MM:SS")

This does NOT fabricate or assume anything about the data — every number
in the report traces back to the real candles you provide. If the data
is too thin for a given check (walk-forward needs 100+ candles, Monte
Carlo wants 30+ trades for confidence), the report says so explicitly
rather than presenting a result with false precision.
"""
import sys
import json
from datetime import datetime

from config.settings import Settings
from market_structure.structure_engine import TrendState
from market_data.data_loader import DataQualityGate
from backtesting.backtest_runner import run_backtest
from backtesting.walk_forward import run_walk_forward
from backtesting.monte_carlo import run_monte_carlo
from backtesting.regime_analytics import analyze_by_regime
from situation_analysis.day_of_week_situational import DayOfWeekAnalyzer


def run_full_validation(csv_path: str, htf_timeframe: str = "1H",
                          walk_forward_windows: int = 4) -> dict:
    settings = Settings()
    report = {"generated_at": datetime.now().isoformat(), "source_file": csv_path}

    # ---- Step 2: load + validate ----
    print("=" * 70)
    print("STEP 2 — Data Quality Gate")
    print("=" * 70)
    gate = DataQualityGate(settings, expected_interval_minutes=5)
    load_result = gate.load_csv(csv_path)

    print(f"Rows seen: {load_result.total_rows_seen}")
    print(f"Accepted candles: {len(load_result.candles)}")
    print(f"Rejected: {load_result.rejected_count}")
    if load_result.issues:
        issue_types = {}
        for i in load_result.issues:
            issue_types[i.issue_type] = issue_types.get(i.issue_type, 0) + 1
        print(f"Rejection breakdown: {issue_types}")

    report["data_quality"] = {
        "total_rows": load_result.total_rows_seen,
        "accepted": len(load_result.candles),
        "rejected": load_result.rejected_count,
        "issue_breakdown": issue_types if load_result.issues else {},
    }

    candles = load_result.candles
    if len(candles) < 100:
        print(f"\n⚠ Only {len(candles)} clean candles — too few for walk-forward or a "
              f"meaningful backtest. Report will note this rather than fabricate results.")
        report["warning"] = "Insufficient clean data for full validation (need 100+ candles)."
        _print_and_return(report)
        return report

    # ---- Step 3: real backtest ----
    print("\n" + "=" * 70)
    print("STEP 3 — Backtest (real multi-timeframe)")
    print("=" * 70)
    backtest_result = run_backtest(candles, settings, base_timeframe="5M",
                                     htf_timeframe=htf_timeframe)
    m = backtest_result.metrics
    print(f"Total trades: {m.total_trades}")
    print(f"Win rate: {m.win_rate:.1f}%")
    print(f"Expectancy: {m.expectancy_r:.3f}R per trade")
    print(f"Profit factor: {m.profit_factor:.2f}")
    print(f"Max drawdown: {m.max_drawdown_r:.2f}R")
    print(f"Max consecutive losses: {m.max_consecutive_losses}")

    if m.total_trades < 20:
        print(f"\n⚠ Only {m.total_trades} trades generated — too few to trust these numbers. "
              f"Either the data window is short, or the strategy is (correctly) selective. "
              f"Do not treat this win rate/expectancy as reliable yet.")

    report["backtest"] = {
        "total_trades": m.total_trades, "win_rate": m.win_rate,
        "expectancy_r": m.expectancy_r, "profit_factor": m.profit_factor,
        "max_drawdown_r": m.max_drawdown_r, "max_consecutive_losses": m.max_consecutive_losses,
        "low_sample_warning": m.total_trades < 20,
    }

    trade_r_multiples = [t["r_multiple"] for t in backtest_result.trade_log]

    # ---- Step 4a: walk-forward ----
    print("\n" + "=" * 70)
    print("STEP 4a — Walk-Forward Validation (overfitting check)")
    print("=" * 70)
    if len(candles) >= 400:
        try:
            wf = run_walk_forward(candles, settings, num_windows=walk_forward_windows,
                                    base_timeframe="5M", htf_timeframe=htf_timeframe)
            print(f"Windows: {wf.num_windows}")
            print(f"Avg train expectancy: {wf.avg_train_expectancy:.3f}R")
            print(f"Avg test (out-of-sample) expectancy: {wf.avg_test_expectancy:.3f}R")
            print(f"Windows with positive out-of-sample expectancy: "
                  f"{wf.windows_with_positive_test_expectancy}/{wf.num_windows} "
                  f"({wf.windows_with_positive_test_expectancy_pct:.0f}%)")
            print(f"Decay: {wf.train_to_test_decay_pct:.0f}%")
            print(f"Verdict: {wf.consistency_flag}")
            report["walk_forward"] = {
                "avg_train_expectancy": wf.avg_train_expectancy,
                "avg_test_expectancy": wf.avg_test_expectancy,
                "decay_pct": wf.train_to_test_decay_pct,
                "windows_with_positive_test_expectancy": wf.windows_with_positive_test_expectancy,
                "windows_with_positive_test_expectancy_pct": wf.windows_with_positive_test_expectancy_pct,
                "is_likely_overfit": wf.is_likely_overfit,
                "verdict": wf.consistency_flag,
            }
        except ValueError as e:
            print(f"⚠ Skipped: {e}")
            report["walk_forward"] = {"skipped_reason": str(e)}
    else:
        msg = f"Skipped — need 400+ candles for meaningful walk-forward windows, have {len(candles)}."
        print(f"⚠ {msg}")
        report["walk_forward"] = {"skipped_reason": msg}

    # ---- Step 4b: Monte Carlo ----
    print("\n" + "=" * 70)
    print("STEP 4b — Monte Carlo (risk of ruin, drawdown distribution)")
    print("=" * 70)
    if trade_r_multiples:
        mc = run_monte_carlo(trade_r_multiples, num_simulations=2000, seed=42)
        print(f"Based on {mc.starting_r_multiples_count} real trades "
              f"({'LOW CONFIDENCE — treat as directional only' if mc.is_low_confidence else 'adequate sample'})")
        print(f"Drawdown — median: {mc.drawdown_p50:.1f}R, 90th pct: {mc.drawdown_p90:.1f}R, "
              f"worst case seen: {mc.max_drawdown_worst_case:.1f}R")
        print(f"Risk of ruin: {mc.risk_of_ruin_pct:.1f}%")
        print(f"Median simulated final equity: {mc.final_equity_r_p50:.1f}R")
        report["monte_carlo"] = {
            "based_on_trades": mc.starting_r_multiples_count,
            "is_low_confidence": mc.is_low_confidence,
            "drawdown_p50_r": mc.drawdown_p50, "drawdown_p90_r": mc.drawdown_p90,
            "risk_of_ruin_pct": mc.risk_of_ruin_pct,
            "final_equity_p50_r": mc.final_equity_r_p50,
        }
    else:
        print("⚠ Skipped — no trades were generated by the backtest to simulate.")
        report["monte_carlo"] = {"skipped_reason": "No trades generated."}

    # ---- Step 4c: regime analytics ----
    print("\n" + "=" * 70)
    print("STEP 4c — Strategy Regime Analytics")
    print("=" * 70)
    if backtest_result.trade_log:
        regime_breakdown = analyze_by_regime(backtest_result.trade_log)
        regime_report = {}
        for regime, m in regime_breakdown.per_regime_metrics.items():
            marker = " <- BEST" if regime == regime_breakdown.best_regime else (
                " <- WORST" if regime == regime_breakdown.worst_regime else "")
            print(f"{regime:20s} n={m.total_trades:4d}  win_rate={m.win_rate:5.1f}%  "
                  f"expectancy={m.expectancy_r:+.3f}R  PF={m.profit_factor:.2f}{marker}")
            regime_report[regime] = {
                "total_trades": m.total_trades, "win_rate": m.win_rate,
                "expectancy_r": m.expectancy_r, "profit_factor": m.profit_factor,
            }
        if regime_breakdown.regime_consistency_warning:
            print(f"\n⚠ {regime_breakdown.regime_consistency_warning}")
        report["regime_analytics"] = {
            "per_regime": regime_report,
            "best_regime": regime_breakdown.best_regime,
            "worst_regime": regime_breakdown.worst_regime,
            "concentration_warning": regime_breakdown.regime_consistency_warning,
        }
    else:
        print("⚠ Skipped — no trades to segment by regime.")
        report["regime_analytics"] = {"skipped_reason": "No trades generated."}

    # ---- Step 5: day-of-week (Hougaard) ----
    print("\n" + "=" * 70)
    print("STEP 5 — Day-of-Week Situational Analysis (Hougaard)")
    print("=" * 70)
    # BUGFIX: feeding raw intraday (5M) candles directly into to_daily_candles()
    # treats every intraday bar as its own "day" sample — with 5M data that
    # compresses the whole weekday distribution into whichever 1-2 calendar
    # days the candle window happens to span. Must resample to genuine daily
    # OHLC (one candle per trading day) first.
    from market_data.resampler import resample
    daily_resampled = resample(candles, base_timeframe="5M", target_timeframe="1D")
    complete_daily = [rc.ohlcv for rc in daily_resampled if rc.is_complete]

    if len(complete_daily) < 20:
        msg = (f"Only {len(complete_daily)} complete daily candles available after "
               f"resampling — too few for any weekday to reach the reliability threshold. "
               f"Provide a longer date range for meaningful day-of-week patterns.")
        print(f"⚠ {msg}")
        report["day_of_week"] = {"skipped_reason": msg}
    else:
        daily_candles = gate.to_daily_candles(complete_daily)
        dow_analyzer = DayOfWeekAnalyzer()
        stats = dow_analyzer.compute_weekday_stats(daily_candles)
        dow_report = {}
        for wd, s in stats.items():
            bias = dow_analyzer.get_bias(wd, stats)
            status = "reliable" if bias.is_reliable else "insufficient data"
            print(f"{wd.name.title():10s} n={s.sample_size:4d}  win_rate={s.win_rate_pct:5.1f}%  "
                  f"bias={bias.bias_score:+6.1f}  ({status})")
            dow_report[wd.name] = {
                "sample_size": s.sample_size, "win_rate_pct": s.win_rate_pct,
                "bias_score": bias.bias_score, "is_reliable": bias.is_reliable,
            }
        report["day_of_week"] = dow_report

    _print_and_return(report)
    return report


def _print_and_return(report: dict):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_full_validation.py path/to/candles.csv")
        sys.exit(1)
    run_full_validation(sys.argv[1])
