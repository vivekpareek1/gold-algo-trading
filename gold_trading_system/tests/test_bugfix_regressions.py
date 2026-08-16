"""
Regression tests for the six bugs found in the full-codebase review.
Each test FAILS against the pre-fix code and passes after the fix, so these
bugs cannot silently return.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from trade_manager.trade_manager import TradeManager, TradeManagerState, TradeState, ExitReason
from execution.broker_adapters.paper_provider import PaperBrokerProvider
from execution.broker_adapters.base import OrderRequest, OrderSide
from risk_engine.risk_engine import RiskEngine, DailyRiskState, VetoReason


def make_long():
    return TradeManagerState(
        direction="LONG", entry_price=63000, original_stop=62900, current_stop=62900,
        original_risk_points=100, target_1=63100, target_2=63200, target_3=63300,
    )


# ---------- BUG 1: trailing stop placed beyond current price ----------

def test_bug1_trailing_stop_never_above_price_long():
    tm = TradeManager(Settings(), make_long())
    # EMA9 sits ABOVE current price (normal during a pullback)
    tm.update_trailing_stop(current_price=63050, ema9=63080, ema21=63000, ema50=62950,
                              atr=20, momentum_health="STRONG", structure_broke_against=False)
    print(f"stop={tm.state.current_stop}, price=63050")
    assert tm.state.current_stop < 63050, \
        f"LONG trailing stop must stay BELOW current price, got {tm.state.current_stop}"
    assert not tm.check_stop_hit(63050), "Stop must not self-trigger at the same price it was set"


def test_bug1_trailing_stop_never_below_price_short():
    st = TradeManagerState(direction="SHORT", entry_price=63000, original_stop=63100,
                            current_stop=63100, original_risk_points=100,
                            target_1=62900, target_2=62800, target_3=62700)
    tm = TradeManager(Settings(), st)
    # EMA9 sits BELOW current price for a short — would place stop through market
    tm.update_trailing_stop(current_price=62950, ema9=62920, ema21=63000, ema50=63050,
                              atr=20, momentum_health="STRONG", structure_broke_against=False)
    assert tm.state.current_stop > 62950, \
        f"SHORT trailing stop must stay ABOVE current price, got {tm.state.current_stop}"


# ---------- BUG 2: intrabar stop detection ----------

def test_bug2_intrabar_stop_detected_when_close_recovers():
    tm = TradeManager(Settings(), make_long())
    # candle pierced the 62900 stop (low 62850) but closed back above it
    hit = tm.check_stop_hit_intrabar(high=63000, low=62850, close=62950)
    print(f"intrabar hit={hit}, state={tm.state.trade_state}")
    assert hit is True, "A candle whose LOW pierced the stop must register as a stop-out"
    assert tm.state.trade_state == TradeState.EXITED
    assert tm.state.exit_reason == ExitReason.STOP_LOSS_HIT


def test_bug2_intrabar_no_false_positive():
    tm = TradeManager(Settings(), make_long())
    hit = tm.check_stop_hit_intrabar(high=63200, low=62950, close=63100)  # never touched 62900
    assert hit is False


def test_bug2_gap_through_stop_fills_worse_than_stop_price():
    """If the whole candle gapped below the stop, the fill must not optimistically
    assume a clean fill at the stop price."""
    tm = TradeManager(Settings(), make_long())
    tm.check_stop_hit_intrabar(high=62800, low=62700, close=62750)  # entire candle below stop
    exit_price = tm.state.state_history[-1].price_at_event
    print(f"gap fill price={exit_price} (stop was 62900)")
    assert exit_price <= 62800, \
        f"Gap-through fill must be at/below the candle high (62800), not the stop price, got {exit_price}"


# ---------- BUG 3: blended R accounting across partials ----------

def test_bug3_blended_r_includes_booked_partials():
    tm = TradeManager(Settings(), make_long())
    tm.check_partial_booking(63100)   # +1R, books 25%
    tm.check_partial_booking(63200)   # T1 (2R), books another 25%
    remaining = tm.state.quantity_remaining_pct
    blended = tm.blended_r_multiple(63000)   # runner round-trips to breakeven
    naive = tm._r_multiple(63000)
    print(f"remaining={remaining}%, naive_r={naive}, blended_r={blended}")
    assert naive == 0.0
    assert blended > 0.0, \
        "Blended R must reflect profit banked at +1R and +2R even if the runner exits at breakeven"


def test_bug3_blended_r_matches_full_runner_when_no_partials():
    tm = TradeManager(Settings(), make_long())
    blended = tm.blended_r_multiple(63200)   # no partials booked
    assert abs(blended - tm._r_multiple(63200)) < 1e-9, \
        "With no partials booked, blended R must equal the plain R multiple"


# ---------- BUG 4: paper broker equity ignores realized P&L ----------

def test_bug4_realized_loss_reduces_equity():
    b = PaperBrokerProvider(starting_equity_inr=500_000.0)
    b.connect()
    b.set_quote("GOLDM", ltp=63000)
    b.place_order(OrderRequest(client_order_id="o1", symbol="GOLDM",
                                 side=OrderSide.BUY, quantity=1))
    b.set_quote("GOLDM", ltp=60000)   # 3000-point adverse move
    b.place_order(OrderRequest(client_order_id="o2", symbol="GOLDM",
                                 side=OrderSide.SELL, quantity=1))
    equity = b.get_balance().equity_inr
    print(f"equity after 3000pt loss: {equity}")
    assert equity < 480_000, \
        f"A 3000-point loss on GOLDM (₹10/pt) should cost ~₹30,000, equity was {equity}"


def test_bug4_realized_profit_increases_equity():
    b = PaperBrokerProvider(starting_equity_inr=500_000.0)
    b.connect()
    b.set_quote("GOLDM", ltp=60000)
    b.place_order(OrderRequest(client_order_id="p1", symbol="GOLDM",
                                 side=OrderSide.BUY, quantity=1))
    b.set_quote("GOLDM", ltp=63000)   # 3000-point favourable move
    b.place_order(OrderRequest(client_order_id="p2", symbol="GOLDM",
                                 side=OrderSide.SELL, quantity=1))
    equity = b.get_balance().equity_inr
    print(f"equity after 3000pt gain: {equity}")
    assert equity > 520_000, f"A 3000-point gain should add ~₹30,000, equity was {equity}"


# ---------- BUG 5: session_quality_ok was read from the wrong dataclass ----------

def test_bug5_session_quality_penalty_actually_applies():
    from market_structure.structure_engine import StructureState, TrendState, StructureEvent
    from situation_analysis.situation_analyzer import SituationSnapshot, MarketRegime
    from gold_intelligence.fair_value import FairValueResult, MacroBiasResult
    from signal_engine.signal_engine import SignalEngine, ConfluenceInputs

    def build(session_ok):
        s = StructureState(); s.trend = TrendState.TRENDING_UP
        sit = SituationSnapshot(
            regime=MarketRegime.TRENDING_UP, htf_trend=TrendState.TRENDING_UP,
            ltf_trend=TrendState.TRENDING_UP, trend_alignment_score=90,
            momentum_health="STRONG", last_structure_event=StructureEvent.NONE,
            is_pullback_opportunity=False, macro_alignment="SUPPORTIVE",
            ready_for_ai_review=True, explanation="t", warnings=[],
        )
        return ConfluenceInputs(
            ltf_structure=s, situation=sit,
            fair_value=FairValueResult(mcx_price=63000, theoretical_price=63000, deviation=0,
                                         deviation_pct=0, deviation_zscore=0.0, is_reliable=False),
            macro=MacroBiasResult(macro_bias=0.0, components={}),
            ema_aligned_bullish=True, ema_aligned_bearish=False,
            macd_bullish=True, macd_bearish=False, rsi=60,
            price_above_vwap=True, volume_supportive=True, bb_squeeze=False,
            session_quality_ok=session_ok,
        )

    engine = SignalEngine(Settings())
    good = engine.evaluate(build(True))
    bad = engine.evaluate(build(False))
    print(f"session ok long_score={good.long_score}, session bad long_score={bad.long_score}")
    assert bad.long_score < good.long_score, \
        "A failing session-quality flag must actually reduce the confluence score"


# ---------- BUG 6: POSITION_ALREADY_OPEN veto was unreachable ----------

def test_bug6_position_already_open_veto_fires():
    engine = RiskEngine(Settings(), DailyRiskState())
    veto_before = engine.check_hard_limits(live_equity_inr=500_000, data_is_stale=False,
                                             position_already_open=False)
    assert veto_before == VetoReason.NONE

    engine.register_position_opened()
    veto_after = engine.check_hard_limits(live_equity_inr=500_000, data_is_stale=False,
                                            position_already_open=True)
    print(f"veto with 1 open position: {veto_after}")
    assert veto_after == VetoReason.POSITION_ALREADY_OPEN, \
        "With max_simultaneous_positions=1 and one open, the veto must fire"


def test_bug6_position_closed_releases_veto():
    engine = RiskEngine(Settings(), DailyRiskState())
    engine.register_position_opened()
    engine.register_position_closed()
    veto = engine.check_hard_limits(live_equity_inr=500_000, data_is_stale=False,
                                      position_already_open=False)
    assert veto == VetoReason.NONE, "Closing the position must release the veto"


if __name__ == "__main__":
    tests = [
        test_bug1_trailing_stop_never_above_price_long,
        test_bug1_trailing_stop_never_below_price_short,
        test_bug2_intrabar_stop_detected_when_close_recovers,
        test_bug2_intrabar_no_false_positive,
        test_bug2_gap_through_stop_fills_worse_than_stop_price,
        test_bug3_blended_r_includes_booked_partials,
        test_bug3_blended_r_matches_full_runner_when_no_partials,
        test_bug4_realized_loss_reduces_equity,
        test_bug4_realized_profit_increases_equity,
        test_bug5_session_quality_penalty_actually_applies,
        test_bug6_position_already_open_veto_fires,
        test_bug6_position_closed_releases_veto,
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
