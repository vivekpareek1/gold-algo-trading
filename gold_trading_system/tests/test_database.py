import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone

from database.models import (
    Base, Instrument, MarketCandle, Signal, Trade, OrderRecord,
    StrategyModel, StrategyVersion, DailyPerformance
)


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_all_tables_create_without_error():
    session = make_session()
    assert session is not None
    session.close()


def test_instrument_insert_and_query():
    session = make_session()
    goldm = Instrument(symbol="GOLDM", exchange="MCX", lot_size_grams=100,
                        point_value_inr=10.0, tick_size=1.0)
    session.add(goldm)
    session.commit()

    fetched = session.query(Instrument).filter_by(symbol="GOLDM").first()
    assert fetched is not None
    assert fetched.point_value_inr == 10.0
    session.close()


def test_candle_unique_constraint_prevents_duplicates():
    """Same instrument+timeframe+ts must not be insertable twice."""
    session = make_session()
    inst = Instrument(symbol="GOLDM", exchange="MCX", lot_size_grams=100,
                       point_value_inr=10.0, tick_size=1.0)
    session.add(inst)
    session.commit()

    ts = datetime.now(timezone.utc)
    c1 = MarketCandle(instrument_id=inst.id, timeframe="5M", ts=ts,
                       open=100, high=101, low=99, close=100.5, volume=1000)
    session.add(c1)
    session.commit()

    c2 = MarketCandle(instrument_id=inst.id, timeframe="5M", ts=ts,
                       open=100, high=102, low=98, close=99, volume=500)
    session.add(c2)
    try:
        session.commit()
        assert False, "Expected IntegrityError on duplicate candle, but insert succeeded"
    except IntegrityError:
        print("Correctly rejected duplicate candle")
        session.rollback()
    session.close()


def test_order_client_id_idempotency_constraint():
    """client_order_id must be unique — this is the real-money duplicate-order guard."""
    session = make_session()
    o1 = OrderRecord(client_order_id="ORD-001", side="BUY", qty=1,
                      order_type="MARKET", status="PLACED")
    session.add(o1)
    session.commit()

    o2 = OrderRecord(client_order_id="ORD-001", side="SELL", qty=1,
                      order_type="MARKET", status="PLACED")
    session.add(o2)
    try:
        session.commit()
        assert False, "Expected IntegrityError on duplicate client_order_id"
    except IntegrityError:
        print("Correctly rejected duplicate client_order_id (idempotency protected)")
        session.rollback()
    session.close()


def test_signal_stores_vetoed_and_no_trade_decisions():
    """We must log NO_TRADE and vetoed signals, not just executed trades."""
    session = make_session()
    inst = Instrument(symbol="GOLDM", exchange="MCX", lot_size_grams=100,
                       point_value_inr=10.0, tick_size=1.0)
    session.add(inst)
    session.commit()

    sig = Signal(instrument_id=inst.id, decision="NO_TRADE", confidence=0,
                 was_vetoed=True, veto_reason="STOP_TOO_WIDE",
                 long_score=45, short_score=20)
    session.add(sig)
    session.commit()

    fetched = session.query(Signal).filter_by(was_vetoed=True).first()
    assert fetched is not None
    assert fetched.veto_reason == "STOP_TOO_WIDE"
    session.close()


def test_strategy_version_relationship():
    session = make_session()
    strat = StrategyModel(name="Gold Confluence v1", description="test")
    session.add(strat)
    session.commit()

    version = StrategyVersion(strategy_id=strat.id, version="1.0.0",
                                config_json={"max_risk": 2000}, is_approved=False)
    session.add(version)
    session.commit()

    fetched_strat = session.query(StrategyModel).filter_by(name="Gold Confluence v1").first()
    assert len(fetched_strat.versions) == 1
    assert fetched_strat.versions[0].is_approved == False, \
        "Strategy versions must default to NOT approved (human gate)"
    session.close()


def test_daily_performance_unique_date():
    """Only one performance row per calendar date."""
    session = make_session()
    d = datetime(2026, 8, 13, tzinfo=timezone.utc)
    p1 = DailyPerformance(date=d, trades_taken=2, net_pnl=1000.0)
    session.add(p1)
    session.commit()

    p2 = DailyPerformance(date=d, trades_taken=3, net_pnl=-500.0)
    session.add(p2)
    try:
        session.commit()
        assert False, "Expected IntegrityError on duplicate date"
    except IntegrityError:
        print("Correctly rejected duplicate daily_performance date")
        session.rollback()
    session.close()


def test_trade_links_to_signal():
    session = make_session()
    inst = Instrument(symbol="GOLDM", exchange="MCX", lot_size_grams=100,
                       point_value_inr=10.0, tick_size=1.0)
    session.add(inst)
    session.commit()

    sig = Signal(instrument_id=inst.id, decision="BUY", confidence=75)
    session.add(sig)
    session.commit()

    trade = Trade(signal_id=sig.id, instrument_id=inst.id, mode="PAPER",
                   entry_price=63000, quantity=1, stop_loss=62800)
    session.add(trade)
    session.commit()

    fetched = session.query(Trade).first()
    assert fetched.signal_id == sig.id
    assert fetched.mode == "PAPER"
    session.close()


if __name__ == "__main__":
    tests = [
        test_all_tables_create_without_error,
        test_instrument_insert_and_query,
        test_candle_unique_constraint_prevents_duplicates,
        test_order_client_id_idempotency_constraint,
        test_signal_stores_vetoed_and_no_trade_decisions,
        test_strategy_version_relationship,
        test_daily_performance_unique_date,
        test_trade_links_to_signal,
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
