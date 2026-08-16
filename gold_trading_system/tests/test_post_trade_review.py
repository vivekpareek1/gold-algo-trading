import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from journal.post_trade_review import (
    PostTradeReviewEngine, ProcessChecklist, TradeGrade, summarize_reviews
)


def clean_checklist(score=80, min_required=70):
    return ProcessChecklist(
        entry_confluence_score=score, min_required_score=min_required,
        position_size_within_risk_limit=True, stop_loss_within_max_distance=True,
        stop_was_ever_widened=False, exited_per_rules=True,
        risk_reward_met_minimum=True, traded_during_blocked_event=False,
    )


def test_winning_clean_trade_is_good_win():
    engine = PostTradeReviewEngine()
    result = engine.review(pnl_r=2.0, checklist=clean_checklist())
    print(f"Clean win: {result.grade}, violations={result.process_violations}")
    assert result.grade == TradeGrade.GOOD_WIN
    assert result.process_violations == []
    assert result.process_score == 100


def test_losing_clean_trade_is_good_loss():
    """THE core principle of this module: a loss with a clean process is GOOD, not bad."""
    engine = PostTradeReviewEngine()
    result = engine.review(pnl_r=-1.0, checklist=clean_checklist())
    print(f"Clean loss: {result.grade}, explanation={result.explanation}")
    assert result.grade == TradeGrade.GOOD_LOSS, \
        "A loss that followed all process rules must be graded GOOD_LOSS, not treated as a mistake"


def test_winning_dirty_trade_is_bad_win():
    """THE other core principle: a profitable trade that broke rules is BAD, not good."""
    engine = PostTradeReviewEngine()
    dirty = clean_checklist()
    dirty.stop_was_ever_widened = True  # broke the never-widen rule, but got lucky and won
    result = engine.review(pnl_r=3.0, checklist=dirty)
    print(f"Dirty win: {result.grade}, violations={result.process_violations}")
    assert result.grade == TradeGrade.BAD_WIN, \
        "A profitable trade that violated process (e.g. widened stop) must be BAD_WIN, " \
        "not treated as a success to repeat"
    assert "widened" in result.process_violations[0].lower()


def test_losing_dirty_trade_is_bad_loss():
    engine = PostTradeReviewEngine()
    dirty = clean_checklist()
    dirty.position_size_within_risk_limit = False
    dirty.exited_per_rules = False
    result = engine.review(pnl_r=-2.0, checklist=dirty)
    assert result.grade == TradeGrade.BAD_LOSS
    assert len(result.process_violations) == 2


def test_below_threshold_entry_flagged_as_violation():
    engine = PostTradeReviewEngine()
    checklist = clean_checklist(score=55, min_required=70)  # entered below the required bar
    result = engine.review(pnl_r=1.5, checklist=checklist)
    assert result.grade == TradeGrade.BAD_WIN, \
        "Winning despite entering below the confluence threshold is still a process violation"


def test_breakeven_trade_ungraded():
    engine = PostTradeReviewEngine()
    result = engine.review(pnl_r=0.0, checklist=clean_checklist())
    assert result.grade == TradeGrade.UNGRADED


def test_blocked_event_trade_flagged():
    engine = PostTradeReviewEngine()
    checklist = clean_checklist()
    checklist.traded_during_blocked_event = True
    result = engine.review(pnl_r=1.0, checklist=checklist)
    assert result.grade == TradeGrade.BAD_WIN
    assert any("news/event" in v for v in result.process_violations)


def test_process_score_partial_credit():
    """5 of 7 checks pass -> score should be ~71%, not 0 or 100."""
    engine = PostTradeReviewEngine()
    checklist = clean_checklist()
    checklist.stop_was_ever_widened = True     # fail 1
    checklist.risk_reward_met_minimum = False  # fail 2
    result = engine.review(pnl_r=1.0, checklist=checklist)
    print(f"Partial credit score: {result.process_score}")
    assert 60 <= result.process_score <= 80, f"Expected ~71% (5/7), got {result.process_score}"


def test_journal_summary_aggregates_correctly():
    engine = PostTradeReviewEngine()
    results = [
        engine.review(2.0, clean_checklist()),    # GOOD_WIN
        engine.review(-1.0, clean_checklist()),   # GOOD_LOSS
        engine.review(0.0, clean_checklist()),    # UNGRADED
    ]
    dirty = clean_checklist()
    dirty.stop_was_ever_widened = True
    results.append(engine.review(3.0, dirty))     # BAD_WIN

    summary = summarize_reviews(results)
    print(f"Summary: {summary}, discipline%={summary.process_discipline_pct}")

    assert summary.total_reviewed == 4
    assert summary.good_win_count == 1
    assert summary.good_loss_count == 1
    assert summary.bad_win_count == 1
    assert summary.ungraded_count == 1
    # discipline % = disciplined(2) / graded(3, excluding ungraded) = 66.7%
    assert abs(summary.process_discipline_pct - 66.7) < 0.5


def test_journal_summary_empty_no_crash():
    summary = summarize_reviews([])
    assert summary.total_reviewed == 0
    assert summary.process_discipline_pct == 0.0


if __name__ == "__main__":
    tests = [
        test_winning_clean_trade_is_good_win,
        test_losing_clean_trade_is_good_loss,
        test_winning_dirty_trade_is_bad_win,
        test_losing_dirty_trade_is_bad_loss,
        test_below_threshold_entry_flagged_as_violation,
        test_breakeven_trade_ungraded,
        test_blocked_event_trade_flagged,
        test_process_score_partial_credit,
        test_journal_summary_aggregates_correctly,
        test_journal_summary_empty_no_crash,
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
