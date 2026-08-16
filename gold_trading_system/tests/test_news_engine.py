import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta, timezone
from news_engine.news_provider import (
    NewsItem, SourceType, NewsImpactClassifier, NewsRiskState, MockNewsProvider
)


def make_item(text, source=SourceType.BLOOMBERG, minutes_ago=0, author="test"):
    return NewsItem(
        source=source, author=author, text=text,
        published_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


def test_high_impact_keyword_detected():
    classifier = NewsImpactClassifier()
    item = make_item("Trump announces new tariffs on Chinese steel imports")
    result = classifier.classify(item)
    print(f"High impact test: {result.impact_level}, keywords={result.matched_keywords}")
    assert result.impact_level == "HIGH"
    assert result.risk_state == NewsRiskState.HIGH_IMPACT
    assert "tariff" in result.matched_keywords or "tariffs" in result.matched_keywords


def test_medium_impact_keyword_detected():
    classifier = NewsImpactClassifier()
    item = make_item("US CPI inflation data due out this morning")
    result = classifier.classify(item)
    assert result.impact_level == "MEDIUM"
    assert result.risk_state == NewsRiskState.EVENT_APPROACHING


def test_irrelevant_news_normal():
    classifier = NewsImpactClassifier()
    item = make_item("Local sports team wins championship game")
    result = classifier.classify(item)
    assert result.impact_level == "LOW"
    assert result.is_gold_relevant == False
    assert result.risk_state == NewsRiskState.NORMAL


def test_truth_social_medium_keyword_elevated_to_high():
    """Trump posts get elevated scrutiny — a medium-tier keyword becomes HIGH from him."""
    classifier = NewsImpactClassifier()
    bloomberg_item = make_item("Dollar weakens against major currencies", source=SourceType.BLOOMBERG)
    truth_item = make_item("Dollar weakens against major currencies", source=SourceType.TRUTH_SOCIAL,
                             author="realDonaldTrump")

    bloomberg_result = classifier.classify(bloomberg_item)
    truth_result = classifier.classify(truth_item)

    print(f"Bloomberg: {bloomberg_result.impact_level}, Truth Social: {truth_result.impact_level}")
    assert bloomberg_result.impact_level == "MEDIUM"
    assert truth_result.impact_level == "HIGH", \
        "The SAME text from Trump's Truth Social should be elevated to HIGH impact"


def test_no_fabricated_events_only_real_text():
    """Verify the classifier only works on text actually in the NewsItem —
    no hidden logic that could 'invent' an event."""
    classifier = NewsImpactClassifier()
    item = make_item("")  # empty text — nothing was actually said
    result = classifier.classify(item)
    assert result.risk_state == NewsRiskState.NORMAL
    assert result.matched_keywords == []


def test_aggregate_risk_recent_high_impact_wins():
    classifier = NewsImpactClassifier()
    items = [
        make_item("Routine market commentary", minutes_ago=60),
        make_item("Fed chair signals emergency rate cut", minutes_ago=5),  # recent + high impact
    ]
    assessments = [classifier.classify(i) for i in items]
    state = classifier.aggregate_risk_state(assessments)
    print(f"Aggregate state (recent high impact): {state}")
    assert state == NewsRiskState.HIGH_IMPACT


def test_aggregate_risk_old_high_impact_becomes_post_event():
    """A high-impact item that happened well outside the window should be
    POST_EVENT_VOLATILITY, not still treated as an active HIGH_IMPACT block."""
    classifier = NewsImpactClassifier()
    items = [make_item("Central bank announces surprise rate hike", minutes_ago=90)]
    assessments = [classifier.classify(i) for i in items]
    state = classifier.aggregate_risk_state(assessments, post_event_window_minutes=30)
    print(f"Aggregate state (old high impact): {state}")
    assert state == NewsRiskState.POST_EVENT_VOLATILITY


def test_aggregate_risk_no_items_normal():
    classifier = NewsImpactClassifier()
    state = classifier.aggregate_risk_state([])
    assert state == NewsRiskState.NORMAL


def test_mock_provider_returns_injected_items_only():
    """Confirms the mock never fabricates data beyond what was explicitly injected."""
    provider = MockNewsProvider()
    assert provider.fetch_recent() == []

    item = make_item("Test headline")
    provider.inject(item)
    result = provider.fetch_recent()
    assert len(result) == 1
    assert result[0] is item


def test_mock_provider_respects_limit():
    provider = MockNewsProvider()
    for i in range(10):
        provider.inject(make_item(f"Headline {i}"))
    result = provider.fetch_recent(limit=3)
    assert len(result) == 3


if __name__ == "__main__":
    tests = [
        test_high_impact_keyword_detected,
        test_medium_impact_keyword_detected,
        test_irrelevant_news_normal,
        test_truth_social_medium_keyword_elevated_to_high,
        test_no_fabricated_events_only_real_text,
        test_aggregate_risk_recent_high_impact_wins,
        test_aggregate_risk_old_high_impact_becomes_post_event,
        test_aggregate_risk_no_items_normal,
        test_mock_provider_returns_injected_items_only,
        test_mock_provider_respects_limit,
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
