"""
News & Event Risk Engine (Sprint 1 §17), extended to cover Bloomberg
headlines and Truth Social posts alongside the economic calendar.

HARD RULE (never violated): the classifier only acts on TEXT ACTUALLY
RETRIEVED from a real source. It never lets an LLM invent, infer, or
guess that an event occurred — every NewsItem here must trace back to
a real fetched headline/post, with its source and timestamp attached.

Same abstraction pattern as the broker layer: build the interface + a
mock provider now, wire in paid real adapters (Truth Social scraper API,
Bloomberg-aggregated RSS) later without touching any downstream code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone


class NewsRiskState(str, Enum):
    NORMAL = "NORMAL"
    EVENT_APPROACHING = "EVENT_APPROACHING"
    HIGH_IMPACT = "HIGH_IMPACT"
    POST_EVENT_VOLATILITY = "POST_EVENT_VOLATILITY"


class SourceType(str, Enum):
    BLOOMBERG = "BLOOMBERG"
    TRUTH_SOCIAL = "TRUTH_SOCIAL"
    ECONOMIC_CALENDAR = "ECONOMIC_CALENDAR"


@dataclass
class NewsItem:
    source: SourceType
    author: str            # e.g. "Bloomberg Markets", "realDonaldTrump"
    text: str               # the actual retrieved headline/post text — never fabricated
    published_at: datetime
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    url: str = ""


@dataclass
class ImpactAssessment:
    item: NewsItem
    is_gold_relevant: bool
    impact_level: str        # LOW / MEDIUM / HIGH
    matched_keywords: list
    risk_state: NewsRiskState


class NewsSourceProvider(ABC):
    """Every news source (Bloomberg aggregator, Truth Social scraper, mock)
    implements this. Downstream code only ever depends on this interface."""

    @abstractmethod
    def fetch_recent(self, limit: int = 20) -> list[NewsItem]: ...


class MockNewsProvider(NewsSourceProvider):
    """For testing/dev — returns whatever items are injected. Real adapters
    (paid Truth Social scraper API, Bloomberg-via-aggregator RSS) implement
    the same interface and are wired in when API credentials are available."""

    def __init__(self):
        self._items: list[NewsItem] = []

    def inject(self, item: NewsItem):
        self._items.append(item)

    def fetch_recent(self, limit: int = 20) -> list[NewsItem]:
        return self._items[-limit:]


# Keyword sets are configuration, not hardcoded truth — same principle as
# confluence weights. Start narrow and expand based on what actually moves
# gold in practice; false negatives are safer than false positives here
# since missing a real event just means normal risk continues, whereas a
# false HIGH_IMPACT unnecessarily blocks valid trades.
HIGH_IMPACT_KEYWORDS = {
    "tariff", "tariffs", "sanctions", "war", "invasion", "nuclear",
    "rate cut", "rate hike", "fed chair", "fomc", "recession",
    "gold reserves", "central bank buying", "de-dollarization",
    "trade deal", "trade war", "emergency", "crisis",
}
MEDIUM_IMPACT_KEYWORDS = {
    "inflation", "cpi", "jobs report", "nonfarm payroll", "gdp",
    "interest rate", "treasury yield", "dollar", "china", "geopolitical",
    "opec", "oil prices", "economic data",
}


class NewsImpactClassifier:
    def __init__(self, high_impact_keywords: set = None, medium_impact_keywords: set = None):
        self.high_keywords = high_impact_keywords or HIGH_IMPACT_KEYWORDS
        self.medium_keywords = medium_impact_keywords or MEDIUM_IMPACT_KEYWORDS

    def classify(self, item: NewsItem) -> ImpactAssessment:
        text_lower = item.text.lower()
        matched_high = [kw for kw in self.high_keywords if kw in text_lower]
        matched_medium = [kw for kw in self.medium_keywords if kw in text_lower]

        if matched_high:
            impact_level = "HIGH"
            risk_state = NewsRiskState.HIGH_IMPACT
            matched = matched_high
        elif matched_medium:
            impact_level = "MEDIUM"
            risk_state = NewsRiskState.EVENT_APPROACHING
            matched = matched_medium
        else:
            impact_level = "LOW"
            risk_state = NewsRiskState.NORMAL
            matched = []

        # Trump/Truth Social posts get elevated scrutiny even on medium-tier
        # keywords, since market reaction speed to his posts has historically
        # been faster and larger than equivalent Bloomberg headlines
        if item.source == SourceType.TRUTH_SOCIAL and impact_level == "MEDIUM":
            impact_level = "HIGH"
            risk_state = NewsRiskState.HIGH_IMPACT

        return ImpactAssessment(
            item=item, is_gold_relevant=bool(matched),
            impact_level=impact_level, matched_keywords=matched,
            risk_state=risk_state,
        )

    def aggregate_risk_state(self, assessments: list[ImpactAssessment],
                               post_event_window_minutes: int = 30) -> NewsRiskState:
        """Combines multiple recent items into a single current risk state
        for the risk engine to act on. HIGH_IMPACT from ANY recent item wins."""
        if not assessments:
            return NewsRiskState.NORMAL

        now = datetime.now(timezone.utc)
        for a in assessments:
            if a.risk_state == NewsRiskState.HIGH_IMPACT:
                minutes_since = (now - a.item.published_at).total_seconds() / 60
                if minutes_since <= post_event_window_minutes:
                    return NewsRiskState.HIGH_IMPACT
                else:
                    return NewsRiskState.POST_EVENT_VOLATILITY

        if any(a.risk_state == NewsRiskState.EVENT_APPROACHING for a in assessments):
            return NewsRiskState.EVENT_APPROACHING

        return NewsRiskState.NORMAL
