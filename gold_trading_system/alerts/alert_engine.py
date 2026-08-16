"""
Alerts Engine (Sprint 1 §34). Same abstraction pattern as broker_adapters
and news_engine: an AlertProvider interface + mock now, real adapters
(Telegram, Email, Web push) wired in later without touching this module.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone


class AlertType(str, Enum):
    TRADE_SETUP_FOUND = "TRADE_SETUP_FOUND"
    TRADE_ENTERED = "TRADE_ENTERED"
    TARGET_HIT = "TARGET_HIT"
    SL_HIT = "SL_HIT"
    TRAILING_ACTIVATED = "TRAILING_ACTIVATED"
    PARTIAL_PROFIT_BOOKED = "PARTIAL_PROFIT_BOOKED"
    TRADE_CLOSED = "TRADE_CLOSED"
    DAILY_RISK_LIMIT_HIT = "DAILY_RISK_LIMIT_HIT"
    DATA_FEED_FAILURE = "DATA_FEED_FAILURE"
    BROKER_DISCONNECT = "BROKER_DISCONNECT"
    HIGH_IMPACT_EVENT = "HIGH_IMPACT_EVENT"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Which severity each alert type carries by default — config, not hardcoded
# logic buried in the dispatch path; callers can override per-alert if needed.
DEFAULT_SEVERITY = {
    AlertType.TRADE_SETUP_FOUND: AlertSeverity.INFO,
    AlertType.TRADE_ENTERED: AlertSeverity.INFO,
    AlertType.TARGET_HIT: AlertSeverity.INFO,
    AlertType.SL_HIT: AlertSeverity.WARNING,
    AlertType.TRAILING_ACTIVATED: AlertSeverity.INFO,
    AlertType.PARTIAL_PROFIT_BOOKED: AlertSeverity.INFO,
    AlertType.TRADE_CLOSED: AlertSeverity.INFO,
    AlertType.DAILY_RISK_LIMIT_HIT: AlertSeverity.CRITICAL,
    AlertType.DATA_FEED_FAILURE: AlertSeverity.CRITICAL,
    AlertType.BROKER_DISCONNECT: AlertSeverity.CRITICAL,
    AlertType.HIGH_IMPACT_EVENT: AlertSeverity.WARNING,
}


@dataclass
class Alert:
    alert_type: AlertType
    message: str
    severity: AlertSeverity
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict = field(default_factory=dict)   # e.g. {"symbol": "GOLDM", "price": 63000}


@dataclass
class DispatchResult:
    provider_name: str
    success: bool
    error: str = ""


class AlertProvider(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> DispatchResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class MockAlertProvider(AlertProvider):
    """Records every alert sent — for testing/dev. Real adapters (Telegram
    bot API, SMTP email, web push) implement the same interface."""

    def __init__(self, name: str = "mock", always_fail: bool = False):
        self._name = name
        self.sent: list[Alert] = []
        self.always_fail = always_fail

    @property
    def name(self) -> str:
        return self._name

    def send(self, alert: Alert) -> DispatchResult:
        if self.always_fail:
            return DispatchResult(provider_name=self._name, success=False,
                                    error="Simulated provider failure")
        self.sent.append(alert)
        return DispatchResult(provider_name=self._name, success=True)


class AlertEngine:
    """
    Dispatches alerts to all configured providers. A provider failure is
    logged but NEVER blocks trading logic — alerting is observability,
    not a trading decision path. One provider going down must not cascade
    into a trading halt.
    """

    def __init__(self, providers: list[AlertProvider] = None,
                 min_severity: AlertSeverity = AlertSeverity.INFO):
        self.providers = providers or []
        self.min_severity = min_severity
        self.dispatch_log: list[dict] = []

    def _severity_rank(self, s: AlertSeverity) -> int:
        return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}[s.value]

    def fire(self, alert_type: AlertType, message: str, context: dict = None,
             severity_override: AlertSeverity = None) -> list[DispatchResult]:
        severity = severity_override or DEFAULT_SEVERITY.get(alert_type, AlertSeverity.INFO)
        alert = Alert(alert_type=alert_type, message=message, severity=severity,
                      context=context or {})

        if self._severity_rank(severity) < self._severity_rank(self.min_severity):
            return []  # below the configured noise threshold, don't dispatch

        results = []
        for provider in self.providers:
            try:
                result = provider.send(alert)
            except Exception as e:
                # a raising provider must not propagate into trading logic
                result = DispatchResult(provider_name=provider.name, success=False,
                                          error=f"Provider raised: {type(e).__name__}: {e}")
            results.append(result)

        self.dispatch_log.append({
            "alert": alert, "results": results,
        })
        return results

    def failed_deliveries(self) -> list[dict]:
        return [entry for entry in self.dispatch_log
                if any(not r.success for r in entry["results"])]
