import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from alerts.alert_engine import (
    AlertEngine, AlertType, AlertSeverity, MockAlertProvider
)


def test_basic_dispatch_to_single_provider():
    provider = MockAlertProvider("telegram")
    engine = AlertEngine(providers=[provider])
    results = engine.fire(AlertType.TRADE_ENTERED, "Entered GOLDM long at 63000")

    assert len(results) == 1
    assert results[0].success == True
    assert len(provider.sent) == 1
    assert provider.sent[0].message == "Entered GOLDM long at 63000"


def test_dispatch_to_multiple_providers():
    telegram = MockAlertProvider("telegram")
    email = MockAlertProvider("email")
    engine = AlertEngine(providers=[telegram, email])
    engine.fire(AlertType.SL_HIT, "Stop loss hit on GOLDM")

    assert len(telegram.sent) == 1
    assert len(email.sent) == 1


def test_default_severity_assigned_correctly():
    provider = MockAlertProvider()
    engine = AlertEngine(providers=[provider])
    engine.fire(AlertType.DAILY_RISK_LIMIT_HIT, "Daily loss limit reached")

    assert provider.sent[0].severity == AlertSeverity.CRITICAL, \
        "DAILY_RISK_LIMIT_HIT should default to CRITICAL severity"


def test_severity_override_respected():
    provider = MockAlertProvider()
    engine = AlertEngine(providers=[provider])
    engine.fire(AlertType.TRADE_ENTERED, "test", severity_override=AlertSeverity.CRITICAL)

    assert provider.sent[0].severity == AlertSeverity.CRITICAL


def test_below_min_severity_not_dispatched():
    provider = MockAlertProvider()
    engine = AlertEngine(providers=[provider], min_severity=AlertSeverity.WARNING)
    results = engine.fire(AlertType.TRADE_ENTERED, "info-level event")  # defaults to INFO

    assert results == []
    assert len(provider.sent) == 0, "INFO-level alert should be filtered out by WARNING min_severity"


def test_at_or_above_min_severity_dispatched():
    provider = MockAlertProvider()
    engine = AlertEngine(providers=[provider], min_severity=AlertSeverity.WARNING)
    engine.fire(AlertType.SL_HIT, "warning-level event")  # defaults to WARNING

    assert len(provider.sent) == 1


def test_failing_provider_does_not_crash_engine():
    """THE critical safety property: one provider failing must not break the whole dispatch."""
    good_provider = MockAlertProvider("good")
    bad_provider = MockAlertProvider("bad", always_fail=True)
    engine = AlertEngine(providers=[bad_provider, good_provider])

    results = engine.fire(AlertType.TRADE_ENTERED, "test message")
    print(f"Results: {results}")

    assert len(results) == 2
    assert results[0].success == False
    assert results[1].success == True
    assert len(good_provider.sent) == 1, "Good provider should still receive the alert"


def test_raising_provider_does_not_propagate():
    """A provider that raises an exception must be caught, not crash the caller."""
    class ExplodingProvider:
        @property
        def name(self):
            return "exploding"
        def send(self, alert):
            raise ConnectionError("network down")

    engine = AlertEngine(providers=[ExplodingProvider()])
    results = engine.fire(AlertType.BROKER_DISCONNECT, "test")  # must not raise

    assert len(results) == 1
    assert results[0].success == False
    assert "ConnectionError" in results[0].error


def test_failed_deliveries_tracked():
    bad_provider = MockAlertProvider("bad", always_fail=True)
    engine = AlertEngine(providers=[bad_provider])
    engine.fire(AlertType.SL_HIT, "test 1")
    engine.fire(AlertType.TRADE_ENTERED, "test 2")

    failures = engine.failed_deliveries()
    assert len(failures) == 2


def test_context_data_attached():
    provider = MockAlertProvider()
    engine = AlertEngine(providers=[provider])
    engine.fire(AlertType.TARGET_HIT, "Target 1 hit", context={"symbol": "GOLDM", "price": 63200})

    assert provider.sent[0].context["symbol"] == "GOLDM"
    assert provider.sent[0].context["price"] == 63200


if __name__ == "__main__":
    tests = [
        test_basic_dispatch_to_single_provider,
        test_dispatch_to_multiple_providers,
        test_default_severity_assigned_correctly,
        test_severity_override_respected,
        test_below_min_severity_not_dispatched,
        test_at_or_above_min_severity_dispatched,
        test_failing_provider_does_not_crash_engine,
        test_raising_provider_does_not_propagate,
        test_failed_deliveries_tracked,
        test_context_data_attached,
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
