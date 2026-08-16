import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timezone, timedelta
from unittest.mock import patch


def _status_at(dt_ist_naive):
    """Helper: compute session status for a given IST-naive datetime by
    monkeypatching datetime.now inside the api.main module."""
    ist = timezone(timedelta(hours=5, minutes=30))
    fixed = dt_ist_naive.replace(tzinfo=ist)
    with patch("api.main.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        from api.main import _get_market_session_status
        return _get_market_session_status()


def test_weekday_during_session_is_open():
    # Monday, 10:00 AM IST
    result = _status_at(datetime(2026, 8, 17, 10, 0, 0))
    assert result["status"] == "OPEN"


def test_weekday_before_session_is_closed():
    # Monday, 6:00 AM IST — before 9 AM open
    result = _status_at(datetime(2026, 8, 17, 6, 0, 0))
    assert result["status"] == "CLOSED"


def test_weekday_after_session_is_closed():
    # Monday, 11:45 PM IST — after 11:30 PM close
    result = _status_at(datetime(2026, 8, 17, 23, 45, 0))
    assert result["status"] == "CLOSED"


def test_saturday_is_closed_even_during_session_hours():
    # Saturday, 12:00 PM IST — within normal hours, but weekend
    result = _status_at(datetime(2026, 8, 22, 12, 0, 0))
    assert result["status"] == "CLOSED"


def test_sunday_is_closed():
    result = _status_at(datetime(2026, 8, 16, 12, 0, 0))
    assert result["status"] == "CLOSED"


def test_exact_open_boundary_is_open():
    result = _status_at(datetime(2026, 8, 17, 9, 0, 0))
    assert result["status"] == "OPEN"


def test_exact_close_boundary_is_open():
    result = _status_at(datetime(2026, 8, 17, 23, 30, 0))
    assert result["status"] == "OPEN"


def test_local_time_formatted_correctly():
    result = _status_at(datetime(2026, 8, 17, 14, 5, 30))
    assert "14:05:30" in result["local_time"]
    assert "IST" in result["local_time"]


if __name__ == "__main__":
    tests = [
        test_weekday_during_session_is_open,
        test_weekday_before_session_is_closed,
        test_weekday_after_session_is_closed,
        test_saturday_is_closed_even_during_session_hours,
        test_sunday_is_closed,
        test_exact_open_boundary_is_open,
        test_exact_close_boundary_is_open,
        test_local_time_formatted_correctly,
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
