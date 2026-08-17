"""
Tests for the build-version marker added after repeated confusion over
whether a deploy actually took effect — the dashboard would show stale
behavior with no visible way to confirm from a screenshot whether new
code was running. This makes it checkable at a glance, server-side
rendered so it works even if something else is broken.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.main import app, BUILD_VERSION

client = TestClient(app)


def test_build_version_is_a_non_empty_string():
    assert isinstance(BUILD_VERSION, str)
    assert len(BUILD_VERSION) > 0


def test_dashboard_html_shows_real_build_version_not_placeholder():
    html = client.get("/").text
    assert "__BUILD_VERSION__" not in html, \
        "The placeholder must always be substituted — if this ever leaks " \
        "into the rendered page, the version marker is useless"
    assert BUILD_VERSION in html, \
        "The actual build version string must appear in the rendered HTML"


def test_health_endpoint_reports_build_version():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["build_version"] == BUILD_VERSION


def test_build_version_visible_without_javascript():
    """
    THE core requirement: this must be visible in the raw server-rendered
    HTML, not injected by client-side JS — so it's checkable even if
    something else (like the chart) is broken client-side, and even via
    view-source or a simple curl, not just a fully-rendered browser.
    """
    html = client.get("/").text
    # the footer text containing the version must be in the raw HTML
    # response body itself, not inside a <script> block that runs later
    footer_start = html.find("Build:")
    assert footer_start != -1, "A 'Build:' label must be present in the raw HTML"
    # confirm it's outside any <script> tag by checking no unclosed script
    # tag precedes it without a closing tag in between
    preceding = html[:footer_start]
    assert preceding.count("<script") == preceding.count("</script>"), \
        "The build version must be in a script-independent constant part of the page"


if __name__ == "__main__":
    tests = [
        test_build_version_is_a_non_empty_string,
        test_dashboard_html_shows_real_build_version_not_placeholder,
        test_health_endpoint_reports_build_version,
        test_build_version_visible_without_javascript,
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
