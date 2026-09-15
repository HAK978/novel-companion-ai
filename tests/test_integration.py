"""End-to-end checks against a running stack.

Skipped unless RUN_INTEGRATION=1, since these need the services, databases and
a loaded model. CI has no GPU, so it runs everything except these.

    RUN_INTEGRATION=1 pytest tests/test_integration.py
"""

import os

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION"),
        reason="needs a running stack; set RUN_INTEGRATION=1",
    ),
]

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def novel_id():
    """First novel with chapters ingested, or skip."""
    novels = httpx.get(f"{GATEWAY}/novels", timeout=10).json()
    ingested = [n for n in novels if n.get("total_chapters", 0) > 0]
    if not ingested:
        pytest.skip("no ingested novel available")
    return ingested[0]["id"]


def test_health_reports_all_services_up():
    body = httpx.get(f"{GATEWAY}/health", timeout=15).json()

    assert body["status"] == "ok", body


def test_search_never_returns_unread_chapters(novel_id):
    httpx.post(
        f"{GATEWAY}/progress",
        json={"novel_id": novel_id, "current_chapter": 200},
        timeout=10,
    )

    body = httpx.post(
        f"{GATEWAY}/query",
        json={"query": "what has happened so far", "novel_id": novel_id, "current_chapter": 200},
        timeout=300,
    ).json()

    assert all(s["chapter_number"] <= 200 for s in body["sources"]), body["sources"]


def test_summary_stays_within_the_requested_range(novel_id):
    body = httpx.post(
        f"{GATEWAY}/summarize",
        json={
            "novel_id": novel_id,
            "start_chapter": 100,
            "end_chapter": 120,
            "current_chapter": 200,
        },
        timeout=300,
    ).json()

    assert "summary" in body, body
    # Reading further than the requested range must not widen the summary
    assert body["start_chapter"] == 100, body
    assert body["end_chapter"] == 120, body


def test_summary_is_clamped_to_reading_progress(novel_id):
    body = httpx.post(
        f"{GATEWAY}/summarize",
        json={
            "novel_id": novel_id,
            "start_chapter": 100,
            "end_chapter": 500,
            "current_chapter": 150,
        },
        timeout=300,
    ).json()

    assert body["end_chapter"] == 150, body


def test_progress_round_trips(novel_id):
    httpx.post(
        f"{GATEWAY}/progress",
        json={"novel_id": novel_id, "current_chapter": 321},
        timeout=10,
    )

    body = httpx.get(f"{GATEWAY}/progress/{novel_id}/default", timeout=10).json()

    assert body["current_chapter"] == 321
