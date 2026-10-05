"""End to end, against a running stack.

Skipped unless RUN_INTEGRATION=1. The tests create their own novel from synthetic chapters,
so they need no existing data, and they delete it afterwards:

    docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --build --wait
    RUN_INTEGRATION=1 NOVEL_DATA_DIR=<host dir mounted at /novels> pytest tests/test_integration.py

With the stub model from docker-compose.ci.yml, answers list the chapters the model was
shown, so the spoiler rule is checked through the whole path: gateway, retrieval,
generation, model.
"""

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("RUN_INTEGRATION"),
                       reason="needs a running stack; set RUN_INTEGRATION=1"),
]

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8000")
DATA_DIR = os.environ.get("NOVEL_DATA_DIR")
CHAPTERS = 12
STUB = os.environ.get("LLM_STUB", "1") == "1"


def chapter_body(n):
    # distinct per chapter, long enough to make a real chunk
    sentence = (f"In chapter {n} the lighthouse keeper Orin counted {n} ships passing the "
                f"northern reef, and wrote the number {n} in his ledger before the storm. ")
    return "<p>" + sentence * 6 + "</p>"


def chapters_seen(answer):
    match = re.search(r"chapters seen: ([\d,]*)", answer or "")
    assert match, f"not a stub answer: {answer!r}"
    return [int(n) for n in match.group(1).split(",") if n]


def wait_for(task_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = httpx.get(f"{GATEWAY}/ingest/status/{task_id}", timeout=10).json()
        if status["status"] in ("SUCCESS", "FAILURE"):
            return status
        time.sleep(2)
    raise TimeoutError(f"ingestion {task_id} did not finish")


@pytest.fixture(scope="module")
def novel():
    if not DATA_DIR:
        pytest.skip("NOVEL_DATA_DIR must name the host directory mounted at /novels")
    folder = Path(DATA_DIR) / f"e2e-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    for n in range(1, CHAPTERS + 1):
        (folder / f"{n:03d}.json").write_text(json.dumps(
            {"title": f"Chapter {n}", "serial": n, "content": chapter_body(n)}))

    novel_id = httpx.post(f"{GATEWAY}/novels", json={"title": "E2E", "source_type": "local_json"},
                          timeout=30).json()["id"]
    task = httpx.post(f"{GATEWAY}/ingest/from-source", timeout=30, json={
        "novel_id": novel_id, "source_type": "local_json", "source_path": f"/novels/{folder.name}",
    }).json()
    result = wait_for(task["task_id"])
    assert result["status"] == "SUCCESS", result

    yield novel_id

    httpx.delete(f"{GATEWAY}/novels/{novel_id}", timeout=30)
    shutil.rmtree(folder, ignore_errors=True)


def test_every_service_is_healthy():
    body = httpx.get(f"{GATEWAY}/health", timeout=15).json()
    assert body["status"] == "ok", body


def test_every_chapter_was_ingested(novel):
    novels = {n["id"]: n for n in httpx.get(f"{GATEWAY}/novels", timeout=10).json()}
    assert novels[novel]["total_chapters"] == CHAPTERS


def test_sources_never_come_from_unread_chapters(novel):
    body = httpx.post(f"{GATEWAY}/query", timeout=120, json={
        "query": "How many ships did Orin count?", "novel_id": novel, "current_chapter": 4,
    }).json()

    assert body["sources"]
    assert all(s["chapter_number"] <= 4 for s in body["sources"]), body["sources"]


@pytest.mark.skipif(not STUB, reason="checks the stub model's echo of what it was shown")
def test_the_model_is_never_shown_unread_chapters(novel):
    body = httpx.post(f"{GATEWAY}/query", timeout=120, json={
        "query": "What happened before the storm?", "novel_id": novel, "current_chapter": 4,
    }).json()

    seen = chapters_seen(body["answer"])
    assert seen and max(seen) <= 4, seen


def test_summary_stays_within_the_requested_range(novel):
    body = httpx.post(f"{GATEWAY}/summarize", timeout=120, json={
        "novel_id": novel, "start_chapter": 2, "end_chapter": 5, "current_chapter": 10,
    }).json()

    # reading further than the requested range must not widen it
    assert (body["start_chapter"], body["end_chapter"]) == (2, 5), body
    if STUB:
        assert set(chapters_seen(body["summary"])) <= {2, 3, 4, 5}


def test_summary_is_clamped_to_reading_progress(novel):
    body = httpx.post(f"{GATEWAY}/summarize", timeout=120, json={
        "novel_id": novel, "start_chapter": 2, "end_chapter": 50, "current_chapter": 6,
    }).json()

    assert body["end_chapter"] == 6, body
    if STUB:
        assert max(chapters_seen(body["summary"])) <= 6


def test_progress_round_trips(novel):
    httpx.post(f"{GATEWAY}/progress", json={"novel_id": novel, "current_chapter": 7}, timeout=10)

    body = httpx.get(f"{GATEWAY}/progress/{novel}/default", timeout=10).json()

    assert body["current_chapter"] == 7


def test_every_chapter_gets_a_summary(novel):
    # written in the background after ingestion, by the summary worker through the model
    deadline = time.time() + 120
    while time.time() < deadline:
        status = httpx.get(f"{GATEWAY}/novels/{novel}/summaries", timeout=10).json()
        if status["summarized"] == CHAPTERS:
            break
        time.sleep(2)

    assert status["summarized"] == CHAPTERS, status
    assert status["highest_chapter"] == CHAPTERS


def test_deleting_a_novel_removes_it():
    novel_id = httpx.post(f"{GATEWAY}/novels", json={"title": "Doomed"}, timeout=30).json()["id"]

    httpx.delete(f"{GATEWAY}/novels/{novel_id}", timeout=30)

    assert novel_id not in {n["id"] for n in httpx.get(f"{GATEWAY}/novels", timeout=10).json()}
