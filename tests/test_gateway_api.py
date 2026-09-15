"""Gateway orchestration, with the downstream services and database stubbed.

The gateway holds the logic worth testing: which chapter it forwards to
retrieval, how it labels context before handing it to the model, and whether
flags survive the hop to the ingestion service.
"""

import pytest
from fastapi.testclient import TestClient


class FakeResult:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self, statements, row=None):
        self._statements = statements
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self._statements.append({"sql": str(statement), "params": params})
        return FakeResult(self._row)

    def commit(self):
        pass


class FakeEngine:
    """Records SQL instead of touching PostgreSQL."""

    def __init__(self, row=None):
        self.statements = []
        self._row = row

    def connect(self):
        return FakeConnection(self.statements, self._row)


SEARCH_HIT = {
    "results": [
        {
            "text": "Sunny stepped through the gate.",
            "chapter_number": 1281,
            "chapter_title": "Leaving the Dark Island",
            "relevance_score": 0.91,
        }
    ]
}
GENERATED = {"answer": "He stepped through the gate.", "model_used": "test-model"}


@pytest.fixture
def client(gateway_main):
    return TestClient(gateway_main.app)


@pytest.fixture
def wire(gateway_main, monkeypatch, fake_async_client):
    """Patch downstream HTTP and the database; return recorded calls."""

    def _wire(routes, row=None):
        factory, calls = fake_async_client(routes)
        monkeypatch.setattr(gateway_main.httpx, "AsyncClient", factory)
        engine = FakeEngine(row)
        monkeypatch.setattr(gateway_main, "engine", engine)
        return calls, engine

    return _wire


def test_query_returns_empty_when_retrieval_finds_nothing(client, wire):
    wire({"/search": {"results": []}})

    body = client.post(
        "/query", json={"query": "who?", "novel_id": 1, "current_chapter": 10}
    ).json()

    assert body["answer"] is None
    assert body["sources"] == []


def test_query_forwards_reading_progress_to_retrieval(client, wire):
    calls, _ = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    client.post("/query", json={"query": "who?", "novel_id": 3, "current_chapter": 500})

    search = next(c for c in calls if "/search" in c["url"])
    assert search["payload"]["current_chapter"] == 500


def test_query_scopes_search_to_the_novels_collection(client, wire):
    calls, _ = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    client.post("/query", json={"query": "who?", "novel_id": 3, "current_chapter": 500})

    search = next(c for c in calls if "/search" in c["url"])
    assert search["payload"]["collection_name"] == "novel_3"


def test_query_labels_context_with_chapter_numbers(client, wire):
    # Unlabelled context lets the model merge passages from distant arcs into
    # one narrative, which is how cross-arc answers happen.
    calls, _ = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    client.post("/query", json={"query": "what happened?", "novel_id": 3, "current_chapter": 1291})

    generate = next(c for c in calls if "/generate" in c["url"])
    chunk = generate["payload"]["context_chunks"][0]
    assert chunk.startswith("[Chapter 1281: Leaving the Dark Island]")
    assert "Sunny stepped through the gate." in chunk


def test_query_returns_answer_and_sources(client, wire):
    wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    body = client.post(
        "/query", json={"query": "what happened?", "novel_id": 3, "current_chapter": 1291}
    ).json()

    assert body["answer"] == "He stepped through the gate."
    assert body["model_used"] == "test-model"
    assert body["sources"][0]["chapter_number"] == 1281


def test_query_is_recorded_in_search_history(client, wire):
    _, engine = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    client.post("/query", json={"query": "what happened?", "novel_id": 3, "current_chapter": 1291})

    assert any("search_history" in s["sql"] for s in engine.statements)


def test_ingest_from_source_defaults_extraction_off(client, wire):
    # Entity extraction costs ~20s per chapter; bulk ingestion must not opt in
    # by accident.
    calls, _ = wire({"/ingest/from-source": {"task_id": "abc", "status": "queued"}})

    client.post(
        "/ingest/from-source",
        json={"novel_id": 1, "source_type": "local_json", "source_path": "/tmp/x"},
    )

    assert calls[0]["payload"]["extract_entities"] is False


def test_ingest_from_source_forwards_extraction_flag(client, wire):
    calls, _ = wire({"/ingest/from-source": {"task_id": "abc", "status": "queued"}})

    client.post(
        "/ingest/from-source",
        json={
            "novel_id": 1,
            "source_type": "local_json",
            "source_path": "/tmp/x",
            "extract_entities": True,
        },
    )

    assert calls[0]["payload"]["extract_entities"] is True


def test_health_reports_unreachable_services_without_failing(client, gateway_main, monkeypatch):
    class Broken:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(gateway_main.httpx, "AsyncClient", lambda *a, **k: Broken())
    monkeypatch.setattr(gateway_main, "engine", FakeEngine())

    body = client.get("/health").json()

    assert body["services"]["retrieval"]["status"] == "unreachable"


# --- regression tests for an audit round ---
#
# Each of these reproduces a specific defect found by review: summaries reaching
# past the requested range, a cache key that collided across novels, and failed
# generations being stored and served as if they were content.


from conftest import FakeResponse  # noqa: E402

SUMMARY_REQUEST = {
    "novel_id": 3,
    "start_chapter": 100,
    "end_chapter": 120,
    "current_chapter": 200,
}


def test_summary_does_not_retrieve_past_the_requested_range(client, wire):
    # Reading position was used as the ceiling, so a summary of 100-120 pulled
    # in chapters up to 200.
    calls, _ = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    client.post("/summarize", json=SUMMARY_REQUEST)

    search = next(c for c in calls if "/search" in c["url"])
    assert search["payload"]["current_chapter"] == 120
    assert search["payload"]["min_chapter"] == 100


def test_summary_clamps_the_range_to_reading_progress(client, wire):
    calls, _ = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    body = client.post(
        "/summarize",
        json={"novel_id": 3, "start_chapter": 100, "end_chapter": 500, "current_chapter": 150},
    ).json()

    search = next(c for c in calls if "/search" in c["url"])
    assert search["payload"]["current_chapter"] == 150
    assert body["end_chapter"] == 150


def test_summary_refuses_a_range_entirely_unread(client, wire):
    calls, _ = wire({})

    body = client.post(
        "/summarize",
        json={"novel_id": 3, "start_chapter": 400, "end_chapter": 500, "current_chapter": 100},
    ).json()

    assert "error" in body
    assert calls == []  # nothing was retrieved or generated


def test_summary_cache_is_keyed_by_novel(client, wire):
    # Keying on the chapter range alone let one novel's summary overwrite
    # another's for the same range.
    _, engine = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    client.post("/summarize", json=SUMMARY_REQUEST)

    insert = next(s for s in engine.statements if "INSERT INTO chapter_summaries" in s["sql"])
    assert "ON CONFLICT (novel_id, start_chapter, end_chapter)" in insert["sql"]
    assert insert["params"]["nid"] == 3


def test_summary_cache_lookup_is_scoped_to_novel_and_clamped_range(client, wire):
    _, engine = wire({"/search": SEARCH_HIT, "/generate": GENERATED})

    client.post("/summarize", json=SUMMARY_REQUEST)

    select = next(s for s in engine.statements if "SELECT summary FROM chapter_summaries" in s["sql"])
    assert select["params"] == {"nid": 3, "s": 100, "e": 120}


def test_cached_summary_skips_the_services(client, wire):
    calls, _ = wire({}, row=("a cached summary",))

    body = client.post("/summarize", json=SUMMARY_REQUEST).json()

    assert body["cached"] is True
    assert body["summary"] == "a cached summary"
    assert calls == []


def test_failed_generation_is_not_cached_as_a_summary(client, wire):
    # Generation used to return "Error: ..." as ordinary text, which was then
    # stored and served forever as the summary for that range.
    _, engine = wire({"/search": SEARCH_HIT, "/generate": FakeResponse({"detail": "boom"}, 502)})

    response = client.post("/summarize", json=SUMMARY_REQUEST)

    assert response.status_code == 502
    assert not any("INSERT INTO chapter_summaries" in s["sql"] for s in engine.statements)


def test_query_surfaces_generation_failure(client, wire):
    wire({"/search": SEARCH_HIT, "/generate": FakeResponse({"detail": "boom"}, 502)})

    response = client.post(
        "/query", json={"query": "what happened?", "novel_id": 3, "current_chapter": 500}
    )

    assert response.status_code == 502
