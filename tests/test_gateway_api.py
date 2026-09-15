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
