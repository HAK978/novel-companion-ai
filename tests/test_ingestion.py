"""Ingestion API and the per-chapter pipeline, with Chroma/Postgres stubbed."""

import pytest
from conftest import load_module
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def ingestion_tasks():
    return load_module("ingestion", "tasks.py")


@pytest.fixture(scope="module")
def ingestion_main():
    return load_module("ingestion", "main.py")


@pytest.fixture
def queued(ingestion_main, monkeypatch):
    """Capture what gets handed to Celery instead of enqueueing it."""
    sent = []

    class FakeTask:
        def __init__(self, name):
            self.name = name

        def delay(self, payload):
            sent.append({"task": self.name, "payload": payload})
            return type("AsyncResult", (), {"id": "task-123"})()

    monkeypatch.setattr(ingestion_main, "ingest_chapter", FakeTask("ingest_chapter"))
    monkeypatch.setattr(ingestion_main, "ingest_from_source", FakeTask("ingest_from_source"))
    return sent


@pytest.fixture
def client(ingestion_main):
    return TestClient(ingestion_main.app)


def test_ingest_queues_a_task(client, queued):
    body = client.post("/ingest", json={"novel_id": 1, "number": 5, "content": "<p>text</p>"}).json()

    assert body["status"] == "queued"
    assert body["task_id"] == "task-123"
    assert len(queued) == 1


def test_ingest_defaults_missing_title(client, queued):
    client.post("/ingest", json={"novel_id": 1, "number": 5, "content": "<p>text</p>"})

    assert queued[0]["payload"]["title"] == "Chapter 5"


def test_ingest_defaults_extraction_off(client, queued):
    client.post("/ingest", json={"novel_id": 1, "number": 5, "content": "<p>text</p>"})

    assert queued[0]["payload"]["extract_entities"] is False


def test_ingest_from_source_passes_limits_through(client, queued):
    client.post(
        "/ingest/from-source",
        json={
            "novel_id": 2,
            "source_type": "local_json",
            "source_path": "/chapters",
            "max_chapters": 50,
            "extract_entities": True,
        },
    )

    payload = queued[0]["payload"]
    assert payload["max_chapters"] == 50
    assert payload["extract_entities"] is True


# --- per-chapter pipeline ---


class FakeCollection:
    def __init__(self):
        self.added = []

    def add(self, documents, metadatas, ids):
        self.added.append({"documents": documents, "metadatas": metadatas, "ids": ids})


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        return type("R", (), {"fetchone": lambda self: (1,)})()

    def commit(self):
        pass


@pytest.fixture
def pipeline(ingestion_tasks, monkeypatch):
    """Stub the vector store and database; record generation-service calls."""
    collection = FakeCollection()
    generation_calls = []

    monkeypatch.setattr(ingestion_tasks, "_get_collection", lambda name: collection)
    fake_engine = type("E", (), {"connect": lambda self: FakeConnection()})()
    monkeypatch.setattr(ingestion_tasks, "engine", fake_engine)

    def fake_post(url, **kwargs):
        generation_calls.append(url)
        return type("Resp", (), {"status_code": 200, "json": lambda self: {"characters": []}})()

    monkeypatch.setattr(ingestion_tasks.httpx, "post", fake_post)
    return collection, generation_calls


CHAPTER = {
    "novel_id": 1,
    "number": 7,
    "title": "Seven",
    "content": "<p>" + ("The shadow moved through the ruined street. " * 60) + "</p>",
}


def test_chapter_is_chunked_and_indexed(ingestion_tasks, pipeline):
    collection, _ = pipeline

    result = ingestion_tasks._process_chapter(dict(CHAPTER))

    assert result["status"] == "complete"
    assert result["chunks_created"] > 0
    assert collection.added


def test_every_chunk_carries_its_chapter_number(ingestion_tasks, pipeline):
    collection, _ = pipeline

    ingestion_tasks._process_chapter(dict(CHAPTER))

    metadatas = collection.added[0]["metadatas"]
    # Chapter metadata is what the spoiler filter queries against
    assert all(m["chapter_number"] == 7 for m in metadatas)


def test_extraction_is_skipped_by_default(ingestion_tasks, pipeline):
    _, generation_calls = pipeline

    result = ingestion_tasks._process_chapter(dict(CHAPTER))

    assert generation_calls == []
    assert result["entities_extracted"] == 0


def test_extraction_runs_when_requested(ingestion_tasks, pipeline):
    _, generation_calls = pipeline

    ingestion_tasks._process_chapter(dict(CHAPTER), extract_entities=True)

    assert any("/extract-entities" in url for url in generation_calls)


def test_empty_chapter_is_skipped(ingestion_tasks, pipeline):
    result = ingestion_tasks._process_chapter({"novel_id": 1, "number": 8, "content": ""})

    assert result["status"] == "skipped"
