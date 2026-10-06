"""Ingestion API and the per-chapter pipeline, with Chroma/Postgres stubbed."""

import pytest
from conftest import load_module
from fastapi.testclient import TestClient


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
        self.deleted = []

    def upsert(self, documents, metadatas, ids, embeddings=None):
        self.added.append({"documents": documents, "metadatas": metadatas, "ids": ids,
                           "embeddings": embeddings})

    def get(self, where, include):
        return {"ids": [i for u in self.added for i, m in zip(u["ids"], u["metadatas"], strict=True)
                        if m["chapter_number"] == where["chapter_number"]]}

    def delete(self, ids):
        self.deleted.append(ids)


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


def test_chunks_are_stored_as_windows_the_model_reads_whole(ingestion_tasks, pipeline):
    collection, _ = pipeline
    chapter = {**CHAPTER, "content": "<p>" + "The shadow moved through the ruined street. " * 150
               + "</p>"}

    result = ingestion_tasks._process_chapter(chapter)

    metas = collection.added[0]["metadatas"]
    docs = collection.added[0]["documents"]
    assert result["windows"] > result["chunks_created"] > 1
    assert all(len(d.split()) <= ingestion_tasks.WINDOW_SIZE for d in docs)
    # a chunk's windows, in order, are the chunk
    _, chunks = ingestion_tasks._chapter_chunks(chapter["content"])
    for i, chunk in enumerate(chunks):
        windows = sorted((m["window_index"], d) for m, d in zip(metas, docs, strict=True)
                         if m["chunk_index"] == i)
        assert " ".join(d for _, d in windows) == chunk


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


# --- ingesting a whole book ---

def book(*numbers):
    # ~900 words each: several chunks per chapter
    return [{"number": n, "title": f"Chapter {n}",
             "content": "<p>" + f"Chapter {n} goes on and on. " * 150 + "</p>"} for n in numbers]


@pytest.fixture
def grouped(ingestion_tasks, pipeline, monkeypatch):
    """Two chapters per embedding call, with a stand-in model whose vector for a text is the
    text itself, so each stored chunk shows where its vector came from. Records each
    embedding call, and which chapters were stored when summaries were scheduled."""
    collection, _ = pipeline
    embed_calls, scheduled = [], []

    def embed(texts):
        embed_calls.append(texts)
        return [("vector of", t) for t in texts]

    def schedule(novel_id, reader_chapter=None):
        stored = {m["chapter_number"] for u in collection.added for m in u["metadatas"]}
        scheduled.append({"reader": reader_chapter, "stored": sorted(stored)})

    monkeypatch.setattr(ingestion_tasks, "EMBED_GROUP", 2)
    monkeypatch.setattr(ingestion_tasks, "_embed", embed)
    monkeypatch.setattr(ingestion_tasks, "_schedule_quietly", schedule)
    return collection, embed_calls, scheduled


def test_chapters_are_embedded_a_group_at_a_time(ingestion_tasks, grouped):
    _, embed_calls, _ = grouped

    ingestion_tasks._ingest_chapters(1, book(1, 2, 3))

    # chapters 1-2 in one call, then chapter 3: not one call per chapter
    assert len(embed_calls) == 2


def test_every_chunk_is_stored_with_its_own_vector(ingestion_tasks, grouped):
    collection, _, _ = grouped

    ingestion_tasks._ingest_chapters(1, book(1, 2, 3))

    assert {m["chapter_number"] for u in collection.added for m in u["metadatas"]} == {1, 2, 3}
    for upsert in collection.added:
        assert upsert["embeddings"] == [("vector of", d) for d in upsert["documents"]]


def test_progress_says_how_far_the_book_is_searchable(ingestion_tasks, grouped):
    reports = []

    ingestion_tasks._ingest_chapters(1, book(1, 2, 3), report=reports.append)

    assert [(r["processed"], r["searchable_up_to"], r["total_chapters"]) for r in reports] == [
        (2, 2, 3), (3, 3, 3)]


@pytest.mark.parametrize("reader,stored_when_scheduled", [
    (3, [1, 2, 3, 4]),  # as soon as the reader's chapter is in, not after the whole book
    (0, [1, 2]),  # a reader who has not started: from the first chapters
])
def test_summaries_start_once_the_readers_chapters_are_in(ingestion_tasks, grouped, reader,
                                                          stored_when_scheduled):
    _, _, scheduled = grouped

    ingestion_tasks._ingest_chapters(1, book(1, 2, 3, 4, 5, 6), reader_chapter=reader)

    assert scheduled == [{"reader": reader, "stored": stored_when_scheduled}]


# --- checking chapter summaries ---

@pytest.mark.parametrize("summary,title,text,flagged", [
    # a name the chapter never uses is flagged: it may come from the model's memory
    ("Sunny crosses the bridge while Mordret watches.", "", "sunny crossed the bridge", ["Mordret"]),
    # the title counts as part of the chapter
    ('In Chapter 16, "Rebirth," Sunny changes.', "Chapter 16 Rebirth", "sunny changed", []),
    # words opening a sentence are capitalized anyway
    ("Sunny falls. Despite this, he rises.", "", "sunny fell and rose", []),
    # plurals and possessives match their stem
    ("The Baskervilles trust Barrymore's wife.", "", "a baskerville and barrymore", []),
    ("Sunny counts his Memories.", "", "sunny earned a memory", []),
    # markdown headings and list numbers still open a sentence
    ("**Summary:** Sunny rests.\n1. Despite everything, he wins.", "", "sunny rested and won", []),
    # nor are the words of the headings the model writes
    ("**Chapter 12 Summary (Part 1 of 2):** Watson waits.", "", "watson waited", []),
])
def test_summary_names_are_checked_against_the_chapter(ingestion_tasks, summary, title, text,
                                                       flagged):
    assert ingestion_tasks._unverified_names(summary, title, text) == flagged
