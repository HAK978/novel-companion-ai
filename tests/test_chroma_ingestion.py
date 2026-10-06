"""Re-ingestion against a real (embedded) Chroma instance.

`add()` silently skips ids that already exist, so re-ingesting a chapter used to keep the
old text while reporting success. A stub cannot show that; real Chroma can. Postgres is
stubbed here, and a fake embedding model stands in since vector values do not matter.
"""

import chromadb
import pytest
from conftest import FakeEmbedding


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        return None

    def commit(self):
        pass


@pytest.fixture
def ingest(ingestion_tasks, tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    real_open = ingestion_tasks.open_collection
    monkeypatch.setattr(ingestion_tasks, "_chroma_client", client)
    monkeypatch.setattr(ingestion_tasks, "_collections", {})
    monkeypatch.setattr(ingestion_tasks, "open_collection",
                        lambda c, name: real_open(c, name, embedding_fn=FakeEmbedding()))
    monkeypatch.setattr(ingestion_tasks, "engine",
                        type("E", (), {"connect": lambda self: FakeConnection()})())

    def _ingest(number, words, novel_id=1, version="v1"):
        # every word carries the version, so stored text shows which ingestion it came from
        text = " ".join(f"{version}w{number}_{i}." for i in range(words))
        return ingestion_tasks._process_chapter(
            {"novel_id": novel_id, "number": number, "title": f"Chapter {number}",
             "content": f"<p>{text}</p>"}
        )

    def stored(number, novel_id=1):
        got = client.get_collection(f"novel_{novel_id}").get(where={"chapter_number": number})
        return sorted(got["documents"])

    _ingest.stored = stored
    _ingest.client = client
    return _ingest


def test_reingestion_replaces_the_text(ingest):
    # e.g. re-ingesting after a cleaning fix: same chunk ids, different text
    ingest(7, 900, version="old")
    ingest(7, 900, version="new")

    stored = ingest.stored(7)
    assert stored and all("new" in doc and "old" not in doc for doc in stored)


def test_reingestion_with_fewer_chunks_removes_the_surplus(ingest):
    assert ingest(7, 2000, version="old")["chunks_created"] > 2
    result = ingest(7, 450, version="new")

    stored = ingest.stored(7)
    assert len(stored) == result["windows"]
    assert not any("old" in doc for doc in stored)


def test_reingestion_leaves_other_chapters_alone(ingest):
    ingest(7, 900)
    ingest(8, 900)
    chapter_8 = ingest.stored(8)

    ingest(7, 450)

    assert ingest.stored(8) == chapter_8


def test_a_chapter_that_now_cleans_to_nothing_loses_its_old_chunks(ingest):
    ingest(7, 900)

    assert ingest(7, 0)["status"] == "skipped"
    assert ingest.stored(7) == []


def test_chapters_embedded_in_groups_match_chapters_embedded_alone(ingestion_tasks, ingest,
                                                                   monkeypatch):
    # bulk ingestion embeds many chapters' chunks per call and hands Chroma the vectors; they
    # must land on the same ids as when the collection embeds each chapter itself
    monkeypatch.setattr(ingestion_tasks, "_embedder", FakeEmbedding())
    monkeypatch.setattr(ingestion_tasks, "EMBED_GROUP", 2)
    monkeypatch.setattr(ingestion_tasks, "_schedule_quietly", lambda *args, **kwargs: {})
    chapters = [{"number": n, "title": f"Chapter {n}",
                 "content": "<p>" + " ".join(f"w{n}_{i}." for i in range(900)) + "</p>"}
                for n in (1, 2, 3)]

    ingestion_tasks._ingest_chapters(1, [dict(ch) for ch in chapters])
    for ch in chapters:
        ingestion_tasks._process_chapter({**ch, "novel_id": 2})

    def stored(novel_id):
        got = ingest.client.get_collection(f"novel_{novel_id}").get(
            include=["documents", "embeddings"])
        return {i: (doc, list(vector)) for i, doc, vector
                in zip(got["ids"], got["documents"], got["embeddings"], strict=True)}

    grouped = stored(1)
    assert len(grouped) > 3  # several chunks per chapter
    assert grouped == stored(2)


def test_chunks_stored_before_windows_are_replaced(ingestion_tasks, ingest):
    # books ingested before windows hold whole chunks; re-ingesting must leave only windows
    ingest(7, 10)  # creates the collection
    collection = ingest.client.get_collection("novel_1")
    collection.add(ids=["ch_0007_chunk_000", "ch_0007_chunk_001"],
                   documents=["an old whole chunk", "another old chunk"],
                   metadatas=[{"chapter_number": 7, "chunk_index": i} for i in range(2)],
                   embeddings=[[1.0, 2.0, 3.0]] * 2)

    ingest(7, 900, version="new")

    got = collection.get(where={"chapter_number": 7}, include=["metadatas"])
    assert all("_w" in i for i in got["ids"])
    assert all("window_index" in m for m in got["metadatas"])


def test_the_summary_worker_reads_a_chapter_back_whole(ingestion_tasks, ingest):
    ingest(7, 900)
    text = " ".join(f"v1w7_{i}." for i in range(900))

    assert ingestion_tasks._chapter_text(1, 7) == ("Chapter 7", text)


def test_a_chapter_without_a_novel_is_rejected(ingestion_tasks):
    # it used to land in a fallback collection that no query could ever reach
    with pytest.raises(ValueError, match="novel_id"):
        ingestion_tasks._process_chapter({"number": 1, "content": "<p>text</p>"})
