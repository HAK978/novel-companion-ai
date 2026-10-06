"""Search ranks windows and returns whole chunks, against a real (embedded) Chroma.

Chunks are stored as windows the embedding model reads whole (it ignores text past 256
tokens). A stand-in model embeds by keyword here, so which window matches a question is
known: a window with the question's keyword is closest, one with no keyword next.
"""

import uuid

import chromadb
import pytest
from conftest import FakeEmbedding

KEYWORDS = ("alpha", "beta")


class KeywordEmbedding(FakeEmbedding):
    def __call__(self, input):
        return [[float(k in text) for k in KEYWORDS] + [0.1] for text in input]


@pytest.fixture
def store(retrieval_search, monkeypatch):
    client = chromadb.EphemeralClient()  # in-process clients share their collections
    name = f"windows_{uuid.uuid4().hex}"
    collection = client.create_collection(name, embedding_function=KeywordEmbedding())
    monkeypatch.setattr(retrieval_search, "get_collection", lambda _: collection)

    def add(chapter, chunk, windows, window_index=True):
        for j, text in enumerate(windows):
            meta = {"chapter_number": chapter, "chapter_title": f"Chapter {chapter}",
                    "chunk_index": chunk}
            if window_index:
                meta["window_index"] = j
            collection.add(ids=[f"{chapter}-{chunk}-{j}"], documents=[text], metadatas=[meta])
    yield add
    client.delete_collection(name)


def search(module, query, current_chapter=10, n_results=5):
    return module.search_chunks(query=query, current_chapter=current_chapter,
                                n_results=n_results, collection_name="any")


def test_a_match_late_in_a_chunk_returns_the_whole_chunk(retrieval_search, store):
    # the reason for windows: a whole 400-word chunk embeds only its first ~200 words
    store(3, 0, ["The chunk opens on the moor.", "Its end holds the alpha clue."])
    store(4, 0, ["Something about beta."])

    results = search(retrieval_search, "alpha", n_results=1)

    assert [(r["chapter_number"], r["text"]) for r in results] == [
        (3, "The chunk opens on the moor. Its end holds the alpha clue.")]


def test_each_chunk_comes_once_ranked_by_its_best_window(retrieval_search, store):
    store(1, 0, ["alpha first", "alpha again", "and alpha a third time"])
    store(2, 0, ["no keyword here"])
    store(2, 1, ["only beta here"])

    results = search(retrieval_search, "alpha", n_results=2)

    assert [(r["chapter_number"], r["text"][:11]) for r in results] == [
        (1, "alpha first"), (2, "no keyword ")]


def test_windows_past_the_readers_chapter_are_never_returned(retrieval_search, store):
    store(9, 0, ["the alpha reveal"])
    store(2, 0, ["no keyword here"])

    results = search(retrieval_search, "alpha", current_chapter=5)

    assert [r["chapter_number"] for r in results] == [2]


def test_chunks_stored_before_windows_still_work(retrieval_search, store):
    # until a book is re-ingested its chunks are whole, with no window index
    store(1, 0, ["A whole chunk with the alpha clue."], window_index=False)

    assert search(retrieval_search, "alpha")[0]["text"] == "A whole chunk with the alpha clue."
