"""Chapter filtering in vector search.

The `$lte` ceiling is the spoiler guarantee at the database level. The `$gte`
floor exists because chapter numbers in a query string contribute almost
nothing to an embedding, so without it a thematically similar passage from a
much earlier arc can outrank the range actually asked for.
"""

import pytest


class FakeCollection:
    def __init__(self, count=100):
        self._count = count
        self.last_query = None

    def count(self):
        return self._count

    def query(self, **kwargs):
        self.last_query = kwargs
        return {
            "documents": [["passage text"]],
            "metadatas": [[{"chapter_number": 12, "chapter_title": "Twelve"}]],
            "distances": [[0.25]],
        }


@pytest.fixture
def collection(retrieval_search, monkeypatch):
    fake = FakeCollection()
    monkeypatch.setattr(retrieval_search, "get_collection", lambda name: fake)
    return fake


def test_ceiling_only_when_no_floor_given(retrieval_search, collection):
    retrieval_search.search_chunks(query="who is the protagonist", current_chapter=500)

    assert collection.last_query["where"] == {"chapter_number": {"$lte": 500}}


def test_floor_and_ceiling_when_range_requested(retrieval_search, collection):
    retrieval_search.search_chunks(
        query="summary of events", current_chapter=1291, min_chapter=1281
    )

    assert collection.last_query["where"] == {
        "$and": [
            {"chapter_number": {"$gte": 1281}},
            {"chapter_number": {"$lte": 1291}},
        ]
    }


def test_empty_collection_returns_no_results(retrieval_search, monkeypatch):
    monkeypatch.setattr(retrieval_search, "get_collection", lambda name: FakeCollection(count=0))

    assert retrieval_search.search_chunks(query="anything", current_chapter=10) == []


def test_complex_queries_retrieve_more_context(retrieval_search, collection):
    retrieval_search.search_chunks(query="who is Sunny", current_chapter=500, n_results=5)
    simple = collection.last_query["n_results"]

    retrieval_search.search_chunks(query="summarize the arc", current_chapter=500, n_results=5)
    complex_ = collection.last_query["n_results"]

    assert complex_ > simple


def test_adaptive_retrieval_is_capped(retrieval_search, collection):
    retrieval_search.search_chunks(query="explain what happened", current_chapter=500, n_results=8)

    assert collection.last_query["n_results"] <= 10


def test_results_carry_chapter_provenance(retrieval_search, collection):
    results = retrieval_search.search_chunks(query="anything", current_chapter=500)

    assert results[0]["chapter_number"] == 12
    assert results[0]["chapter_title"] == "Twelve"
    assert 0 <= results[0]["relevance_score"] <= 1
