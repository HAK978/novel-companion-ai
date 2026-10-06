"""Chapter filtering in vector search.

The `$lte` ceiling is the spoiler guarantee at the database level. The `$gte`
floor exists because chapter numbers in a query string contribute almost
nothing to an embedding, so without it a thematically similar passage from a
much earlier arc can outrank the range actually asked for.
"""

import pytest


def search(module, **kwargs):
    return module.search_chunks(collection_name="novel_1", **kwargs)


class FakeCollection:
    """Every window a query asks for, each from a different chunk of chapter 12."""

    def __init__(self, count=100):
        self._count = count
        self.last_query = None

    def count(self):
        return self._count

    def query(self, **kwargs):
        self.last_query = kwargs
        n = kwargs["n_results"]
        return {
            "metadatas": [[{"chapter_number": 12, "chapter_title": "Twelve", "chunk_index": i}
                           for i in range(n)]],
            "distances": [[0.25] * n],
        }

    def get(self, where, include):
        keys = [c["$and"] for c in where.get("$or", [where])]
        return {"documents": ["passage text"] * len(keys),
                "metadatas": [{"chapter_number": k[0]["chapter_number"],
                               "chunk_index": k[1]["chunk_index"]} for k in keys]}


@pytest.fixture
def collection(retrieval_search, monkeypatch):
    fake = FakeCollection()
    monkeypatch.setattr(retrieval_search, "get_collection", lambda name: fake)
    return fake


def test_ceiling_only_when_no_floor_given(retrieval_search, collection):
    search(retrieval_search, query="who is the protagonist", current_chapter=500)

    assert collection.last_query["where"] == {"chapter_number": {"$lte": 500}}


def test_floor_and_ceiling_when_range_requested(retrieval_search, collection):
    search(retrieval_search, 
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

    assert search(retrieval_search, query="anything", current_chapter=10) == []


def test_complex_queries_retrieve_more_context(retrieval_search, collection):
    simple = search(retrieval_search, query="who is Sunny", current_chapter=500, n_results=5)
    complex_ = search(retrieval_search, query="summarize the arc", current_chapter=500,
                      n_results=5)

    assert len(complex_) > len(simple)


def test_adaptive_retrieval_is_capped(retrieval_search, collection):
    results = search(retrieval_search, query="explain what happened", current_chapter=500,
                     n_results=8)

    assert len(results) <= 10


def test_results_carry_chapter_provenance(retrieval_search, collection):
    results = search(retrieval_search, query="anything", current_chapter=500)

    assert results[0]["chapter_number"] == 12
    assert results[0]["chapter_title"] == "Twelve"
    assert 0 <= results[0]["relevance_score"] <= 1


@pytest.mark.parametrize("question,broad", [
    ("Who is Cassie?", False),
    ("Summarize the Forgotten Shore arc", True),
    ("What happened at the Black Skull?", True),
    ("Tell me about the story so far", True),
    # substring matches that used to count as broad
    ("Who is the archer?", False),
    ("Search for Sunny's history", False),
    ("What did they eat on the march?", False),
])
def test_only_whole_words_mark_a_question_broad(retrieval_search, question, broad):
    assert retrieval_search.is_complex_query(question) is broad
