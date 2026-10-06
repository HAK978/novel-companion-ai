import re
from typing import Any

import chromadb
from config import CHROMADB_URL
from vector_store import open_collection

# Broad questions fetch more passages. Matched as whole words: as substrings, "arc" hit
# "search", "archer" and "march", and "story" hit "history". Whether the extra passages
# help at all is unmeasured; the evaluation harness should decide if this stays.
_COMPLEX = re.compile(
    r"\b(?:summari[sz]e|explain|arcs?|stor(?:y|ies)|what happened|tell me about)\b",
    re.IGNORECASE,
)


# Chunks are indexed as windows the embedding model reads whole: it ignores text past 256
# tokens, about half of a 400-word chunk. A search ranks windows and returns the chunks they
# belong to, best first, each once. Measured (eval/experiments/chunk_windows.py), the answer
# was retrieved for 17 of 25 Hound fact questions instead of 12, and for 140 of 262 Shadow
# Slave questions instead of 110. Windows fetched per chunk wanted: several from one chunk
# must still leave enough distinct chunks.
WINDOWS_PER_CHUNK = 8


# One client per process; per-request clients leak server-side connections
# (ChromaDB has a 1024-FD ulimit by default).
_client = None
_collections: dict = {}


def _get_client():
    global _client
    if _client is None:
        host = CHROMADB_URL.replace("http://", "").split(":")[0]
        port = int(CHROMADB_URL.split(":")[-1])
        _client = chromadb.HttpClient(host=host, port=port)
    return _client


def get_collection(collection_name: str):
    if collection_name not in _collections:
        _collections[collection_name] = open_collection(_get_client(), collection_name)
    return _collections[collection_name]


def is_complex_query(query: str) -> bool:
    return bool(_COMPLEX.search(query))


def search_chunks(
    query: str,
    current_chapter: int,
    n_results: int = 5,
    *,
    collection_name: str,
    min_chapter: int | None = None,
) -> list[dict[str, Any]]:
    """min_chapter sets a floor for range-scoped questions (e.g. summaries).
    Without it, similarity search is arc-blind: thematically similar chunks
    from hundreds of chapters earlier outrank the relevant window — numbers
    in the query text contribute nothing to the embedding."""
    collection = get_collection(collection_name)

    if collection.count() == 0:
        return []

    if is_complex_query(query):
        n_results = min(n_results * 2, 10)

    if min_chapter is not None:
        where_filter = {"$and": [
            {"chapter_number": {"$gte": min_chapter}},
            {"chapter_number": {"$lte": current_chapter}},
        ]}
    else:
        where_filter = {"chapter_number": {"$lte": current_chapter}}

    results = collection.query(
        query_texts=[query],
        n_results=n_results * WINDOWS_PER_CHUNK,
        where=where_filter,
        include=["metadatas", "distances"],
    )

    # each chunk once, ranked by its best window
    best: dict[tuple[int, int], tuple[str, float]] = {}
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0], strict=True):
        key = (metadata["chapter_number"], metadata.get("chunk_index", 0))
        if key not in best:
            best[key] = (metadata.get("chapter_title", ""), distance)
            if len(best) == n_results:
                break

    texts = _chunk_texts(collection, list(best))
    return [{
        "text": texts.get(key, ""),
        "chapter_number": key[0],
        "chapter_title": title,
        # cosine distance under the pinned model, so this is cosine similarity
        "relevance_score": round(1 - distance, 4),
    } for key, (title, distance) in best.items()]


def _chunk_texts(collection, keys: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    """Whole chunks, put back together from their windows. A chunk stored before windows is
    its own single window."""
    if not keys:
        return {}
    conditions = [{"$and": [{"chapter_number": chapter}, {"chunk_index": chunk}]}
                  for chapter, chunk in keys]
    got = collection.get(where=conditions[0] if len(conditions) == 1 else {"$or": conditions},
                         include=["documents", "metadatas"])
    windows: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for document, metadata in zip(got["documents"], got["metadatas"], strict=True):
        key = (metadata["chapter_number"], metadata.get("chunk_index", 0))
        windows.setdefault(key, []).append((metadata.get("window_index", 0), document))
    return {key: " ".join(text for _, text in sorted(parts)) for key, parts in windows.items()}


def delete_collection(collection_name: str):
    client = _get_client()
    try:
        client.delete_collection(collection_name)
        _collections.pop(collection_name, None)
        return True
    except Exception:
        return False
