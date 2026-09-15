from typing import List, Dict, Any
import chromadb
from config import CHROMADB_URL


COMPLEX_KEYWORDS = [
    "summarize", "explain", "arc", "story", "what happened", "tell me about"
]


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


def get_collection(collection_name: str = "shadow_slave"):
    if collection_name not in _collections:
        _collections[collection_name] = _get_client().get_or_create_collection(collection_name)
    return _collections[collection_name]


def is_complex_query(query: str) -> bool:
    return any(kw in query.lower() for kw in COMPLEX_KEYWORDS)


def search_chunks(
    query: str,
    current_chapter: int,
    n_results: int = 5,
    collection_name: str = "shadow_slave",
    min_chapter: int = None,
) -> List[Dict[str, Any]]:
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
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    if results and results["documents"][0]:
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            formatted.append({
                "text": doc,
                "chapter_number": metadata["chapter_number"],
                "chapter_title": metadata.get("chapter_title", ""),
                "relevance_score": round(1 - distance, 4),
            })

    return formatted


def delete_collection(collection_name: str):
    client = _get_client()
    try:
        client.delete_collection(collection_name)
        _collections.pop(collection_name, None)
        return True
    except Exception:
        return False
