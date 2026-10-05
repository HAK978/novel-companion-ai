"""Opening vector collections with the embedding model pinned in code.

Ingestion embeds chunks and retrieval embeds questions, and both must use the same model:
vectors from two different models compare without any error, only with wrong results.
Identical copies of this file live in services/ingestion and services/retrieval, since the
services are built and deployed separately; tests/test_vector_store.py fails if they drift.

The model is deliberately not an environment variable. Changing it means re-ingesting into
a new collection, so it should never be a one-line config flip.
"""

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_MODELS = {"all-MiniLM-L6-v2": ONNXMiniLM_L6_V2}


class UnpinnedCollectionError(RuntimeError):
    """A collection built with a different embedding function than the pinned one."""


def embedding_function(model: str = EMBEDDING_MODEL):
    try:
        return _MODELS[model]()
    except KeyError:
        raise ValueError(f"unknown embedding model {model!r}; known: {sorted(_MODELS)}") from None


def open_collection(client, name: str, embedding_fn=None):
    """Get or create `name` with the pinned embedding function.

    Collections created before the model was pinned record their function as "default".
    Chroma refuses to reopen those with an explicit one, so this names the fix instead of
    surfacing an opaque ValueError.
    """
    embedding_fn = embedding_fn or embedding_function()
    try:
        return client.get_or_create_collection(name, embedding_function=embedding_fn)
    except ValueError:
        config = client.get_collection(name).configuration_json or {}
        persisted = (config.get("embedding_function") or {}).get("name")
        if persisted and persisted != embedding_fn.name():
            raise UnpinnedCollectionError(
                f"collection {name!r} was built with embedding function {persisted!r}, not "
                f"{embedding_fn.name()!r}. Run scripts/migrate_embeddings.py to move it over "
                f"without re-embedding."
            ) from None
        raise
