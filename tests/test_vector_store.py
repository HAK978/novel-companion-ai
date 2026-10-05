"""Embedding-model pinning, and moving an existing collection onto the pinned model.

Ingestion embeds chunks and retrieval embeds questions; if the two ever use different
models, search returns wrong passages with no error. These tests use a real (embedded)
Chroma instance; none of them computes embeddings, so the model is never downloaded.
"""

from pathlib import Path

import chromadb
import numpy as np
import pytest
from conftest import ROOT, load_module, load_script

COPIES = [ROOT / "services" / svc / "vector_store.py" for svc in ("ingestion", "retrieval")]
PINNED = "onnx_mini_lm_l6_v2"


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path))


@pytest.fixture(scope="module")
def vector_store():
    return load_module("ingestion", "vector_store.py", alias="ingestion_vector_store")


@pytest.fixture(scope="module")
def migrator():
    return load_script("migrate_embeddings.py", "novel_migrate_embeddings")


def embedding_name(collection):
    return (collection.configuration_json.get("embedding_function") or {}).get("name")


def legacy_collection(client, name, n=12):
    """A collection the way novel_3 was built: no embedding function given."""
    col = client.get_or_create_collection(name)
    col.add(
        ids=[f"ch_{i:04d}_chunk_000" for i in range(n)],
        documents=[f"passage {i}" for i in range(n)],
        metadatas=[{"chapter_number": i, "chunk_index": 0} for i in range(n)],
        # unit length, as the real model's output is (verified on novel_3)
        embeddings=[list(v / np.linalg.norm(v)) for v in
                    (np.array([float(i), float(i % 3), 1.0]) for i in range(n))],
    )
    return col


def test_both_services_carry_identical_copies():
    assert Path(COPIES[0]).read_text() == Path(COPIES[1]).read_text()


def test_both_services_pin_the_same_model():
    names = {
        load_module(svc, "vector_store.py", alias=f"{svc}_vector_store").embedding_function().name()
        for svc in ("ingestion", "retrieval")
    }
    assert names == {PINNED}


def test_new_collections_record_the_pinned_model(client, vector_store):
    col = vector_store.open_collection(client, "novel_1")

    assert embedding_name(col) == PINNED


def test_an_unpinned_collection_names_the_fix(client, vector_store):
    legacy_collection(client, "novel_3")

    with pytest.raises(vector_store.UnpinnedCollectionError, match="migrate_embeddings"):
        vector_store.open_collection(client, "novel_3")


def test_migration_copies_vectors_without_reembedding(client, migrator):
    before = legacy_collection(client, "novel_3").get(include=["documents", "metadatas", "embeddings"])

    assert migrator.migrate_collection(client, "novel_3", out=lambda _: None)

    after_col = client.get_collection("novel_3")
    after = after_col.get(ids=before["ids"], include=["documents", "metadatas", "embeddings"])
    assert embedding_name(after_col) == PINNED
    assert after["ids"] == before["ids"]
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]
    # same values to float32 precision: the pinned collection measures cosine rather than
    # L2 distance, and storing under it round-trips the floats
    assert np.allclose(after["embeddings"], before["embeddings"], atol=1e-6)


def test_migration_keeps_the_original_until_told_to_drop_it(client, migrator):
    legacy_collection(client, "novel_3")
    migrator.migrate_collection(client, "novel_3", out=lambda _: None)

    names = {c.name for c in client.list_collections()}
    assert names == {"novel_3", "novel_3__legacy_default"}
    assert client.get_collection("novel_3__legacy_default").count() == 12


def test_migrated_collection_opens_through_the_service_helper(client, migrator, vector_store):
    legacy_collection(client, "novel_3")
    migrator.migrate_collection(client, "novel_3", out=lambda _: None)

    assert vector_store.open_collection(client, "novel_3").count() == 12


def test_migration_leaves_pinned_collections_alone(client, migrator, vector_store):
    vector_store.open_collection(client, "novel_1")

    assert not migrator.migrate_collection(client, "novel_1", out=lambda _: None)
    assert {c.name for c in client.list_collections()} == {"novel_1"}
