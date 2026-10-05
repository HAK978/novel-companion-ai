"""Move collections built with Chroma's implicit default embedding onto the pinned model.

Collections created without an explicit embedding function record it as "default", and
Chroma refuses to reopen them with an explicit one, even the identical model. Since the
pinned model (ONNX all-MiniLM-L6-v2) produces exactly the vectors the default did, this
copies stored vectors as they are, with no re-embedding.

One thing does change: implicit-default collections measure L2 distance, the pinned model
measures cosine. MiniLM vectors are unit length, so both metrics rank results identically
(checked on novel_3: 50 of 50 sampled queries, top 10). Scores change meaning: retrieval
reports 1 - distance, which under L2 was 2cos - 1 and under cosine is cosine similarity.

Steps:

  1. copy every record (ids, documents, metadata, embeddings) into a pinned collection
  2. verify: same record count, and every sampled chunk still finds itself among its
     nearest neighbours in the new collection
  3. only then rename: original -> <name>__legacy_default, copy -> <name>

The legacy collection is kept until removed with --drop-legacy.

Usage:
    python scripts/migrate_embeddings.py               # migrate every unpinned collection
    python scripts/migrate_embeddings.py --dry-run     # report only
    python scripts/migrate_embeddings.py --drop-legacy # delete kept legacy collections
"""

import argparse
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ingestion"))
from vector_store import embedding_function

LEGACY_SUFFIX = "__legacy_default"
STAGING_SUFFIX = "__pinning"
PAGE = 1000


class VerificationError(RuntimeError):
    pass


def _embedding_name(collection) -> str | None:
    config = collection.configuration_json or {}
    return (config.get("embedding_function") or {}).get("name")


def _read_all(collection):
    """Yield (ids, documents, metadatas, embeddings) pages."""
    offset = 0
    while True:
        page = collection.get(
            include=["documents", "metadatas", "embeddings"], limit=PAGE, offset=offset
        )
        if not page["ids"]:
            return
        yield page["ids"], page["documents"], page["metadatas"], page["embeddings"]
        offset += len(page["ids"])


def _verify(old, new, samples: int, out: Callable[[str], None]) -> None:
    if old.count() != new.count():
        raise VerificationError(f"record count differs: {old.count()} vs {new.count()}")

    sample = old.get(include=["embeddings"], limit=old.count())
    picks = random.Random(0).sample(range(len(sample["ids"])), min(samples, len(sample["ids"])))
    overlap = 0.0
    for i in picks:
        vector = [float(x) for x in sample["embeddings"][i]]
        k = min(5, old.count())
        old_top = old.query(query_embeddings=[vector], n_results=k)["ids"][0]
        new_top = new.query(query_embeddings=[vector], n_results=k)["ids"][0]
        if sample["ids"][i] not in new_top:
            raise VerificationError(f"{sample['ids'][i]} no longer finds itself in the copy")
        overlap += len(set(old_top) & set(new_top)) / k
    out(f"  verified {new.count()} records; top-5 neighbour overlap {overlap / len(picks):.0%}"
        f" over {len(picks)} sampled queries")


def migrate_collection(client, name: str, samples: int = 50, out: Callable[[str], None] = print) -> bool:
    """Migrate one collection if it is unpinned. Returns True if it was migrated."""
    target_fn = embedding_function()
    original = client.get_collection(name)
    current = _embedding_name(original)
    if current == target_fn.name():
        out(f"  {name}: already pinned ({current})")
        return False

    staging_name = name + STAGING_SUFFIX
    if staging_name in {c.name for c in client.list_collections()}:
        client.delete_collection(staging_name)  # leftover from an interrupted run
    staging = client.create_collection(staging_name, embedding_function=target_fn)

    for ids, documents, metadatas, embeddings in _read_all(original):
        staging.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    _verify(original, staging, samples, out)

    original.modify(name=name + LEGACY_SUFFIX)
    staging.modify(name=name)
    out(f"  {name}: {current!r} -> {target_fn.name()!r}; original kept as {name + LEGACY_SUFFIX}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drop-legacy", action="store_true")
    args = parser.parse_args()

    url = os.environ.get("CHROMADB_URL", "http://localhost:8005")
    host, port = url.replace("http://", "").split(":")
    client = chromadb.HttpClient(host=host, port=int(port))
    pinned = embedding_function().name()

    names = sorted(c.name for c in client.list_collections())
    if args.drop_legacy:
        for name in names:
            if name.endswith(LEGACY_SUFFIX):
                client.delete_collection(name)
                print(f"  dropped {name}")
        return

    for name in names:
        if name.endswith((LEGACY_SUFFIX, STAGING_SUFFIX)):
            continue
        collection = client.get_collection(name)
        if args.dry_run:
            state = "pinned" if _embedding_name(collection) == pinned else "needs migration"
            print(f"  {name}: {collection.count()} records, {_embedding_name(collection)!r} ({state})")
        else:
            migrate_collection(client, name)


if __name__ == "__main__":
    main()
