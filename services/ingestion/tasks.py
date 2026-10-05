from datetime import UTC, datetime

import chromadb
import httpx
from celery import Celery
from chunking import chunk_text, clean_html
from config import (
    CHROMADB_URL,
    CHUNK_SIZE,
    DATABASE_URL,
    GENERATION_SERVICE_URL,
    REDIS_URL,
)
from sqlalchemy import create_engine
from sqlalchemy import text as sa_text
from vector_store import UnpinnedCollectionError, open_collection

celery_app = Celery("ingestion", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

engine = create_engine(DATABASE_URL)


# One client per worker process. Creating a client per chapter leaks
# server-side connections: ChromaDB held ~2 FDs per client and died at its
# 1024-FD ulimit around chapter ~490 on bulk ingests.
_chroma_client = None
_collections: dict = {}


def _get_collection(collection_name: str):
    global _chroma_client
    if _chroma_client is None:
        host = CHROMADB_URL.replace("http://", "").split(":")[0]
        port = int(CHROMADB_URL.split(":")[-1])
        _chroma_client = chromadb.HttpClient(host=host, port=port)
    if collection_name not in _collections:
        _collections[collection_name] = open_collection(_chroma_client, collection_name)
    return _collections[collection_name]


def _replace_chapter_chunks(collection_name, chapter_num, ids, chunks, metadatas):
    """Make the stored chunks for one chapter exactly `chunks`.

    `add()` silently skips ids that already exist, so re-ingesting a chapter used to keep its
    old text, plus any surplus old chunks when the new chunking yielded fewer, while the task
    reported success. Upserting replaces existing ids; pruning removes the surplus. In that
    order a crash midway leaves extra chunks at worst, never a chapter with none.
    """
    batch_size = 100

    def write(collection):
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.upsert(
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        collection.delete(where={"$and": [
            {"chapter_number": chapter_num},
            {"chunk_index": {"$gte": len(chunks)}},
        ]})

    try:
        write(_get_collection(collection_name))
    except UnpinnedCollectionError:
        raise
    except Exception:
        # the cached handle may be stale (collection deleted and recreated); refresh once
        _collections.pop(collection_name, None)
        write(_get_collection(collection_name))


def _process_chapter(chapter_data: dict, extract_entities: bool = False) -> dict:
    """
    Core logic: clean HTML, chunk, embed, store.

    chapter_data: {"novel_id": int, "number": int, "title": str, "content": str, "volume": int}

    extract_entities: when True, also run LLM character extraction via the
    generation service (~15-20s/chapter). Off by default; recall queries
    answer character questions from RAG without pre-extracted entities.
    """
    novel_id = chapter_data.get("novel_id")
    if not novel_id:
        # There used to be a fallback collection named after the first novel ever ingested;
        # chapters without a novel landed there, unreachable by any query.
        raise ValueError("novel_id is required to ingest a chapter")
    chapter_num = chapter_data["number"]
    title = chapter_data.get("title", f"Chapter {chapter_num}")
    volume = chapter_data.get("volume", 1)

    collection_name = f"novel_{novel_id}"

    # 1. Clean HTML
    text = clean_html(chapter_data["content"])
    word_count = len(text.split())

    # 2. Chunk text
    chunks = chunk_text(text, CHUNK_SIZE)
    if not chunks:
        # a re-ingested chapter that now cleans to nothing must not keep its old chunks
        _replace_chapter_chunks(collection_name, chapter_num, [], [], [])
        return {"status": "skipped", "chapter": chapter_num, "reason": "no content"}

    # 3. Store in ChromaDB

    ids = [f"ch_{chapter_num:04d}_chunk_{i:03d}" for i in range(len(chunks))]
    metadatas = [
        {
            "chapter_number": chapter_num,
            "chapter_title": title,
            "chunk_index": i,
            "volume": volume,
        }
        for i in range(len(chunks))
    ]

    _replace_chapter_chunks(collection_name, chapter_num, ids, chunks, metadatas)

    # 4. Update PostgreSQL
    with engine.connect() as conn:
        conn.execute(
            sa_text("""
                INSERT INTO chapters
                    (novel_id, chapter_number, title, word_count, volume, ingestion_status, ingested_at)
                VALUES (:nid, :num, :title, :wc, :vol, 'complete', :ts)
                ON CONFLICT (novel_id, chapter_number)
                DO UPDATE SET ingestion_status = 'complete', ingested_at = :ts, word_count = :wc
            """),
            {
                "nid": novel_id,
                "num": chapter_num,
                "title": title,
                "wc": word_count,
                "vol": volume,
                "ts": datetime.now(UTC),
            },
        )

        # Update novel chapter count
        if novel_id:
            conn.execute(
                sa_text("""
                    UPDATE novels SET total_chapters = (
                        SELECT COUNT(*) FROM chapters WHERE novel_id = :nid AND ingestion_status = 'complete'
                    ) WHERE id = :nid
                """),
                {"nid": novel_id},
            )

        conn.commit()

    # 5. Optionally extract character entities via generation service
    entities_extracted = 0
    if extract_entities:
        try:
            resp = httpx.post(
                f"{GENERATION_SERVICE_URL}/extract-entities",
                json={"chapter_text": text[:6000], "chapter_number": chapter_num},
                timeout=90,
            )
            if resp.status_code == 200:
                characters = resp.json().get("characters", [])
                entities_extracted = _store_entities(novel_id, chapter_num, characters)
        except Exception:
            pass  # Entity extraction is best-effort, don't fail ingestion

    return {
        "status": "complete",
        "novel_id": novel_id,
        "chapter": chapter_num,
        "chunks_created": len(chunks),
        "word_count": word_count,
        "entities_extracted": entities_extracted,
    }


# Characters can be extracted from chapters in any order (re-ingestion, parallel workers),
# so "first seen" values keep the earliest chapter rather than whichever arrived first. The
# description comes from the earliest chapter too: one written from a later chapter can
# carry what that chapter revealed, and an earlier chapter arriving late resets it.
_UPSERT_CHARACTER = sa_text("""
    INSERT INTO characters (novel_id, name, first_appearance, description, updated_at)
    VALUES (:nid, :name, :ch, :desc, :ts)
    ON CONFLICT (novel_id, name) DO UPDATE SET
        description = CASE
            WHEN EXCLUDED.first_appearance < characters.first_appearance
                THEN EXCLUDED.description
            WHEN EXCLUDED.first_appearance = characters.first_appearance
                THEN COALESCE(characters.description, EXCLUDED.description)
            ELSE characters.description
        END,
        first_appearance = LEAST(characters.first_appearance, EXCLUDED.first_appearance),
        updated_at = EXCLUDED.updated_at
    RETURNING id
""")

# Each alias records the chapter that revealed it; reads hide it before then.
_UPSERT_ALIAS = sa_text("""
    INSERT INTO character_aliases (character_id, alias, first_chapter)
    VALUES (:cid, :alias, :ch)
    ON CONFLICT (character_id, alias) DO UPDATE SET
        first_chapter = LEAST(character_aliases.first_chapter, EXCLUDED.first_chapter)
""")

_INSERT_MENTION = sa_text("""
    INSERT INTO character_mentions (novel_id, character_id, chapter_number, context)
    VALUES (:nid, :cid, :ch, :ctx)
    ON CONFLICT (character_id, chapter_number) DO NOTHING
""")

_UPSERT_RELATIONSHIP = sa_text("""
    INSERT INTO character_relationships
        (novel_id, character_a_id, character_b_id, relationship_type, first_chapter, updated_at)
    VALUES (:nid, :a, :b, :rel, :ch, :ts)
    ON CONFLICT (character_a_id, character_b_id, relationship_type) DO UPDATE SET
        first_chapter = LEAST(character_relationships.first_chapter, EXCLUDED.first_chapter),
        updated_at = EXCLUDED.updated_at
""")


def _store_entities(novel_id: int, chapter_num: int, characters: list) -> int:
    """Store one chapter's extracted characters, aliases, mentions and relationships."""
    if not characters or not novel_id:
        return 0

    now = datetime.now(UTC)
    stored = 0
    with engine.connect() as conn:
        for char in characters:
            name = (char.get("name") or "").strip()
            if len(name) < 2:
                continue
            role = (char.get("role") or "").strip() or None

            char_id = conn.execute(
                _UPSERT_CHARACTER,
                {"nid": novel_id, "name": name, "ch": chapter_num, "desc": role, "ts": now},
            ).scalar_one()

            aliases = {
                alias.strip()
                for alias in char.get("aliases") or []
                if isinstance(alias, str)
                and alias.strip()
                and alias.strip().lower() != name.lower()
            }
            for alias in sorted(aliases):
                conn.execute(_UPSERT_ALIAS, {"cid": char_id, "alias": alias, "ch": chapter_num})

            conn.execute(
                _INSERT_MENTION,
                {"nid": novel_id, "cid": char_id, "ch": chapter_num, "ctx": (role or "")[:200]},
            )

            for rel in char.get("relationships") or []:
                other_name = (rel.get("character") or "").strip()
                if len(other_name) < 2 or other_name == name:
                    continue
                # Known only as a relationship target so far: no description of its own,
                # which also clears a later-chapter description if this chapter is earlier.
                other_id = conn.execute(
                    _UPSERT_CHARACTER,
                    {"nid": novel_id, "name": other_name, "ch": chapter_num, "desc": None, "ts": now},
                ).scalar_one()
                conn.execute(
                    _UPSERT_RELATIONSHIP,
                    {
                        "nid": novel_id,
                        "a": char_id,
                        "b": other_id,
                        "rel": (rel.get("type") or "").strip() or "unknown",
                        "ch": chapter_num,
                        "ts": now,
                    },
                )

            stored += 1

        conn.commit()
    return stored


@celery_app.task(bind=True, name="ingest_chapter")
def ingest_chapter(self, chapter_data: dict):
    """Celery task wrapper for single chapter ingestion."""
    extract = chapter_data.pop("extract_entities", False)
    return _process_chapter(chapter_data, extract_entities=extract)


@celery_app.task(bind=True, name="ingest_from_source")
def ingest_from_source(self, source_data: dict):
    """
    Fetch chapters from a source adapter and ingest them all.

    source_data: {"novel_id": int, "source_type": str, "source_path": str,
                  "max_chapters": int|None, "extract_entities": bool}
    """
    from adapters import get_adapter

    novel_id = source_data["novel_id"]
    source_type = source_data["source_type"]
    source_path = source_data["source_path"]
    max_chapters = source_data.get("max_chapters")
    extract_entities = source_data.get("extract_entities", False)

    self.update_state(state="FETCHING", meta={"novel_id": novel_id, "source_type": source_type})

    adapter = get_adapter(source_type, source_path)
    chapters = adapter.fetch_all(max_chapters=max_chapters)

    total = len(chapters)
    self.update_state(state="INGESTING", meta={"novel_id": novel_id, "total_chapters": total, "processed": 0})

    results = []
    for i, ch in enumerate(chapters, 1):
        ch["novel_id"] = novel_id
        result = _process_chapter(ch, extract_entities=extract_entities)
        results.append(result)
        if i % 25 == 0 or i == total:
            self.update_state(
                state="INGESTING",
                meta={"novel_id": novel_id, "total_chapters": total, "processed": i},
            )

    return {
        "status": "complete",
        "novel_id": novel_id,
        "chapters_processed": len(results),
        "chapters_succeeded": sum(1 for r in results if r["status"] == "complete"),
    }
