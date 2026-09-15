from celery import Celery
import chromadb
import httpx
from sqlalchemy import create_engine, text as sa_text
from datetime import datetime, timezone

from chunking import clean_html, chunk_text
from config import (
    REDIS_URL, CHROMADB_URL, DATABASE_URL,
    CHUNK_SIZE, GENERATION_SERVICE_URL,
)

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
        _collections[collection_name] = _chroma_client.get_or_create_collection(collection_name)
    return _collections[collection_name]


def _process_chapter(chapter_data: dict, extract_entities: bool = False) -> dict:
    """
    Core logic: clean HTML, chunk, embed, store.

    chapter_data: {"novel_id": int, "number": int, "title": str, "content": str, "volume": int}

    extract_entities: when True, also run LLM character extraction via the
    generation service (~15-20s/chapter). Off by default; recall queries
    answer character questions from RAG without pre-extracted entities.
    """
    novel_id = chapter_data.get("novel_id")
    chapter_num = chapter_data["number"]
    title = chapter_data.get("title", f"Chapter {chapter_num}")
    volume = chapter_data.get("volume", 1)

    collection_name = f"novel_{novel_id}" if novel_id else "shadow_slave"

    # 1. Clean HTML
    text = clean_html(chapter_data["content"])
    word_count = len(text.split())

    # 2. Chunk text
    chunks = chunk_text(text, CHUNK_SIZE)
    if not chunks:
        return {"status": "skipped", "chapter": chapter_num, "reason": "no content"}

    # 3. Store in ChromaDB
    collection = _get_collection(collection_name)

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

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        try:
            collection.add(
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        except Exception:
            # cached handle may be stale (collection deleted/recreated) —
            # refresh once and retry
            _collections.pop(collection_name, None)
            collection = _get_collection(collection_name)
            collection.add(
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

    # 4. Update PostgreSQL
    with engine.connect() as conn:
        conn.execute(
            sa_text("""
                INSERT INTO chapters (novel_id, chapter_number, title, word_count, volume, ingestion_status, ingested_at)
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
                "ts": datetime.now(timezone.utc),
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


def _store_entities(novel_id: int, chapter_num: int, characters: list) -> int:
    """Store extracted character entities in PostgreSQL."""
    if not characters or not novel_id:
        return 0

    stored = 0
    with engine.connect() as conn:
        for char in characters:
            name = char.get("name", "").strip()
            if not name or len(name) < 2:
                continue

            aliases = char.get("aliases", [])
            role = char.get("role", "")

            # Upsert character
            result = conn.execute(
                sa_text("""
                    INSERT INTO characters (novel_id, name, aliases, first_appearance, description, updated_at)
                    VALUES (:nid, :name, :aliases, :ch, :desc, :ts)
                    ON CONFLICT (novel_id, name)
                    DO UPDATE SET
                        aliases = CASE
                            WHEN characters.aliases IS NULL THEN :aliases
                            ELSE characters.aliases || :aliases
                        END,
                        description = CASE
                            WHEN characters.description IS NULL THEN :desc
                            ELSE characters.description
                        END,
                        updated_at = :ts
                    RETURNING id
                """),
                {
                    "nid": novel_id,
                    "name": name,
                    "aliases": aliases,
                    "ch": chapter_num,
                    "desc": role,
                    "ts": datetime.now(timezone.utc),
                },
            )
            row = result.fetchone()
            if not row:
                continue
            char_id = row[0]

            # Record mention
            conn.execute(
                sa_text("""
                    INSERT INTO character_mentions (novel_id, character_id, chapter_number, context)
                    VALUES (:nid, :cid, :ch, :ctx)
                    ON CONFLICT (character_id, chapter_number) DO NOTHING
                """),
                {"nid": novel_id, "cid": char_id, "ch": chapter_num, "ctx": role[:200]},
            )

            # Store relationships
            for rel in char.get("relationships", []):
                other_name = rel.get("character", "").strip()
                rel_type = rel.get("type", "unknown")
                if not other_name:
                    continue

                # Get or create the other character
                other_result = conn.execute(
                    sa_text("""
                        INSERT INTO characters (novel_id, name, first_appearance, updated_at)
                        VALUES (:nid, :name, :ch, :ts)
                        ON CONFLICT (novel_id, name)
                        DO UPDATE SET updated_at = :ts
                        RETURNING id
                    """),
                    {"nid": novel_id, "name": other_name, "ch": chapter_num, "ts": datetime.now(timezone.utc)},
                )
                other_row = other_result.fetchone()
                if not other_row:
                    continue

                conn.execute(
                    sa_text("""
                        INSERT INTO character_relationships (novel_id, character_a_id, character_b_id, relationship_type, first_chapter, updated_at)
                        VALUES (:nid, :a, :b, :rel, :ch, :ts)
                        ON CONFLICT (character_a_id, character_b_id, relationship_type) DO NOTHING
                    """),
                    {
                        "nid": novel_id,
                        "a": char_id,
                        "b": other_row[0],
                        "rel": rel_type,
                        "ch": chapter_num,
                        "ts": datetime.now(timezone.utc),
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
