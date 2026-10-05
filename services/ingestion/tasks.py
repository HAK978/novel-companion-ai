import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import chromadb
import httpx
import redis
from adapters import get_adapter
from celery import Celery
from chunking import chunk_text, clean_html
from config import (
    CHROMADB_URL,
    CHUNK_SIZE,
    DATABASE_URL,
    GENERATION_SERVICE_URL,
    REDIS_URL,
    SUMMARY_BATCH,
    SUMMARY_LOOKAHEAD,
    SUMMARY_PARALLEL,
)
from sqlalchemy import create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError
from vector_store import UnpinnedCollectionError, open_collection

celery_app = Celery("ingestion", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    # summaries go to their own queue and worker, so they never hold up ingestion
    task_routes={"summarize_chapters": {"queue": "summaries"}},
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

    try:
        summaries = schedule_summaries(novel_id)
    except Exception as exc:  # summaries are an extra: never fail an ingest over them
        summaries = {"error": str(exc)}

    return {
        "status": "complete",
        "novel_id": novel_id,
        "chapters_processed": len(results),
        "chapters_succeeded": sum(1 for r in results if r["status"] == "complete"),
        "summaries": summaries,
    }


# --- Chapter summaries ---------------------------------------------------------------
# One summary per chapter, so broad questions need not reread every chapter. They are
# written on the "summaries" queue by their own worker, which waits while the model is not
# being served, and only up to the furthest reader plus SUMMARY_LOOKAHEAD chapters.

_redis = redis.Redis.from_url(REDIS_URL)
_PENDING_TTL = 24 * 3600


def _pending_key(novel_id: int) -> str:
    # chapters already queued, so repeated scheduling (every progress update) adds no duplicates
    return f"summaries:pending:{novel_id}"


def _unqueue(novel_id: int, chapters) -> None:
    if chapters:
        _redis.srem(_pending_key(novel_id), *chapters)


def _summary_version() -> int | None:
    try:
        resp = httpx.get(f"{GENERATION_SERVICE_URL}/chapter-summary", timeout=10)
        return resp.json()["prompt_version"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _furthest_reader(novel_id: int) -> int:
    with engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT COALESCE(MAX(current_chapter), 0) FROM reading_progress WHERE novel_id = :nid"
        ), {"nid": novel_id}).scalar() or 0


def _chapters_without_summaries(novel_id: int, up_to: int, version: int | None) -> list[int]:
    """Ingested chapters up to `up_to` with no summary from the current prompt (or with no
    summary at all, when the current prompt version is unknown)."""
    with engine.connect() as conn:
        return list(conn.execute(sa_text("""
            SELECT c.chapter_number FROM chapters c
            WHERE c.novel_id = :nid AND c.chapter_number <= :up
              AND c.ingestion_status = 'complete'
              AND NOT EXISTS (
                  SELECT 1 FROM chapter_summaries s
                  WHERE s.novel_id = c.novel_id AND s.chapter_number = c.chapter_number
                    AND (CAST(:version AS INTEGER) IS NULL OR s.prompt_version = :version))
            ORDER BY c.chapter_number
        """), {"nid": novel_id, "up": up_to, "version": version}).scalars())


def schedule_summaries(novel_id: int, reader_chapter: int | None = None,
                       up_to: int | None = None) -> dict:
    """Queue summaries for the chapters readers can use: up to the furthest reader (or
    `reader_chapter`) plus SUMMARY_LOOKAHEAD, or up to `up_to` when given."""
    if up_to is None:
        if reader_chapter is None:
            reader_chapter = _furthest_reader(novel_id)
        up_to = reader_chapter + SUMMARY_LOOKAHEAD
    missing = _chapters_without_summaries(novel_id, up_to, _summary_version())
    key = _pending_key(novel_id)
    # SADD reports whether each chapter was new, so two schedules racing (readers updating
    # progress at once) cannot both queue the same chapter
    chapters = [n for n in missing if _redis.sadd(key, n)]
    if chapters:
        _redis.expire(key, _PENDING_TTL)
    batches = [chapters[i:i + SUMMARY_BATCH] for i in range(0, len(chapters), SUMMARY_BATCH)]
    for batch in batches:
        summarize_chapters.delay(novel_id, batch)
    return {"novel_id": novel_id, "up_to": up_to, "queued": len(chapters),
            "already_queued": len(missing) - len(chapters), "batches": len(batches)}


def summary_status(novel_id: int) -> dict:
    with engine.connect() as conn:
        written, flagged, highest = conn.execute(sa_text("""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE cardinality(unverified_names) > 0),
                   MAX(chapter_number)
            FROM chapter_summaries WHERE novel_id = :nid
        """), {"nid": novel_id}).one()
    return {"novel_id": novel_id, "summarized": written, "flagged": flagged,
            "highest_chapter": highest, "queued": _redis.scard(_pending_key(novel_id))}


def _chapter_text(novel_id: int, chapter_number: int) -> tuple[str, str]:
    """A chapter's title and text, put back together from its indexed chunks (they do not
    overlap)."""
    got = _get_collection(f"novel_{novel_id}").get(
        where={"chapter_number": chapter_number}, include=["documents", "metadatas"])
    pieces = sorted(zip(got["metadatas"], got["documents"], strict=True),
                    key=lambda piece: piece[0].get("chunk_index", 0))
    title = pieces[0][0].get("chapter_title", "") if pieces else ""
    return title, " ".join(document for _, document in pieces)


# no letters since the last sentence end (or the start): quotes, "**" and list numbers allowed
_OPENS_SENTENCE = re.compile(r"(^|[.!?:;\n])[^A-Za-z]*$")


def _stem(word: str) -> str:
    word = word.lower()
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    for suffix in ("'s", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def _unverified_names(summary: str, title: str, text: str) -> list[str]:
    """Capitalized words in a summary that its chapter (title included) never uses, in any
    form. A summary should name only what its chapter does, so these may come from the
    model's memory of the book: that is how a summary could carry a spoiler. Words opening a
    sentence are skipped (they are capitalized anyway), and plurals and possessives match
    their stem."""
    vocabulary = {_stem(w) for w in re.findall(r"[A-Za-z']+", f"{title} {text}")}
    flagged = set()
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", summary):
        if _OPENS_SENTENCE.search(summary[:match.start()]) or match.group() == "Chapter":
            continue
        if _stem(match.group()) not in vocabulary:
            flagged.add(match.group())
    return sorted(flagged)


def recheck_summary_flags(novel_id: int) -> dict:
    """Recompute unverified_names for a novel's stored summaries, after the check changes."""
    with engine.connect() as conn:
        rows = conn.execute(sa_text(
            "SELECT chapter_number, summary FROM chapter_summaries WHERE novel_id = :nid"
        ), {"nid": novel_id}).all()
        flagged = 0
        for chapter_number, summary in rows:
            names = _unverified_names(summary, *_chapter_text(novel_id, chapter_number))
            flagged += bool(names)
            conn.execute(sa_text(
                "UPDATE chapter_summaries SET unverified_names = :names "
                "WHERE novel_id = :nid AND chapter_number = :ch"
            ), {"names": names, "nid": novel_id, "ch": chapter_number})
        conn.commit()
    return {"novel_id": novel_id, "checked": len(rows), "flagged": flagged}


_UPSERT_SUMMARY = sa_text("""
    INSERT INTO chapter_summaries
        (novel_id, chapter_number, summary, model, prompt_version, unverified_names, created_at)
    VALUES (:nid, :ch, :summary, :model, :version, :names, :ts)
    ON CONFLICT (novel_id, chapter_number) DO UPDATE SET
        summary = EXCLUDED.summary, model = EXCLUDED.model,
        prompt_version = EXCLUDED.prompt_version,
        unverified_names = EXCLUDED.unverified_names, created_at = EXCLUDED.created_at
""")


# acks_late: a task lost with its worker (a restart mid-batch) is delivered again, which is
# safe because finished chapters are skipped
@celery_app.task(bind=True, name="summarize_chapters", max_retries=288, acks_late=True,
                 reject_on_worker_lost=True)
def summarize_chapters(self, novel_id: int, chapter_numbers: list[int]):
    """Summarize these chapters. While the model is not being served, try again every 5
    minutes for up to a day. Chapters already summarized with the current prompt are skipped,
    so a task that runs twice costs nothing extra."""
    def retry_later():
        if self.request.retries >= self.max_retries:
            _unqueue(novel_id, chapter_numbers)  # give up for now; a later schedule requeues
        return self.retry(countdown=300)

    with engine.connect() as conn:
        novel = conn.execute(sa_text("SELECT 1 FROM novels WHERE id = :nid"),
                             {"nid": novel_id}).first()
    if novel is None:  # deleted since it was queued; opening its collection would recreate it
        _unqueue(novel_id, chapter_numbers)
        return {"status": "skipped", "reason": "novel deleted"}

    try:
        serving = httpx.get(f"{GENERATION_SERVICE_URL}/health", timeout=10).json()
        version = _summary_version()
    except (httpx.HTTPError, ValueError):
        serving, version = {}, None
    if serving.get("status") != "ok" or version is None:
        raise retry_later()

    with engine.connect() as conn:
        done = set(conn.execute(sa_text("""
            SELECT chapter_number FROM chapter_summaries
            WHERE novel_id = :nid AND prompt_version = :v AND chapter_number = ANY(:chapters)
        """), {"nid": novel_id, "v": version, "chapters": chapter_numbers}).scalars())
    _unqueue(novel_id, done)

    def summarize(chapter_number):
        title, text = _chapter_text(novel_id, chapter_number)
        if not text:
            return chapter_number, None
        resp = httpx.post(f"{GENERATION_SERVICE_URL}/chapter-summary", timeout=600,
                          json={"chapter_number": chapter_number, "chapter_text": text})
        resp.raise_for_status()
        result = resp.json()
        result["unverified_names"] = _unverified_names(result["summary"], title, text)
        return chapter_number, result

    written = 0
    try:
        with ThreadPoolExecutor(max_workers=SUMMARY_PARALLEL) as pool:
            for chapter_number, result in pool.map(
                    summarize, [n for n in chapter_numbers if n not in done]):
                if result:
                    with engine.connect() as conn:
                        conn.execute(_UPSERT_SUMMARY, {
                            "nid": novel_id, "ch": chapter_number, "summary": result["summary"],
                            "model": result["model"], "version": result["prompt_version"],
                            "names": result["unverified_names"], "ts": datetime.now(UTC),
                        })
                        conn.commit()
                    written += 1
                _unqueue(novel_id, [chapter_number])
    except httpx.HTTPError:
        raise retry_later() from None  # the model went away: what was written stays
    except IntegrityError:
        _unqueue(novel_id, chapter_numbers)
        return {"status": "skipped", "reason": "novel deleted", "written": written}

    return {"status": "complete", "novel_id": novel_id, "written": written,
            "already_done": len(done)}
