import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import chromadb
import httpx
import redis
from adapters import get_adapter
from celery import Celery
from chunking import chunk_text, clean_html, split_windows
from config import (
    CHROMADB_URL,
    CHUNK_SIZE,
    DATABASE_URL,
    GENERATION_SERVICE_URL,
    REDIS_URL,
    SUMMARY_BATCH,
    SUMMARY_LOOKAHEAD,
    SUMMARY_PARALLEL,
    WINDOW_SIZE,
)
from sqlalchemy import create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError
from vector_store import UnpinnedCollectionError, embedding_function, open_collection
from words import CAPITALIZED, stem, word_uses

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


def _replace_chapter_chunks(collection_name, chapter_num, ids, documents, metadatas,
                            embeddings=None):
    """Make the stored records for one chapter exactly these.

    `add()` silently skips ids that already exist, so re-ingesting a chapter used to keep its
    old text, plus any surplus old chunks when the new chunking yielded fewer, while the task
    reported success. Upserting replaces existing ids; pruning removes whatever else the
    chapter had (a longer earlier version, or whole chunks stored before windows). In that
    order a crash midway leaves extra records at worst, never a chapter with none.
    """
    batch_size = 100

    def write(collection):
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            collection.upsert(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
                # computed for many chapters at once; without them, the collection embeds
                **({"embeddings": embeddings[start:end]} if embeddings is not None else {}),
            )
        stored = collection.get(where={"chapter_number": chapter_num}, include=[])["ids"]
        stale = sorted(set(stored) - set(ids))
        if stale:
            collection.delete(ids=stale)

    try:
        write(_get_collection(collection_name))
    except UnpinnedCollectionError:
        raise
    except Exception:
        # the cached handle may be stale (collection deleted and recreated); refresh once
        _collections.pop(collection_name, None)
        write(_get_collection(collection_name))


def _chapter_chunks(content: str) -> tuple[str, list[str]]:
    text = clean_html(content)
    return text, chunk_text(text, CHUNK_SIZE)


def _windows(chunks: list[str]) -> list[tuple[int, int, str]]:
    """(chunk index, window index, text) for each window of each chunk. Windows are what is
    embedded and searched: the model reads only the first 256 tokens of a text, about half of
    a chunk. A search returns the chunks its best windows belong to (retrieval's search.py)."""
    return [(i, j, window) for i, chunk in enumerate(chunks)
            for j, window in enumerate(split_windows(chunk, WINDOW_SIZE))]


# Windows are embedded EMBED_GROUP chapters at a time: one chapter at a time, 300 chapters took
# 52.6 s to ingest; 32 at a time, 35.4 s (whole chunks, before windows). The vectors are
# identical either way: the model pads every text to the same length.
EMBED_GROUP = 32
_embedder = None


def _embed(texts: list[str]) -> list:
    """Vectors from the pinned model, the same one the collections use."""
    global _embedder
    if not texts:
        return []
    if _embedder is None:
        _embedder = embedding_function()
    return list(_embedder(texts))


def _process_chapter(chapter_data: dict, extract_entities: bool = False,
                     chunked: tuple[str, list[str]] | None = None,
                     embeddings: list | None = None) -> dict:
    """
    Core logic: clean HTML, chunk, embed, store.

    chapter_data: {"novel_id": int, "number": int, "title": str, "content": str, "volume": int}

    extract_entities: when True, also run LLM character extraction via the
    generation service (~15-20s/chapter). Off by default; recall queries
    answer character questions from RAG without pre-extracted entities.

    chunked, embeddings: the cleaned text and chunks, and the vectors of their windows, when
    the caller already has them (see _ingest_chapters); otherwise computed here.
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

    # 1-2. Clean HTML and chunk
    text, chunks = chunked or _chapter_chunks(chapter_data["content"])
    word_count = len(text.split())
    if not chunks:
        # a re-ingested chapter that now cleans to nothing must not keep its old chunks
        _replace_chapter_chunks(collection_name, chapter_num, [], [], [])
        return {"status": "skipped", "chapter": chapter_num, "reason": "no content"}

    # 3. Store in ChromaDB, each chunk as the windows the embedding model reads whole
    windows = _windows(chunks)
    ids = [f"ch_{chapter_num:04d}_chunk_{i:03d}_w{j}" for i, j, _ in windows]
    metadatas = [
        {
            "chapter_number": chapter_num,
            "chapter_title": title,
            "chunk_index": i,
            "window_index": j,
            "volume": volume,
        }
        for i, j, _ in windows
    ]

    _replace_chapter_chunks(collection_name, chapter_num, ids, [w for _, _, w in windows],
                            metadatas, embeddings)

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

        _record_words(conn, novel_id, chapter_num, f"{title} {text}")

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
        "windows": len(windows),
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

    results = _ingest_chapters(
        novel_id, chapters, extract_entities, _furthest_reader(novel_id),
        report=lambda meta: self.update_state(state="INGESTING", meta=meta))
    # the rest of the summaries, for wherever readers are now (they may have moved meanwhile)
    summaries = _schedule_quietly(novel_id)

    return {
        "status": "complete",
        "novel_id": novel_id,
        "chapters_processed": len(results),
        "chapters_succeeded": sum(1 for r in results if r["status"] == "complete"),
        "summaries": summaries,
    }


# Where each word of a book first appears, and first appears in lowercase (a word never
# written in lowercase is a name): answers and summaries are checked against it for names the
# reader has not reached. Chapters arrive in any order (re-ingestion, several workers), so the
# earliest chapter wins.
_RECORD_WORDS = sa_text("""
    INSERT INTO book_words (novel_id, word, first_chapter, first_lowercase_chapter)
    SELECT :nid, word, :ch, CASE WHEN lowercase THEN :ch END
    FROM unnest(CAST(:words AS TEXT[]), CAST(:lowercase AS BOOLEAN[])) AS w(word, lowercase)
    ON CONFLICT (novel_id, word) DO UPDATE SET
        first_chapter = LEAST(book_words.first_chapter, EXCLUDED.first_chapter),
        first_lowercase_chapter = LEAST(book_words.first_lowercase_chapter,
                                        EXCLUDED.first_lowercase_chapter)
""")


def _record_words(conn, novel_id: int, chapter_number: int, text: str) -> None:
    uses = word_uses(text)
    conn.execute(_RECORD_WORDS, {"nid": novel_id, "ch": chapter_number, "words": list(uses),
                                 "lowercase": list(uses.values())})


def rebuild_word_index(novel_id: int) -> dict:
    """Record where each word first appears, from the chapters already indexed: for novels
    ingested before the record existed. Ingestion keeps it up to date from then on."""
    with engine.connect() as conn:
        chapters = conn.execute(sa_text(
            "SELECT chapter_number FROM chapters "
            "WHERE novel_id = :nid AND ingestion_status = 'complete'"
        ), {"nid": novel_id}).scalars().all()
    for number in chapters:
        title, text = _chapter_text(novel_id, number)
        with engine.connect() as conn:
            _record_words(conn, novel_id, number, f"{title} {text}")
            conn.commit()
    return {"novel_id": novel_id, "chapters": len(chapters)}


def _ingest_chapters(novel_id: int, chapters: list[dict], extract_entities: bool = False,
                     reader_chapter: int = 0, report=None) -> list[dict]:
    """Index chapters in reading order, EMBED_GROUP chapters' windows per embedding call. Each
    chapter is searchable as soon as it is stored, so a reader can ask about the chapters they
    have read long before the rest of the book is in. Summaries start then too, rather than
    after the whole book."""
    results, summaries_started = [], False
    for start in range(0, len(chapters), EMBED_GROUP):
        group = chapters[start:start + EMBED_GROUP]
        prepared = [_chapter_chunks(ch["content"]) for ch in group]
        windows = [_windows(chunks) for _, chunks in prepared]
        vectors = iter(_embed([w for ws in windows for _, _, w in ws]))
        for chapter, (text, chunks), ws in zip(group, prepared, windows, strict=True):
            chapter["novel_id"] = novel_id
            results.append(_process_chapter(chapter, extract_entities, chunked=(text, chunks),
                                            embeddings=[next(vectors) for _ in ws]))
        searchable_up_to = group[-1]["number"]
        if report:
            report({"novel_id": novel_id, "total_chapters": len(chapters),
                    "processed": len(results), "searchable_up_to": searchable_up_to})
        if not summaries_started and searchable_up_to >= reader_chapter:
            summaries_started = True
            _schedule_quietly(novel_id, reader_chapter)
    return results


def _schedule_quietly(novel_id: int, reader_chapter: int | None = None) -> dict:
    try:
        return schedule_summaries(novel_id, reader_chapter=reader_chapter)
    except Exception as exc:  # summaries are an extra: never fail an ingest over them
        return {"error": str(exc)}


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
    if reader_chapter is None:
        reader_chapter = _furthest_reader(novel_id)
    if up_to is None:
        up_to = reader_chapter + SUMMARY_LOOKAHEAD
    missing = _chapters_without_summaries(novel_id, up_to, _summary_version())
    # newest first back from the reader, so "catch me up on the last chapters" works first;
    # chapters past the reader after that, in order
    missing = (sorted((n for n in missing if n <= reader_chapter), reverse=True)
               + [n for n in missing if n > reader_chapter])
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
    """A chapter's title and text, put back together from its indexed windows (they do not
    overlap). Chunks stored before windows count as one window each."""
    got = _get_collection(f"novel_{novel_id}").get(
        where={"chapter_number": chapter_number}, include=["documents", "metadatas"])
    pieces = sorted(zip(got["metadatas"], got["documents"], strict=True),
                    key=lambda piece: (piece[0].get("chunk_index", 0),
                                       piece[0].get("window_index", 0)))
    title = pieces[0][0].get("chapter_title", "") if pieces else ""
    return title, " ".join(document for _, document in pieces)


# no letters since the last sentence end (or the start): quotes, "**" and list numbers allowed
_OPENS_SENTENCE = re.compile(r"(^|[.!?:;\n])[^A-Za-z]*$")


_HEADING_WORDS = ("Chapter", "Part", "Summary")


def _name_candidates(summary: str) -> list[str]:
    """Capitalized words in a summary that could be names. Words opening a sentence are
    capitalized anyway, and the headings the model writes ("Chapter 12 Summary (Part 1 of
    2):") are not the book's words."""
    return [m.group() for m in CAPITALIZED.finditer(summary)
            if not (_OPENS_SENTENCE.search(summary[:m.start()]) or m.group() in _HEADING_WORDS)]


def _unverified_names(summary: str, first_chapter: dict[str, int], chapter: int) -> list[str]:
    """Names in a chapter's summary that the book has not used by that chapter, or never uses
    (`first_chapter`: where each reduced word first appears). They come from the model's
    memory of the book, which is how a summary could carry a spoiler. Checked against the
    book so far, not the chapter alone, since a reader knows the names of earlier chapters:
    on Shadow Slave that left 6 of 33 flags, with every real one."""
    return sorted({w for w in _name_candidates(summary)
                   if first_chapter.get(stem(w), chapter + 1) > chapter})


def _first_chapters(novel_id: int, words: list[str]) -> dict[str, int]:
    """Where each of these words first appears in the novel, keyed by its reduced form."""
    keys = sorted({stem(w) for w in words})
    if not keys:
        return {}
    with engine.connect() as conn:
        return dict(conn.execute(sa_text(
            "SELECT word, first_chapter FROM book_words "
            "WHERE novel_id = :nid AND word = ANY(:words)"
        ), {"nid": novel_id, "words": keys}).all())


def recheck_summary_flags(novel_id: int) -> dict:
    """Recompute unverified_names for a novel's stored summaries, after the check or the word
    record changes."""
    with engine.connect() as conn:
        rows = conn.execute(sa_text(
            "SELECT chapter_number, summary FROM chapter_summaries WHERE novel_id = :nid"
        ), {"nid": novel_id}).all()
    first = _first_chapters(novel_id, [w for _, summary in rows for w in _name_candidates(summary)])
    with engine.connect() as conn:
        flagged = 0
        for chapter_number, summary in rows:
            names = _unverified_names(summary, first, chapter_number)
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
        _, text = _chapter_text(novel_id, chapter_number)
        if not text:
            return chapter_number, None
        resp = httpx.post(f"{GENERATION_SERVICE_URL}/chapter-summary", timeout=600,
                          json={"chapter_number": chapter_number, "chapter_text": text})
        resp.raise_for_status()
        result = resp.json()
        first = _first_chapters(novel_id, _name_candidates(result["summary"]))
        result["unverified_names"] = _unverified_names(result["summary"], first, chapter_number)
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
