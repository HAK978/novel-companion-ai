import asyncio
import re
from datetime import UTC, datetime

import httpx
from config import (
    CORS_ORIGINS,
    DATABASE_URL,
    GENERATION_SERVICE_URL,
    INGESTION_SERVICE_URL,
    RETRIEVAL_SERVICE_URL,
)
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

app = FastAPI(title="Novel Companion AI - Gateway")

# Only the configured frontend origins. This allowed any site with credentials, so once
# deployed, any page a user visited could call the API from their browser. No cookies or
# auth are used, so credentials stay off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DATABASE_URL)
HEALTH_TIMEOUT = 3  # seconds per downstream check; container probes allow 4

# Database calls are synchronous. Handlers that only touch the database are plain `def`
# (FastAPI runs those in a threadpool); handlers that also await other services push their
# database work through run_in_threadpool, so neither blocks the event loop.


# --- Request / Response models ---

class NovelCreate(BaseModel):
    title: str
    author: str | None = None
    source_type: str = "local_json"
    source_url: str | None = None


class QueryRequest(BaseModel):
    query: str
    current_chapter: int
    novel_id: int
    n_results: int = 5
    conversation_context: str = ""


class ChunkResult(BaseModel):
    text: str
    chapter_number: int
    chapter_title: str = ""
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str | None
    model_used: str = "none"
    sources: list[ChunkResult]
    # "chapter summaries" when a recap was answered from the stored summaries of its
    # chapters, "passages" when from search results
    source: str = "passages"
    # [first, last] chapter the question was limited to, when it named some ("the last 3
    # chapters", "so far", "chapter 12"); None when it could draw on every chapter read
    scope: list[int] | None = None
    # "withheld" when the answer, or a chapter summary it drew on, named something the reader
    # has not reached and it was taken out; never what, since that would be the spoiler
    spoiler_check: str = "clean"


class IngestRequest(BaseModel):
    novel_id: int
    number: int
    title: str | None = None
    content: str
    volume: int = 1
    extract_entities: bool = False


class IngestFromSourceRequest(BaseModel):
    novel_id: int
    source_type: str
    source_path: str
    max_chapters: int | None = None
    extract_entities: bool = False


class ProgressRequest(BaseModel):
    novel_id: int
    current_chapter: int
    user_id: str = "default"


# --- Novel Management ---

@app.post("/novels")
async def create_novel(request: NovelCreate):
    def reserve_id():
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT nextval(pg_get_serial_sequence('novels', 'id'))")
            ).scalar()

    novel_id = await run_in_threadpool(reserve_id)

    # Collections are named by novel id. The id is new to Postgres, so a collection that
    # already carries the name is left over from a database reset that kept the vector store,
    # and ingesting into it would mix another novel's chunks into this one. Clear it before
    # the novel exists, so a failure here leaves nothing half-created.
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                f"{RETRIEVAL_SERVICE_URL}/delete-collection", json={"novel_id": novel_id}
            )
        except httpx.HTTPError:
            resp = None
        if (resp is None or resp.status_code != 200
                or resp.json().get("status") not in ("deleted", "absent")):
            # a deletion that failed is not an empty collection: it used to be accepted
            raise HTTPException(
                status_code=503,
                detail="Could not confirm the new novel's vector collection is empty, so the "
                       "novel was not created.",
            )

    def insert():
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    INSERT INTO novels (id, title, author, source_type, source_url)
                    VALUES (:id, :title, :author, :source_type, :source_url)
                    RETURNING id, title, created_at
                """),
                {
                    "id": novel_id,
                    "title": request.title,
                    "author": request.author,
                    "source_type": request.source_type,
                    "source_url": request.source_url,
                },
            ).fetchone()
            conn.commit()
            return row

    row = await run_in_threadpool(insert)
    return {"id": row[0], "title": row[1], "created_at": str(row[2])}


@app.get("/novels")
def list_novels():
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, title, author, source_type, total_chapters, created_at FROM novels ORDER BY id")
        ).fetchall()
    return [
        {
            "id": r[0], "title": r[1], "author": r[2],
            "source_type": r[3], "total_chapters": r[4],
            "created_at": str(r[5]),
        }
        for r in rows
    ]


@app.get("/novels/{novel_id}")
def get_novel(novel_id: int):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, title, author, source_type, source_url, total_chapters, created_at "
                "FROM novels WHERE id = :id"
            ),
            {"id": novel_id},
        ).fetchone()
    if not row:
        return {"error": "Novel not found"}
    return {
        "id": row[0], "title": row[1], "author": row[2],
        "source_type": row[3], "source_url": row[4],
        "total_chapters": row[5], "created_at": str(row[6]),
    }


@app.delete("/novels/{novel_id}")
async def delete_novel(novel_id: int):
    await run_in_threadpool(_delete_novel_rows, novel_id)

    # Delete ChromaDB collection
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(
                f"{RETRIEVAL_SERVICE_URL}/delete-collection",
                json={"novel_id": novel_id},
            )
        except Exception:
            pass

    return {"status": "deleted", "novel_id": novel_id}


def _delete_novel_rows(novel_id: int) -> None:
    with engine.connect() as conn:
        # Mentions/relationships may predate novel_id stamping; also match via the character
        conn.execute(text("""
            DELETE FROM character_mentions WHERE novel_id = :id
                OR character_id IN (SELECT id FROM characters WHERE novel_id = :id)
        """), {"id": novel_id})
        conn.execute(text("""
            DELETE FROM character_relationships WHERE novel_id = :id
                OR character_a_id IN (SELECT id FROM characters WHERE novel_id = :id)
                OR character_b_id IN (SELECT id FROM characters WHERE novel_id = :id)
        """), {"id": novel_id})
        for table in ["search_history", "reading_progress", "range_summaries",
                      "chapter_summaries", "book_words", "characters", "chapters"]:
            conn.execute(text(f"DELETE FROM {table} WHERE novel_id = :id"), {"id": novel_id})
        conn.execute(text("DELETE FROM novels WHERE id = :id"), {"id": novel_id})
        conn.commit()


def _relay(resp) -> JSONResponse:
    """A downstream reply passed on with its status code: errors used to arrive as 200s, so
    a failed ingest or lookup looked like a successful one."""
    try:
        body = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Unreadable reply from a service") from None
    return JSONResponse(body, status_code=resp.status_code)


# --- Ingestion ---

@app.post("/ingest")
async def ingest(request: IngestRequest):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{INGESTION_SERVICE_URL}/ingest",
            json=request.model_dump(),
        )
    return _relay(resp)


@app.post("/ingest/from-source")
async def ingest_from_source(request: IngestFromSourceRequest):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{INGESTION_SERVICE_URL}/ingest/from-source",
            json=request.model_dump(),
        )
    return _relay(resp)


@app.get("/ingest/status/{task_id}")
async def ingest_status(task_id: str):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{INGESTION_SERVICE_URL}/ingest/status/{task_id}")
    return _relay(resp)


# --- Characters ---

@app.get("/characters/list")
async def list_characters(novel_id: int, current_chapter: int):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{RETRIEVAL_SERVICE_URL}/characters",
            params={"novel_id": novel_id, "current_chapter": current_chapter},
        )
    return _relay(resp)


@app.get("/characters/{name}")
async def character_recall(name: str, novel_id: int, current_chapter: int):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{RETRIEVAL_SERVICE_URL}/characters/{name}",
            params={"novel_id": novel_id, "current_chapter": current_chapter},
        )
    return _relay(resp)


# --- Query ---

# What a reader is told when every sentence of an answer named someone they have not reached
NOTHING_LEFT = "I couldn't find that in the chapters you've read."


async def _generate_checked(client, payload: dict, novel_id: int, said: str) -> tuple[dict, str]:
    """Generate, then take out any name the reader has not reached.

    Passages only come from chapters the reader has read, but the model may know the book: at
    chapter 14 of the Hound it called Stapleton's wife "Beryl Garcia", a name the book first
    uses in chapter 15, in 3 of 3 runs. Retrieval finds such names in the book's word record
    and takes them out. Asking the model to rewrite without them was tried first: it still
    used the name in 5 of 12 tries, and complying rewrites got worse (naming the wrong woman),
    while the first answer minus the name ("his wife, Beryl") was right. With this, that
    question got the name in 7 of 24 answers and every one was withheld. `said` is what the
    reader wrote, whose names are theirs. Returns the generation's response and whether a
    name was withheld (see QueryResponse).
    """
    resp = await client.post(f"{GENERATION_SERVICE_URL}/generate", json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Generation service failed")
    data = resp.json()
    check = await client.post(f"{RETRIEVAL_SERVICE_URL}/withhold-unread-names", json={
        "novel_id": novel_id, "current_chapter": payload["current_chapter"],
        "text": data.get("answer") or "", "ignore": said})
    if check.status_code != 200:
        raise HTTPException(status_code=502, detail="Spoiler check failed")
    checked = check.json()
    if not checked.get("withheld"):
        return data, "clean"
    data["answer"] = checked["text"] or NOTHING_LEFT
    return data, "withheld"


# --- What a question is about ---------------------------------------------------------
#
# Read deterministically, as two separate things: the chapters a question limits itself to,
# and whether it asks for a recap of events (answered from chapter summaries) or something
# specific (answered from the book's own passages within those chapters). One keyword used to
# decide both: "so far" meant the last 10 chapters, "the last couple of chapters" meant 10,
# and "Why did Sunny change his mind recently?" was answered from summaries alone, which get
# who-did-what wrong about 1 time in 5.

RECENT_CHAPTERS = 10  # "recently", or a recap that names no chapters
_COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "couple of": 2, "couple": 2, "few": 3,
           "several": 5}
_COUNT_WORDS = "|".join(w.replace(" ", r"\s+") for w in sorted(_COUNTS, key=len, reverse=True))
_EXPLICIT = re.compile(r"\bchapters?\s+(\d+)(?:\s*(?:-|\u2013|to|through|and)\s*(\d+))?", re.I)
_LAST = re.compile(rf"\b(?:last|past|previous)\s+(?:(\d+|{_COUNT_WORDS})\s+)?chapters?\b", re.I)
_RECENTLY = re.compile(r"\b(?:recently|lately)\b", re.I)
_SO_FAR = re.compile(r"\b(?:so far|until now|up to now|up until now)\b", re.I)
_RECAP_REQUEST = re.compile(
    r"\b(?:recap|catch me up|summari[sz]e|summary|what did i miss|bring me up to speed)\b", re.I)
_WHAT_HAPPENED = re.compile(
    r"^\W*(?:so\W+)?what(?:['\u2019]s|\s+has|\s+have|\s+had)?\s+(?:just\s+)?happen(?:ed|s)\b(.*)$",
    re.I | re.S)
# words that only place a question in time: what remains after them is its subject
_PLACING = re.compile(rf"\b(?:in|over|during|the|so|far|until|up|now|recently|lately|chapters?|"
                      rf"last|past|previous|and|through|to|\d+|{_COUNT_WORDS})\b|[^\w\s]", re.I)


def _question_scope(question: str, chapter: int) -> tuple[int, int] | None:
    """The chapters a question is about, when it says: "chapter 12", "chapters 3 to 5", "the
    last 3 chapters", "recently" (the last 10), "so far" (every chapter read). The end never
    passes the reader's chapter; a start past it comes back as asked, for the caller to refuse."""
    if match := _EXPLICIT.search(question):
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        return min(first, last), min(max(first, last), chapter)
    if match := _LAST.search(question):
        word = " ".join((match.group(1) or "one").lower().split())
        count = max(1, int(word) if word.isdigit() else _COUNTS[word])
        return max(1, chapter - count + 1), chapter
    if _RECENTLY.search(question):
        return max(1, chapter - RECENT_CHAPTERS + 1), chapter
    if _SO_FAR.search(question):
        return 1, chapter
    return None


def _is_recap(question: str) -> bool:
    """Whether a question asks for a recap of events rather than something specific: "Catch me
    up", "What has happened so far?", but not "What happened to Sunny's sword?"."""
    if _RECAP_REQUEST.search(question):
        return True
    match = _WHAT_HAPPENED.match(question)
    return bool(match) and not _PLACING.sub(" ", match.group(1)).strip()


def _recap_note(scope: tuple[int, int], covered: tuple[int, int] | None) -> str:
    """What a recap answer must say about its coverage, or "" when it covers what was asked:
    the reader should not take a partial recap for a whole one."""
    if covered is None:
        return (f"\n\n(Summaries of chapters {scope[0]}\u2013{scope[1]} are not ready yet, so this "
                f"is from a search of those chapters, not a full recap.)")
    if covered[0] > scope[0]:
        return (f"\n\n(This recap covers chapters {covered[0]}\u2013{covered[1]}: a recap reads at "
                f"most {RECAP_FROM_SUMMARIES_MAX} chapters at a time.)")
    return ""


async def _withheld_summaries(client, novel_id: int, chapter: int, stored: list, said: str
                              ) -> tuple[list[tuple[int, str, str]], int]:
    """Chapter summaries with what the reader has not reached taken out, before they reach the
    model or the response. A summary can name something the book only introduces later (its
    model knows the book): chapter 33's said "the Underworld", first in the book at chapter
    250. Returns the summaries left, and how many words were withheld."""
    async def check(summary: str) -> dict:
        resp = await client.post(f"{RETRIEVAL_SERVICE_URL}/withhold-unread-names", json={
            "novel_id": novel_id, "current_chapter": chapter, "text": summary,
            "ignore": said, "strict": True})
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Spoiler check failed")
        return resp.json()

    checked = await asyncio.gather(*(check(summary) for _, _, summary in stored))
    kept = [(n, title, c["text"]) for (n, title, _), c in zip(stored, checked, strict=True)
            if c["text"].strip()]
    return kept, sum(c.get("withheld", 0) for c in checked)


def _check(resp, service: str) -> None:
    """A failed downstream call is an error, not an empty result: a failed search used to come
    back as "nothing found"."""
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"{service} service failed")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    chapter = request.current_chapter
    said = f"{request.query} {request.conversation_context}"
    scope = _question_scope(request.query, chapter)
    if scope and scope[0] > chapter:
        # the same reply for every chapter past the reader's, before any search
        return QueryResponse(answer=f"You haven't reached chapter {scope[0]} yet.", sources=[],
                             scope=list(scope))
    recap = _is_recap(request.query)
    if recap and scope is None:
        scope = (max(1, chapter - RECENT_CHAPTERS + 1), chapter)
    covered, stored = None, []
    if recap:
        # search has no sense of time (asked at chapter 1291 it offered chapters 527 and 398 as
        # "the last few"), so a recap reads the summaries of its chapters in order, the last
        # RECAP_FROM_SUMMARIES_MAX of them when it asks for more, and says so
        covered = (max(scope[0], scope[1] - RECAP_FROM_SUMMARIES_MAX + 1), scope[1])
        stored = await run_in_threadpool(_chapter_summaries, request.novel_id, *covered)
        covered = covered if stored else None
    async with httpx.AsyncClient(timeout=120) as client:
        summaries_withheld = 0
        if stored:
            stored, summaries_withheld = await _withheld_summaries(
                client, request.novel_id, chapter, stored, said=said)
            results = [{"text": summary, "chapter_number": n, "chapter_title": title,
                        "relevance_score": 1.0} for n, title, summary in stored]
        else:
            # within the question's chapters when it names some: a recap whose summaries were
            # missing used to search the whole book, and present older events as recent
            retrieval_resp = await client.post(f"{RETRIEVAL_SERVICE_URL}/search", json={
                "query": request.query,
                "current_chapter": scope[1] if scope else chapter,
                "n_results": request.n_results,
                "collection_name": f"novel_{request.novel_id}",
                **({"min_chapter": scope[0]} if scope else {}),
            })
            _check(retrieval_resp, "Retrieval")
            results = retrieval_resp.json().get("results", [])

        if not results:
            return QueryResponse(answer=None, sources=[], scope=list(scope) if scope else None)

        # Label each passage with its chapter so the model can see (and say)
        # when retrieved passages come from distant parts of the story
        context_chunks = [
            f"[Chapter {r['chapter_number']}: {r.get('chapter_title', '')}]\n{r['text']}"
            for r in results
        ]
        gen_data, check = await _generate_checked(client, {
            "query": request.query,
            "context_chunks": context_chunks,
            "conversation_context": request.conversation_context,
            "current_chapter": chapter,
        }, request.novel_id, said=said)

    def record_search():
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO search_history (novel_id, query, results_count, timestamp)
                    VALUES (:nid, :q, :c, :ts)
                """),
                {"nid": request.novel_id, "q": request.query, "c": len(results),
                 "ts": datetime.now(UTC)},
            )
            conn.commit()

    await run_in_threadpool(record_search)

    answer = gen_data.get("answer")
    if recap and answer:
        answer += _recap_note(scope, covered)
    return QueryResponse(
        answer=answer,
        model_used=gen_data.get("model_used", "none"),
        sources=results,
        source="chapter summaries" if stored else "passages",
        scope=list(scope) if scope else None,
        spoiler_check="withheld" if summaries_withheld else check,
    )


# --- Summarization ---

class SummarizeRequest(BaseModel):
    novel_id: int
    start_chapter: int
    end_chapter: int
    current_chapter: int | None = None
    # False writes a fresh recap and leaves the cache alone: evaluations must grade what the
    # current code produces, not a recap cached by an earlier version
    use_cache: bool = True


class CatchMeUpRequest(BaseModel):
    novel_id: int
    user_id: str = "default"
    use_cache: bool = True


# A range recap reads every chapter's stored summary when all of them exist: ~120 words each,
# so 60 chapters stay well inside the model's context. Wider ranges search instead, for now.
RECAP_FROM_SUMMARIES_MAX = 60


def _chapter_summaries(novel_id: int, start: int, end: int) -> list[tuple[int, str, str]]:
    """(chapter, title, summary) for every ingested chapter in the range, or [] if any lacks
    a summary or the range is too wide."""
    if end - start + 1 > RECAP_FROM_SUMMARIES_MAX:
        return []
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.chapter_number, c.title, s.summary
            FROM chapters c LEFT JOIN chapter_summaries s
              ON s.novel_id = c.novel_id AND s.chapter_number = c.chapter_number
            WHERE c.novel_id = :nid AND c.chapter_number BETWEEN :s AND :e
              AND c.ingestion_status = 'complete'
            ORDER BY c.chapter_number
        """), {"nid": novel_id, "s": start, "e": end}).fetchall()
    if not rows or any(r[2] is None for r in rows):
        return []
    return [(r[0], r[1] or "", r[2]) for r in rows]


def _recap_length(chapters: int) -> tuple[int, int]:
    """Words to ask a recap for, and its token cap. With one answer-sized cap (800 tokens) and
    no length asked, an 11-chapter recap ran out mid-sentence in 3 of 3 runs, so the reader's
    latest chapters, written last, were squeezed or lost. The model writes about twice the
    words asked (an 11-chapter recap asked for 385 ran to ~720), so the cap allows for that."""
    words = min(max(35 * chapters, 150), 900)
    return words, min(4 * words + 300, 3000)


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    reader_at = request.current_chapter or request.end_chapter
    # The ceiling is the tighter of "what was asked for" and "what has been
    # read". Using reading position alone let a summary of chapters 100-120
    # pull in everything up to the reader's position.
    end_ch = min(request.end_chapter, reader_at)

    if request.start_chapter > end_ch:
        return {
            "error": (
                f"Nothing to summarize: chapter {request.start_chapter} is past "
                f"reading progress (chapter {reader_at})."
            )
        }

    # Cache is keyed by novel as well as range; a range alone collides across novels.
    def cached_summary():
        with engine.connect() as conn:
            return conn.execute(
                text("""
                    SELECT summary FROM range_summaries
                    WHERE novel_id = :nid AND start_chapter = :s AND end_chapter = :e
                """),
                {"nid": request.novel_id, "s": request.start_chapter, "e": end_ch},
            ).fetchone()

    cached = await run_in_threadpool(cached_summary) if request.use_cache else None

    if cached:
        return {
            "summary": cached[0],
            "start_chapter": request.start_chapter,
            "end_chapter": end_ch,
            "cached": True,
        }

    # every chapter of the range, in order (ten search hits covered a few moments of it): the
    # last RECAP_FROM_SUMMARIES_MAX of them when the range is wider, and the recap says so
    covered = (max(request.start_chapter, end_ch - RECAP_FROM_SUMMARIES_MAX + 1), end_ch)
    stored = await run_in_threadpool(_chapter_summaries, request.novel_id, *covered)
    collection_name = f"novel_{request.novel_id}"
    async with httpx.AsyncClient(timeout=120) as client:
        if stored:
            stored, _ = await _withheld_summaries(client, request.novel_id, end_ch, stored,
                                                  said="")
        if stored:
            source = "chapter summaries"
            context = [f"[Chapter {n}: {title}]\n{text}" for n, title, text in stored]
        else:
            covered = None
            source = "passages"
            retrieval_resp = await client.post(
                f"{RETRIEVAL_SERVICE_URL}/search",
                json={
                    "query": f"summary of events in chapters {request.start_chapter} to {end_ch}",
                    "current_chapter": end_ch,
                    "n_results": 10,
                    "collection_name": collection_name,
                    # floor the search to the requested range; without it,
                    # similar chunks from much earlier arcs pollute the summary
                    "min_chapter": request.start_chapter,
                },
            )
            _check(retrieval_resp, "Retrieval")
            results = retrieval_resp.json().get("results", [])
            if not results:
                return {"error": "No content found for this chapter range"}
            context = [f"[Chapter {r['chapter_number']}: {r.get('chapter_title', '')}]\n{r['text']}"
                       for r in results]

        # A failed generation is a failure, not a summary: it raises, and is never cached.
        first = covered[0] if covered else request.start_chapter
        words, max_tokens = _recap_length(end_ch - first + 1)
        gen_data, _ = await _generate_checked(client, {
            "query": (
                f"Recap chapters {first} to {end_ch} in about {words} words: "
                f"the key events, in chapter order, a sentence or two for each chapter, giving "
                f"the last chapters as much room as the first."
            ),
            "context_chunks": context,
            "current_chapter": end_ch,
            "max_tokens": max_tokens,
        }, request.novel_id, said="")

    summary = gen_data.get("answer") or ""
    if not summary:
        raise HTTPException(status_code=502, detail="Generation returned no summary")
    summary += _recap_note((request.start_chapter, end_ch), covered)

    def cache_summary():
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO range_summaries (novel_id, start_chapter, end_chapter, summary)
                    VALUES (:nid, :s, :e, :sum)
                    ON CONFLICT (novel_id, start_chapter, end_chapter) DO UPDATE SET summary = :sum
                """),
                {"nid": request.novel_id, "s": request.start_chapter, "e": end_ch, "sum": summary},
            )
            conn.commit()

    if request.use_cache:
        await run_in_threadpool(cache_summary)

    return {
        "summary": summary,
        "start_chapter": request.start_chapter,
        "end_chapter": end_ch,
        "cached": False,
        "source": source,
        "covered": list(covered) if covered else None,
    }


@app.post("/catch-me-up")
async def catch_me_up(request: CatchMeUpRequest):
    def progress():
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT current_chapter FROM reading_progress "
                     "WHERE novel_id = :nid AND user_id = :uid"),
                {"nid": request.novel_id, "uid": request.user_id},
            ).fetchone()

    row = await run_in_threadpool(progress)

    if not row or not row[0]:
        return {"error": "No reading progress found. Set your progress first with POST /progress."}

    current_chapter = row[0]

    # Determine range to summarize (last 10 chapters or from start)
    start = max(1, current_chapter - 10)

    # Get summary
    summary_resp = await summarize(SummarizeRequest(
        novel_id=request.novel_id,
        start_chapter=start,
        end_chapter=current_chapter,
        current_chapter=current_chapter,
        use_cache=request.use_cache,
    ))
    if "error" in summary_resp:
        # passed on: it used to come back as an empty recap
        return {"user_id": request.user_id, "current_chapter": current_chapter,
                "error": summary_resp["error"]}

    # Get characters introduced in this range
    async with httpx.AsyncClient(timeout=10) as client:
        chars_resp = await client.get(
            f"{RETRIEVAL_SERVICE_URL}/characters",
            params={"novel_id": request.novel_id, "current_chapter": current_chapter},
        )
    _check(chars_resp, "Retrieval")
    all_chars = chars_resp.json()
    recent_chars = [c for c in all_chars if c.get("first_appearance", 0) >= start]

    return {
        "user_id": request.user_id,
        "current_chapter": current_chapter,
        "summary_range": {"start": start, "end": current_chapter},
        "summary": summary_resp.get("summary", ""),
        "cached": summary_resp.get("cached", False),
        "source": summary_resp.get("source"),
        "new_characters": recent_chars,
    }


# --- Reading Progress ---

@app.post("/progress")
def update_progress(request: ProgressRequest):
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO reading_progress (novel_id, user_id, current_chapter, updated_at)
                VALUES (:nid, :uid, :ch, :ts)
                ON CONFLICT (novel_id, user_id)
                DO UPDATE SET current_chapter = :ch, updated_at = :ts
            """),
            {
                "nid": request.novel_id,
                "uid": request.user_id,
                "ch": request.current_chapter,
                "ts": datetime.now(UTC),
            },
        )
        conn.commit()
    # Chapter summaries are written only a little ahead of readers, so keep them ahead of
    # this one. Best effort: progress is saved whether or not ingestion is reachable.
    try:
        httpx.post(f"{INGESTION_SERVICE_URL}/summaries", timeout=3, json={
            "novel_id": request.novel_id, "reader_chapter": request.current_chapter})
    except httpx.HTTPError:
        pass
    return {
        "novel_id": request.novel_id,
        "user_id": request.user_id,
        "current_chapter": request.current_chapter,
    }


@app.get("/novels/{novel_id}/summaries")
async def novel_summaries(novel_id: int):
    """How far chapter summaries have got for a novel."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{INGESTION_SERVICE_URL}/summaries/{novel_id}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Ingestion service failed")
    return resp.json()


@app.get("/progress/{novel_id}/{user_id}")
def get_progress(novel_id: int, user_id: str = "default"):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT current_chapter, updated_at FROM reading_progress "
                "WHERE novel_id = :nid AND user_id = :uid"
            ),
            {"nid": novel_id, "uid": user_id},
        ).fetchone()

    if not row:
        return {"novel_id": novel_id, "user_id": user_id, "current_chapter": None}
    return {"novel_id": novel_id, "user_id": user_id, "current_chapter": row[0], "updated_at": str(row[1])}


# --- Health ---

@app.get("/health")
async def health():
    services = {
        "ingestion": INGESTION_SERVICE_URL,
        "retrieval": RETRIEVAL_SERVICE_URL,
        "generation": GENERATION_SERVICE_URL,
    }
    # Concurrently, with a short timeout: checked one after another at 5 s each, a single
    # hanging service made this endpoint slower than the container's own health probe.
    async def check(client, name, url):
        try:
            resp = await client.get(f"{url}/health")
            return name, resp.json()
        except Exception as e:
            return name, {"status": "unreachable", "error": str(e)}

    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
        status = dict(await asyncio.gather(
            *(check(client, name, url) for name, url in services.items())
        ))

    def ping_database():
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    try:
        await run_in_threadpool(ping_database)
        status["postgres"] = {"status": "ok"}
    except Exception as e:
        status["postgres"] = {"status": "unreachable", "error": str(e)}

    all_ok = all(s.get("status") == "ok" for s in status.values())
    return {"status": "ok" if all_ok else "degraded", "services": status}
