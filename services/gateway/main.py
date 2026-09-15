from datetime import UTC, datetime

import httpx
from config import (
    DATABASE_URL,
    GENERATION_SERVICE_URL,
    INGESTION_SERVICE_URL,
    RETRIEVAL_SERVICE_URL,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

app = FastAPI(title="Novel Companion AI - Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DATABASE_URL)


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
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO novels (title, author, source_type, source_url)
                VALUES (:title, :author, :source_type, :source_url)
                RETURNING id, title, created_at
            """),
            {
                "title": request.title,
                "author": request.author,
                "source_type": request.source_type,
                "source_url": request.source_url,
            },
        )
        row = result.fetchone()
        conn.commit()
    return {"id": row[0], "title": row[1], "created_at": str(row[2])}


@app.get("/novels")
async def list_novels():
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
async def get_novel(novel_id: int):
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
        for table in ["conversations", "search_history", "reading_progress",
                       "chapter_summaries", "characters", "chapters"]:
            conn.execute(text(f"DELETE FROM {table} WHERE novel_id = :id"), {"id": novel_id})
        conn.execute(text("DELETE FROM novels WHERE id = :id"), {"id": novel_id})
        conn.commit()

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


# --- Ingestion ---

@app.post("/ingest")
async def ingest(request: IngestRequest):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{INGESTION_SERVICE_URL}/ingest",
            json=request.model_dump(),
        )
    return resp.json()


@app.post("/ingest/from-source")
async def ingest_from_source(request: IngestFromSourceRequest):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{INGESTION_SERVICE_URL}/ingest/from-source",
            json=request.model_dump(),
        )
    return resp.json()


@app.get("/ingest/status/{task_id}")
async def ingest_status(task_id: str):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{INGESTION_SERVICE_URL}/ingest/status/{task_id}")
    return resp.json()


# --- Characters ---

@app.get("/characters/list")
async def list_characters(novel_id: int, current_chapter: int = 9999):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{RETRIEVAL_SERVICE_URL}/characters",
            params={"novel_id": novel_id, "current_chapter": current_chapter},
        )
    return resp.json()


@app.get("/characters/{name}")
async def character_recall(name: str, novel_id: int, current_chapter: int = 9999):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{RETRIEVAL_SERVICE_URL}/characters/{name}",
            params={"novel_id": novel_id, "current_chapter": current_chapter},
        )
    return resp.json()


# --- Query ---

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    collection_name = f"novel_{request.novel_id}"

    async with httpx.AsyncClient(timeout=120) as client:
        retrieval_resp = await client.post(
            f"{RETRIEVAL_SERVICE_URL}/search",
            json={
                "query": request.query,
                "current_chapter": request.current_chapter,
                "n_results": request.n_results,
                "collection_name": collection_name,
            },
        )
        retrieval_data = retrieval_resp.json()
        results = retrieval_data.get("results", [])

        if not results:
            return QueryResponse(answer=None, sources=[])

        # Label each passage with its chapter so the model can see (and say)
        # when retrieved passages come from distant parts of the story
        context_chunks = [
            f"[Chapter {r['chapter_number']}: {r.get('chapter_title', '')}]\n{r['text']}"
            for r in results
        ]
        gen_resp = await client.post(
            f"{GENERATION_SERVICE_URL}/generate",
            json={
                "query": request.query,
                "context_chunks": context_chunks,
                "conversation_context": request.conversation_context,
            },
        )
        gen_data = gen_resp.json()

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO search_history (novel_id, query, results_count, timestamp)
                VALUES (:nid, :q, :c, :ts)
            """),
            {"nid": request.novel_id, "q": request.query, "c": len(results), "ts": datetime.now(UTC)},
        )
        conn.commit()

    return QueryResponse(
        answer=gen_data.get("answer"),
        model_used=gen_data.get("model_used", "none"),
        sources=results,
    )


# --- Summarization ---

class SummarizeRequest(BaseModel):
    novel_id: int
    start_chapter: int
    end_chapter: int
    current_chapter: int | None = None


class CatchMeUpRequest(BaseModel):
    novel_id: int
    user_id: str = "default"


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    current_ch = request.current_chapter or request.end_chapter

    # Check cache first
    with engine.connect() as conn:
        cached = conn.execute(
            text("""
                SELECT summary FROM chapter_summaries
                WHERE novel_id = :nid AND start_chapter = :s AND end_chapter = :e
            """),
            {"nid": request.novel_id, "s": request.start_chapter, "e": request.end_chapter},
        ).fetchone()

    if cached:
        return {
            "summary": cached[0],
            "start_chapter": request.start_chapter,
            "end_chapter": request.end_chapter,
            "cached": True,
        }

    # Retrieve chunks from the chapter range
    collection_name = f"novel_{request.novel_id}"
    async with httpx.AsyncClient(timeout=120) as client:
        # Search for content across the chapter range
        retrieval_resp = await client.post(
            f"{RETRIEVAL_SERVICE_URL}/search",
            json={
                "query": f"summary of events in chapters {request.start_chapter} to {request.end_chapter}",
                "current_chapter": current_ch,
                "n_results": 10,
                "collection_name": collection_name,
                # floor the search to the requested range; without it,
                # similar chunks from much earlier arcs pollute the summary
                "min_chapter": request.start_chapter,
            },
        )
        results = retrieval_resp.json().get("results", [])

        if not results:
            return {"error": "No content found for this chapter range"}

        # Generate summary
        gen_resp = await client.post(
            f"{GENERATION_SERVICE_URL}/generate",
            json={
                "query": (
                    f"Summarize the key events, character developments, and plot points "
                    f"from chapters {request.start_chapter} to {request.end_chapter}. "
                    f"Be comprehensive but concise."
                ),
                "context_chunks": [
                    f"[Chapter {r['chapter_number']}: {r.get('chapter_title', '')}]\n{r['text']}"
                    for r in results
                ],
            },
        )
        gen_data = gen_resp.json()
        summary = gen_data.get("answer", "")

    if not summary:
        return {"error": "Failed to generate summary"}

    # Cache the summary
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO chapter_summaries (novel_id, start_chapter, end_chapter, summary)
                VALUES (:nid, :s, :e, :sum)
                ON CONFLICT (start_chapter, end_chapter) DO UPDATE SET summary = :sum
            """),
            {"nid": request.novel_id, "s": request.start_chapter, "e": request.end_chapter, "sum": summary},
        )
        conn.commit()

    return {
        "summary": summary,
        "start_chapter": request.start_chapter,
        "end_chapter": request.end_chapter,
        "cached": False,
    }


@app.post("/catch-me-up")
async def catch_me_up(request: CatchMeUpRequest):
    # Get reading progress
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT current_chapter FROM reading_progress WHERE novel_id = :nid AND user_id = :uid"),
            {"nid": request.novel_id, "uid": request.user_id},
        ).fetchone()

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
    ))

    # Get characters introduced in this range
    async with httpx.AsyncClient(timeout=10) as client:
        chars_resp = await client.get(
            f"{RETRIEVAL_SERVICE_URL}/characters",
            params={"novel_id": request.novel_id, "current_chapter": current_chapter},
        )
    all_chars = chars_resp.json()
    recent_chars = [c for c in all_chars if c.get("first_appearance", 0) >= start]

    return {
        "user_id": request.user_id,
        "current_chapter": current_chapter,
        "summary_range": {"start": start, "end": current_chapter},
        "summary": summary_resp.get("summary", ""),
        "cached": summary_resp.get("cached", False),
        "new_characters": recent_chars,
    }


# --- Reading Progress ---

@app.post("/progress")
async def update_progress(request: ProgressRequest):
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
    return {
        "novel_id": request.novel_id,
        "user_id": request.user_id,
        "current_chapter": request.current_chapter,
    }


@app.get("/progress/{novel_id}/{user_id}")
async def get_progress(novel_id: int, user_id: str = "default"):
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
    status = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in services.items():
            try:
                resp = await client.get(f"{url}/health")
                status[name] = resp.json()
            except Exception as e:
                status[name] = {"status": "unreachable", "error": str(e)}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["postgres"] = {"status": "ok"}
    except Exception as e:
        status["postgres"] = {"status": "unreachable", "error": str(e)}

    all_ok = all(s.get("status") == "ok" for s in status.values())
    return {"status": "ok" if all_ok else "degraded", "services": status}
