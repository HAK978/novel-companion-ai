
from celery.result import AsyncResult
from fastapi import FastAPI
from pydantic import BaseModel
from tasks import (
    celery_app,
    ingest_chapter,
    ingest_from_source,
    schedule_summaries,
    summary_status,
)

app = FastAPI(title="Ingestion Service")

# Plain `def` handlers: enqueueing talks to Redis synchronously, and FastAPI runs sync
# handlers in a threadpool rather than on the event loop.


class ChapterIngest(BaseModel):
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


@app.post("/ingest")
def ingest(chapter: ChapterIngest):
    task = ingest_chapter.delay({
        "novel_id": chapter.novel_id,
        "number": chapter.number,
        "title": chapter.title or f"Chapter {chapter.number}",
        "content": chapter.content,
        "volume": chapter.volume,
        "extract_entities": chapter.extract_entities,
    })
    return {
        "task_id": task.id,
        "status": "queued",
        "chapter_number": chapter.number,
    }


@app.post("/ingest/from-source")
def ingest_source(request: IngestFromSourceRequest):
    task = ingest_from_source.delay({
        "novel_id": request.novel_id,
        "source_type": request.source_type,
        "source_path": request.source_path,
        "max_chapters": request.max_chapters,
        "extract_entities": request.extract_entities,
    })
    return {
        "task_id": task.id,
        "status": "queued",
        "novel_id": request.novel_id,
        "source_type": request.source_type,
    }


@app.get("/ingest/status/{task_id}")
def ingest_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    if result.ready():
        response["result"] = result.result
    elif result.info:
        response["meta"] = result.info
    return response


class SummaryRequest(BaseModel):
    novel_id: int
    # the reader's chapter; defaults to the furthest any reader has got
    reader_chapter: int | None = None
    # summarize up to this chapter instead of the reader's position plus the lookahead
    up_to: int | None = None


@app.post("/summaries")
def summaries_schedule(request: SummaryRequest):
    return schedule_summaries(request.novel_id, request.reader_chapter, request.up_to)


@app.get("/summaries/{novel_id}")
def summaries_status(novel_id: int):
    return summary_status(novel_id)


@app.get("/health")
def health():
    try:
        inspect = celery_app.control.inspect()
        active = inspect.active()
        worker_count = len(active) if active else 0
        return {"status": "ok", "workers": worker_count}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
