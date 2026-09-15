from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from celery.result import AsyncResult

from tasks import celery_app, ingest_chapter, ingest_from_source

app = FastAPI(title="Ingestion Service")


class ChapterIngest(BaseModel):
    novel_id: Optional[int] = None
    number: int
    title: Optional[str] = None
    content: str
    volume: int = 1
    extract_entities: bool = False


class IngestFromSourceRequest(BaseModel):
    novel_id: int
    source_type: str
    source_path: str
    max_chapters: Optional[int] = None
    extract_entities: bool = False


@app.post("/ingest")
async def ingest(chapter: ChapterIngest):
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
async def ingest_source(request: IngestFromSourceRequest):
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
async def ingest_status(task_id: str):
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


@app.get("/health")
async def health():
    try:
        inspect = celery_app.control.inspect()
        active = inspect.active()
        worker_count = len(active) if active else 0
        return {"status": "ok", "workers": worker_count}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
