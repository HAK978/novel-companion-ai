from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# --- Ingestion ---

class ChapterIngest(BaseModel):
    number: int
    title: Optional[str] = None
    content: str  # raw HTML or text
    volume: Optional[int] = None


class IngestResponse(BaseModel):
    task_id: str
    status: str
    chapter_number: int


# --- Retrieval ---

class SearchRequest(BaseModel):
    query: str
    current_chapter: int
    n_results: int = 5


class ChunkResult(BaseModel):
    text: str
    chapter_number: int
    chapter_title: Optional[str] = None
    relevance_score: float


class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResult]


class CharacterResponse(BaseModel):
    name: str
    aliases: List[str] = []
    first_appearance: Optional[int] = None
    description: Optional[str] = None
    mentions: List[dict] = []
    relationships: List[dict] = []


# --- Generation ---

class GenerateRequest(BaseModel):
    query: str
    context_chunks: List[str]
    conversation_context: str = ""


class GenerateResponse(BaseModel):
    answer: str
    model_used: str


# --- Gateway ---

class QueryRequest(BaseModel):
    query: str
    current_chapter: int
    n_results: int = 5


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    sources: List[ChunkResult]


class SummarizeRequest(BaseModel):
    start_chapter: int
    end_chapter: int
    current_chapter: int


class SummarizeResponse(BaseModel):
    summary: str
    start_chapter: int
    end_chapter: int
    cached: bool = False


class ProgressRequest(BaseModel):
    current_chapter: int
    user_id: str = "default"


class ProgressResponse(BaseModel):
    user_id: str
    current_chapter: int
    updated_at: datetime


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    dependencies: dict = {}
