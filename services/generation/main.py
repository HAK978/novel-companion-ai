import json

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from llm_client import LLMClient, LLMError
from pydantic import BaseModel

app = FastAPI(title="Generation Service")
llm = LLMClient()


class GenerateRequest(BaseModel):
    query: str
    context_chunks: list[str]
    conversation_context: str = ""
    # the reader's position, so the prompt can forbid anything later
    current_chapter: int | None = None


class GenerateResponse(BaseModel):
    answer: str
    model_used: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        answer = await run_in_threadpool(
            llm.answer, request.query, request.context_chunks,
            request.conversation_context, request.current_chapter,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return GenerateResponse(answer=answer, model_used=llm.model)


class ExtractRequest(BaseModel):
    chapter_text: str
    chapter_number: int


@app.post("/extract-entities")
async def extract_entities(request: ExtractRequest):
    """Extract character entities from a chapter using the LLM."""
    prompt = f"""You are a precise entity extractor. Extract all characters mentioned in this chapter.

For each character, provide:
- name: their primary name
- aliases: list of other names/titles they go by (empty list if none)
- role: one short phrase describing what they do in this chapter
- relationships: list of objects with "character" (name) and "type" (ally/enemy/mentor/family/rival/unknown)

Return ONLY a valid JSON array. No explanation, no markdown, just JSON.

Example format:
[{{"name": "Mara Vell", "aliases": ["the Archivist"], "role": "searches the flooded library", "relationships": [{{"character": "Tobin", "type": "ally"}}]}}]

Chapter {request.chapter_number} text:
{request.chapter_text[:6000]}

JSON array:"""

    # Sent as written. It used to go through the Q&A template, which appended "write
    # 100-400 words and cite chapters" after "return ONLY a JSON array".
    try:
        raw = await run_in_threadpool(llm.complete, prompt)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        # Handle cases where LLM wraps in markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        characters = json.loads(cleaned)
        if not isinstance(characters, list):
            characters = []
    except json.JSONDecodeError:
        characters = []

    return {"characters": characters, "raw": raw}


@app.get("/models")
def models():
    return {"endpoint": llm.base_url, "model": llm.model}


@app.get("/health")
def health():
    # reports whether the model is actually reachable; it used to say "ok" unconditionally
    return llm.status()
