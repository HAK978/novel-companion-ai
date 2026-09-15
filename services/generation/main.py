
from fastapi import FastAPI
from llm_handler import LLMHandler
from pydantic import BaseModel

app = FastAPI(title="Generation Service")
llm = LLMHandler()


class GenerateRequest(BaseModel):
    query: str
    context_chunks: list[str]
    conversation_context: str = ""


class GenerateResponse(BaseModel):
    answer: str | None
    model_used: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    answer = llm.generate(
        query=request.query,
        context_chunks=request.context_chunks,
        conversation_context=request.conversation_context,
    )
    return GenerateResponse(answer=answer, model_used=llm.model or "none")


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
[{{"name": "Sunny", "aliases": ["Sunless", "Mongrel"], "role": "fights the creature", "relationships": [{{"character": "Nephis", "type": "ally"}}]}}]

Chapter {request.chapter_number} text:
{request.chapter_text[:6000]}

JSON array:"""

    raw = llm.generate(query=prompt, context_chunks=[], conversation_context="")
    if not raw:
        return {"characters": [], "error": "LLM unavailable"}

    # Try to parse JSON from the response
    import json
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
async def models():
    return llm.get_available_models()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": llm.backend,
        "model": llm.model,
    }
