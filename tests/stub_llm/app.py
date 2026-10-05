"""Stand-in for an OpenAI-compatible model server, for end-to-end runs without a GPU.

It replies with the chapter labels present in the prompt, so a test can assert exactly
which chapters reached the model, through the real gateway, retrieval and generation.
"""

import re

from fastapi import FastAPI

MODEL = "stub-model"
app = FastAPI(title="Stub LLM")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    prompt = "\n".join(m.get("content") or "" for m in body.get("messages", []))
    if "precise entity extractor" in prompt:
        content = "[]"
    else:
        chapters = sorted({int(n) for n in re.findall(r"\[Chapter (\d+):", prompt)})
        content = "chapters seen: " + ",".join(map(str, chapters))
    return {
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    }
