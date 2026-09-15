"""
Novel Companion MCP server.

Exposes the Novel Companion AI gateway as Model Context Protocol tools so any
MCP client (Claude Code, Claude Desktop, ...) can query ingested novels.

Spoiler rule enforced here, server-side: every content tool resolves the
user's stored reading progress and never returns material past that chapter,
regardless of what the client asks for. The client LLM orchestrates; this
layer decides what data it may see.

Run:  python server.py          (stdio transport)
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
USER_ID = os.environ.get("NOVEL_USER_ID", "default")

mcp = FastMCP("novel-companion")


def _get(path: str, params: dict | None = None) -> dict | list:
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{GATEWAY_URL}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


def _post(path: str, payload: dict, timeout: float = 180) -> dict:
    # generous timeout: /query and /summarize run a local 12B model
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{GATEWAY_URL}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


def _reading_progress(novel_id: int) -> int | None:
    data = _get(f"/progress/{novel_id}/{USER_ID}")
    return data.get("current_chapter")


def _scoped_chapter(novel_id: int, requested: int | None) -> int | dict:
    """Resolve the chapter ceiling for content tools.

    Uses stored reading progress; an explicit request may only narrow the
    window, never widen it past progress. Returns an error dict if no
    progress is set.
    """
    progress = _reading_progress(novel_id)
    if progress is None:
        return {
            "error": "No reading progress set for this novel. "
                     "Ask the reader what chapter they are on, then call "
                     "set_reading_progress before querying content."
        }
    if requested is None:
        return progress
    return min(requested, progress)


@mcp.tool()
def list_novels() -> list:
    """List all novels available in the companion, with ids, titles, authors,
    and how many chapters are ingested."""
    return _get("/novels")


@mcp.tool()
def get_reading_progress(novel_id: int) -> dict:
    """Get the reader's current chapter for a novel. Content tools never
    reveal anything past this chapter."""
    return _get(f"/progress/{novel_id}/{USER_ID}")


@mcp.tool()
def set_reading_progress(novel_id: int, chapter: int) -> dict:
    """Update the reader's current chapter for a novel. Call this when the
    reader says they've read further (or want to rewind)."""
    return _post("/progress", {
        "novel_id": novel_id,
        "current_chapter": chapter,
        "user_id": USER_ID,
    })


@mcp.tool()
def query_novel(question: str, novel_id: int, current_chapter: int | None = None) -> dict:
    """Ask any question about a novel's story, characters, or events.
    Answered with RAG over the actual chapter text, restricted to chapters
    the reader has already read (their stored progress). Pass current_chapter
    only to narrow the window further, e.g. 'as of chapter 100, what did X know?'"""
    scoped = _scoped_chapter(novel_id, current_chapter)
    if isinstance(scoped, dict):
        return scoped
    result = _post("/query", {
        "query": question,
        "novel_id": novel_id,
        "current_chapter": scoped,
        "n_results": 5,
    })
    return {
        "answer": result.get("answer"),
        "scoped_to_chapter": scoped,
        "sources": [
            {"chapter": s.get("chapter_number"), "title": s.get("chapter_title")}
            for s in result.get("sources", [])
        ],
    }


@mcp.tool()
def get_character(name: str, novel_id: int, current_chapter: int | None = None) -> dict:
    """Look up a character in the structured character database: description,
    aliases, first appearance, relationships, recent mentions — limited to what
    the reader has read so far. Only populated for novels ingested with entity
    extraction; if the character is not found, fall back to query_novel, which
    answers from the chapter text itself."""
    scoped = _scoped_chapter(novel_id, current_chapter)
    if isinstance(scoped, dict):
        return scoped
    return _get(f"/characters/{name}", params={
        "novel_id": novel_id,
        "current_chapter": scoped,
    })


@mcp.tool()
def summarize_chapters(novel_id: int, start_chapter: int, end_chapter: int) -> dict:
    """Summarize a chapter range of a novel. The range is clamped to the
    reader's progress so it can never summarize unread chapters."""
    scoped = _scoped_chapter(novel_id, end_chapter)
    if isinstance(scoped, dict):
        return scoped
    if start_chapter > scoped:
        return {"error": f"Reader has only read up to chapter {scoped}; "
                         f"cannot summarize starting at {start_chapter}."}
    return _post("/summarize", {
        "novel_id": novel_id,
        "start_chapter": start_chapter,
        "end_chapter": scoped,
        "current_chapter": scoped,
    })


@mcp.tool()
def catch_me_up(novel_id: int) -> dict:
    """Recap the last stretch of the novel up to the reader's current
    chapter — for readers returning after a break. Includes recently
    introduced characters."""
    return _post("/catch-me-up", {"novel_id": novel_id, "user_id": USER_ID})


@mcp.tool()
def system_health() -> dict:
    """Check health of the Novel Companion services (gateway, ingestion,
    retrieval, generation, database)."""
    return _get("/health")


if __name__ == "__main__":
    mcp.run()
