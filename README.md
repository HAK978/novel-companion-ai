# Novel Companion AI

A reading companion for long web novels. Ask questions about the story, look up characters,
or get a recap of where you left off — answered from the actual chapter text, and scoped so
nothing past your current chapter is ever used.

Runs as a set of Python microservices with a RAG pipeline, and ships an MCP server so
Model Context Protocol clients (Claude Code, Claude Desktop) can query it as tools.

## Why

Web serials often run to thousands of chapters. It's easy to forget who a character is or
what happened a few hundred chapters back, but wikis and summaries are written for people
who finished the book, so looking anything up means getting spoiled.

Here, every chunk of text is indexed with its chapter number, and retrieval filters on
`chapter_number <= current_chapter` before searching. The model never sees unread content,
so answers can't spoil what's ahead. The filter is applied server-side rather than asked for
in the prompt.

## Architecture

```
                    ┌──────────────┐        ┌─────────────────┐
   Next.js UI ─────▶│              │        │  MCP clients    │
                    │   Gateway    │◀───────│  (Claude Code,  │
   REST clients ───▶│    :8000     │        │   Desktop)      │
                    └──────┬───────┘        └─────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌───────────┐     ┌───────────┐      ┌────────────┐
  │ Ingestion │     │ Retrieval │      │ Generation │
  │   :8001   │     │   :8002   │      │   :8003    │
  └─────┬─────┘     └─────┬─────┘      └──────┬─────┘
        │                 │                   │
   ┌────▼────┐       ┌────▼─────┐        ┌────▼─────┐
   │ Celery  │       │ ChromaDB │        │  vLLM    │
   │ + Redis │──────▶│ (vectors)│        │  :8004   │
   └────┬────┘       └──────────┘        └──────────┘
        │
   ┌────▼───────┐
   │ PostgreSQL │  chapters, characters, relationships, progress
   └────────────┘
```

- **Gateway** — entry point; orchestrates retrieval → generation, manages novels and progress
- **Ingestion** — Celery workers: clean HTML → chunk → embed → store; pluggable source adapters
- **Retrieval** — vector search with chapter filtering and range scoping
- **Generation** — LLM inference via vLLM, falling back to transformers or the OpenAI API

## Stack

FastAPI · Celery · Redis · PostgreSQL 16 · ChromaDB · sentence-transformers
(`all-MiniLM-L6-v2`) · vLLM serving Mistral Nemo 12B · Next.js + TypeScript + Tailwind ·
Docker Compose

## MCP server

`services/mcp/server.py` exposes the system over stdio:

| Tool | Purpose |
|---|---|
| `list_novels` | Available novels and chapter counts |
| `query_novel` | Question answering, scoped to reading progress |
| `get_character` | Character lookup: aliases, relationships, mentions |
| `summarize_chapters` | Summarize a chapter range |
| `catch_me_up` | Recap recent chapters |
| `get_reading_progress` / `set_reading_progress` | Read and update position |
| `system_health` | Service health across the stack |

Content tools read stored progress and clamp the requested chapter to it, so the scoping
holds no matter what the client asks for.

```bash
claude mcp add novel-companion --scope user \
  --env GATEWAY_URL=http://localhost:8000 \
  -- python /path/to/services/mcp/server.py
```

## Running it

Needs Docker, Python 3.10+, and a CUDA GPU for local inference (or an `OPENAI_API_KEY` to
use a hosted model instead).

```bash
cp .env.example .env
bash scripts/start_services.sh
```

That starts PostgreSQL, Redis, and ChromaDB, waits for health, brings up vLLM, then the four
services. For the web UI: `cd frontend && npm install && npm run dev`.

Add a novel and ingest it:

```bash
curl -X POST localhost:8000/novels -H 'Content-Type: application/json' \
  -d '{"title": "My Novel", "source_type": "local_json"}'

curl -X POST localhost:8000/ingest/from-source -H 'Content-Type: application/json' \
  -d '{"novel_id": 1, "source_type": "local_json", "source_path": "/path/to/chapters"}'
```

Adapters handle directories of JSON chapter files and EPUB. Ingestion runs through Celery;
poll `/ingest/status/{task_id}` for progress. Roughly a second per chapter.

Ask something:

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "Who is the protagonist travelling with?", "novel_id": 1, "current_chapter": 500}'
```

## Layout

```
services/       gateway, ingestion, retrieval, generation, mcp
shared/         Pydantic schemas and database helpers
migrations/     PostgreSQL schema
scripts/        startup and seeding
frontend/       Next.js app
training/       dataset generation for fine-tuning
```

## Status

Working: ingestion at scale (tested on a 3,000-chapter novel), question answering,
summarization, character recall, progress tracking, MCP server, web UI.

Next: an evaluation harness, request tracing, streaming responses, hybrid search with
reranking, tests and CI, and fine-tuning on a distilled dataset.

## Data

No novel text is included here. Chapter content belongs to its authors; the ingestion
adapters read from a local directory you point them at.
