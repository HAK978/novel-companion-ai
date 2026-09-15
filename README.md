# Novel Companion AI

A spoiler-aware reading companion for long-running web novels. Ask questions about a
3,000-chapter story and get answers grounded in the actual text — with a hard guarantee
that nothing past your current chapter is ever used.

Built as Python microservices with a RAG pipeline, and exposed both as a web app and as an
**MCP server** so any Model Context Protocol client (Claude Code, Claude Desktop) can query
your library as a set of tools.

---

## The problem

Progression fantasy and web serials run to thousands of chapters. Readers forget who
characters are, what happened 400 chapters ago, and where a plotline left off. Every
existing option — wikis, fan summaries, asking an LLM directly — spoils the story, because
none of them know where you are in it.

## The approach

Every text chunk is indexed with its chapter number. Retrieval applies a metadata filter
(`chapter_number <= current_chapter`) before similarity search, so the model physically
cannot see unread content. The rule is enforced server-side, not in the prompt — a client
asking about chapter 3000 while the reader is on 500 gets results scoped to 500.

---

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

**Gateway** — single entry point, orchestrates retrieval → generation, novel/progress management.
**Ingestion** — Celery workers: clean HTML → chunk → embed → store, with pluggable source adapters.
**Retrieval** — vector search with spoiler filtering and chapter-range scoping.
**Generation** — LLM inference via vLLM (OpenAI-compatible), with transformers and OpenAI fallbacks.

## Stack

| Layer | Choice |
|---|---|
| Services | FastAPI, httpx (async) |
| Async jobs | Celery + Redis |
| Vector store | ChromaDB |
| Relational | PostgreSQL 16 |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM serving | vLLM — Mistral Nemo 12B Instruct |
| Tool interface | Model Context Protocol (Python SDK) |
| Frontend | Next.js, TypeScript, Tailwind |
| Orchestration | Docker Compose |

---

## MCP server

`services/mcp/server.py` exposes the system as tools over stdio. Register it and Claude can
answer novel questions by calling into the pipeline directly:

| Tool | Purpose |
|---|---|
| `list_novels` | Available novels and ingested chapter counts |
| `query_novel` | RAG question answering, scoped to reading progress |
| `get_character` | Character lookup: aliases, relationships, mentions |
| `summarize_chapters` | Summarize a chapter range (clamped to progress) |
| `catch_me_up` | Recap recent chapters for a returning reader |
| `get_reading_progress` / `set_reading_progress` | Read and update position |
| `system_health` | Service health across the stack |

Every content tool resolves stored reading progress server-side and clamps the requested
chapter to it, so the spoiler guarantee holds regardless of what the client asks for.

```bash
claude mcp add novel-companion --scope user \
  --env GATEWAY_URL=http://localhost:8000 \
  -- python /path/to/services/mcp/server.py
```

---

## Running it

**Prerequisites:** Docker, Python 3.10+, a CUDA GPU for local inference (or set
`OPENAI_API_KEY` to use a hosted model instead).

```bash
cp .env.example .env          # adjust credentials
bash scripts/start_services.sh # infra + vLLM + all four services
```

The script brings up PostgreSQL, Redis, and ChromaDB, waits for health, starts vLLM, then
the services in dependency order. Frontend: `cd frontend && npm install && npm run dev`.

**Ingesting a novel:**

```bash
curl -X POST localhost:8000/novels -H 'Content-Type: application/json' \
  -d '{"title": "My Novel", "source_type": "local_json"}'

curl -X POST localhost:8000/ingest/from-source -H 'Content-Type: application/json' \
  -d '{"novel_id": 1, "source_type": "local_json", "source_path": "/path/to/chapters"}'
```

Source adapters handle directories of JSON chapter files and EPUB. Ingestion runs
asynchronously through Celery; poll `/ingest/status/{task_id}` for progress.

**Asking a question:**

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "Who is the protagonist travelling with?", "novel_id": 1, "current_chapter": 500}'
```

---

## Engineering notes

A few problems worth documenting, since they shaped the design:

**Connection leak under bulk ingestion.** Ingesting 3,000 chapters failed at roughly the
same chapter twice. Root cause: a new ChromaDB HTTP client per chapter, each leaking
server-side file descriptors until the container hit its 1024 limit and its embedded SQLite
store began failing. Fixed by caching one client per worker process, with the container FD
limit raised as a backstop.

**Cross-arc retrieval pollution.** Summary requests for a recent chapter range returned
content from hundreds of chapters earlier. Chapter numbers in a query string contribute
almost nothing to an embedding, so a thematically similar passage from an earlier arc
outranked the correct window — and the model then merged both into one confident, wrong
narrative. Fixed with an explicit `min_chapter` floor on range queries, chapter-labeled
context passages, and prompt instructions against merging distant chapters. The general
lesson: vector similarity measures topical resemblance, not contextual appropriateness —
anything structured belongs in metadata filters, not in the text you hope the embedding picks up.

**Silent persistence failure.** The vector store volume was mounted at a path the image
doesn't use, so data lived in the container's ephemeral layer and vanished on restart. The
healthcheck that should have caught trouble was itself broken — it invoked a binary the
image doesn't ship, so the container reported unhealthy indefinitely and the signal was ignored.

---

## Status and roadmap

Working: ingestion at scale (3,000+ chapters), spoiler-scoped RAG question answering,
chapter summarization, reading-progress tracking, MCP server, web UI.

Planned: evaluation harness (RAGAS metrics plus a deterministic spoiler-leakage check),
tracing/observability, response streaming, hybrid search with reranking, test suite and CI,
and QLoRA fine-tuning on a distilled dataset.

## Note on data

No novel text is included in this repository. Chapter content is copyrighted by its
respective authors; the ingestion adapters read from a local directory you supply.
