# Novel Companion AI — working notes

Spoiler-aware reading companion for long web novels, built as FastAPI microservices:
Celery/Redis async ingestion into vector embeddings, entity extraction into a PostgreSQL
relationship graph, and summarization / character recall / Q&A backed by RAG that is
scoped to the reader's progress.

## Architecture

- **Gateway** (8000) — `services/gateway/main.py`. Single entry point, routes via
  httpx.AsyncClient, CORS enabled. Endpoints: /novels CRUD, /ingest, /ingest/from-source,
  /query, /characters/list, /characters/{name}, /summarize, /catch-me-up, /progress,
  /health (aggregated).
- **Ingestion** (8001) — `services/ingestion/`. Celery tasks (`tasks.py`, core logic in
  `_process_chapter()`): clean HTML → chunk (~400 words, sentence-boundary, `chunking.py`)
  → embed (sentence-transformers all-MiniLM-L6-v2) → store in ChromaDB → chapter record in
  PostgreSQL → optional entity extraction via generation service. Source adapters in
  `adapters/` (local_json handles nested dirs via rglob, epub).
- **Retrieval** (8002) — `services/retrieval/`. ChromaDB similarity search with spoiler
  filter `chapter_number <= current_chapter`, optional `min_chapter` floor, adaptive
  n_results, character lookup from PostgreSQL.
- **Generation** (8003) — `services/generation/`. `LLMHandler` prefers vLLM (probes
  `VLLM_URL`), falls back to in-process transformers, then OpenAI. Endpoints: /generate,
  /extract-entities, /models, /health.
- **MCP** — `services/mcp/server.py`, 8 tools over stdio. Spoiler rule enforced
  server-side: content tools resolve stored progress and clamp requested chapters to it.
- **Frontend** (3000) — `frontend/`, Next.js + TypeScript + Tailwind.
- **Infra** — docker-compose.yml: PostgreSQL 16 (5432), Redis 7 (6379), ChromaDB
  (host 8005 → container 8000). App services run via uvicorn on the host in dev.

## Database

PostgreSQL `novel_companion`. Migrations in `migrations/`. Tables: novels, chapters,
characters (unique on (novel_id, name)), character_aliases, character_relationships, character_mentions,
chapter_summaries, reading_progress, search_history, conversations. All scoped by novel_id.
Each novel gets its own ChromaDB collection (`novel_{id}`).

## Operations

- **Start everything:** `bash scripts/start_services.sh` (idempotent). Starts infra, applies
  pending migrations, then vLLM (loads the model onto GPU, takes minutes), then services.
- **Database state:** `python scripts/migrate.py --status`.
- **Bulk ingest:** POST /ingest/from-source with `extract_entities: false` (default).
  ~1s/chapter; extraction on is ~20s/chapter.

## Gotchas — do not regress

- **Schema changes only through `scripts/migrate.py`**, which records each file in
  `schema_migrations`. There is no docker-entrypoint initdb mount: it ran files once, recorded
  nothing, and is how migration 003 sat unapplied. `--baseline FILE` only records history.
- **Embedding model is pinned** in `services/{ingestion,retrieval}/vector_store.py` (identical
  copies; a test fails if they drift). Changing models means a new collection plus
  re-ingestion. Collections created before pinning are labelled `"default"` and must go
  through `scripts/migrate_embeddings.py` (copies vectors, no re-embedding). Pinned
  collections measure cosine distance, so `relevance_score` is cosine similarity.
- **Re-ingestion upserts then prunes** (`tasks.py` `_replace_chapter_chunks`). Chroma's `add()`
  silently skips existing ids, so plain `add()` would keep stale text.
- **Character graph is order-independent and reveal-aware.** Chapters may be processed in any
  order: first_appearance / alias first_chapter / relationship first_chapter keep the
  earliest chapter, and a character's description comes from its earliest chapter. Aliases
  live in `character_aliases` with their reveal chapter; reads filter on it for both
  matching and display.
- **`current_chapter` is required** on every character endpoint (it defaulted to 9999: the
  filter failed open). **`novel_id` is required** end to end and NOT NULL in every table.
- **No blocking calls in `async def` handlers** (`tests/test_event_loop.py` enforces this):
  use plain `def` handlers or `run_in_threadpool`.
- **Chapter numbers are reading order** and the spoiler filter trusts them. The JSON adapter
  sorts numerically ("2" before "10"), prefers `serial`, and rejects duplicate numbers.
- ChromaDB clients are cached per process; per-request clients leak server FDs.
- Chroma volume mounts at `/data`, ulimit nofile 65536, healthcheck uses bash /dev/tcp.
- Range queries (`/summarize`, `/catch-me-up`) pass `min_chapter`; context chunks are labelled
  `[Chapter N: title]`. Summary cache is keyed by (novel_id, start, end).
- Do not restart the generation service while vLLM is down: its transformers fallback loads
  a 24 GB model onto a shared GPU (removal planned).
- In `tasks.py`, SQLAlchemy `text` is imported as `sa_text` to avoid shadowing.

## Roadmap

Single OpenAI-compatible LLM client, containerized services, evaluation harness (RAGAS
metrics + deterministic spoiler-leakage check), populated character graph, hybrid search with
reranking, SSE streaming, tracing via Langfuse/OpenTelemetry.

**Keep this file updated as work progresses** so a fresh session can resume from here.
