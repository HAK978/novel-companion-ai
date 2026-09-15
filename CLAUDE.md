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
characters (unique index on (novel_id, name)), character_relationships, character_mentions,
chapter_summaries, reading_progress, search_history, conversations. All scoped by novel_id.
Each novel gets its own ChromaDB collection (`novel_{id}`).

## Operations

- **Start everything:** `bash scripts/start_services.sh` (idempotent). Starts infra, then
  vLLM (loads the model onto GPU, takes minutes), then services in dependency order.
- **Bulk ingest:** POST /ingest/from-source with `extract_entities: false` (default).
  ~1s/chapter; extraction on is ~20s/chapter.

## Gotchas — do not regress

- ChromaDB clients are cached per process (`tasks.py` `_get_collection`, `search.py`
  `_get_client`). Per-request clients leak server FDs and kill bulk ingestion.
- Chroma volume mounts at `/data`, ulimit nofile 65536, healthcheck uses bash /dev/tcp
  (the image ships no curl).
- `character_mentions` inserts must include novel_id; DELETE /novels also matches
  mentions/relationships via character ids.
- Range queries (`/summarize`, `/catch-me-up`) must pass `min_chapter`, and context chunks
  are labeled `[Chapter N: title]` — without both, thematically similar passages from much
  earlier arcs pollute recent-events answers.
- `chapter_summaries` caches by range, not code version; clear it after retrieval changes.
- In `tasks.py`, SQLAlchemy `text` is imported as `sa_text` to avoid shadowing.

## Roadmap

Evaluation harness (RAGAS metrics + deterministic spoiler-leakage check), tracing via
Langfuse/OpenTelemetry, SSE streaming, hybrid search with reranking, pytest suite + CI,
QLoRA fine-tuning with MLflow tracking.

**Keep this file updated as work progresses** so a fresh session can resume from here.
