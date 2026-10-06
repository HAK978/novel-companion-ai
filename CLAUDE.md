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
  → embed (all-MiniLM-L6-v2, Chroma's ONNX build; bulk ingests embed 32 chapters per call in
  `_ingest_chapters()`) → store in ChromaDB → chapter record in
  PostgreSQL → optional entity extraction via generation service. Source adapters in
  `adapters/` (local_json handles nested dirs via rglob, epub). Chapter summaries: the
  `summary-worker` (Celery queue `summaries`) writes one per chapter through generation's
  `/chapter-summary`, up to the furthest reader + `SUMMARY_LOOKAHEAD` (100), newest first back
  from the reader, then ahead. Scheduled as soon as an ingest has indexed the reader's chapter,
  again when it ends, and on every progress update (`POST /summaries`, `GET /summaries/{novel_id}`).
- **Retrieval** (8002) — `services/retrieval/`. ChromaDB similarity search with spoiler
  filter `chapter_number <= current_chapter`, optional `min_chapter` floor, adaptive
  n_results, character lookup from PostgreSQL.
- **Generation** (8003) — `services/generation/`. One OpenAI-compatible client
  (`llm_client.py`) configured by `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`; the default
  is the compose `vllm` service (`http://vllm:8000/v1`, host port 8004). No GPU code: an
  unreachable endpoint is a 502 and `/health` says degraded. Answers send spoiler rules as a
  system message plus the reader's chapter. Endpoints: /generate, /extract-entities, /chapter-summary, /models, /health.
- **MCP** — `services/mcp/server.py`, 8 tools over stdio. Spoiler rule enforced
  server-side: content tools resolve stored progress and clamp requested chapters to it.
- **Frontend** (3000) — `frontend/`, Next.js + TypeScript + Tailwind.
- **Compose** — docker-compose.yml runs everything: PostgreSQL 16 (5432), Redis 7 (6379),
  ChromaDB (host 8005 → container 8000), a one-shot `migrate` job, the four services, the
  Celery ingestion and summary workers, the frontend, and vLLM under the `gpu` profile. Ports bind 127.0.0.1 only.
  docker-compose.ci.yml swaps vLLM for a stub model (`tests/stub_llm/`) for e2e runs.

## Database

PostgreSQL `novel_companion`. Migrations in `migrations/`. Tables: novels, chapters,
characters (unique on (novel_id, name)), character_aliases, character_relationships, character_mentions,
chapter_summaries (one per chapter), range_summaries (cache of /summarize answers), reading_progress,
search_history. All scoped by novel_id.
Each novel gets its own ChromaDB collection (`novel_{id}`).

## Operations

- **Start everything:** `docker compose up -d` (all but the model); after code changes
  `docker compose up -d --build`. The model: `scripts/start_vllm.sh` (idle GPU, ~2 min to
  healthy); `docker compose stop vllm` releases the GPU. Never leave it running unattended.
- **Database state:** `docker compose run --rm migrate python scripts/migrate.py --status`.
- **Chapter summaries:** need the model served; tasks retry every 5 min while it is not.
  Extend or backfill: `curl -X POST localhost:8001/summaries -H 'content-type: application/json'
  -d '{"novel_id": N}'` (add `"up_to": M` to go further). ~0.7 s per chapter with 16 model calls
  in flight (`SUMMARY_PARALLEL`; 8 was ~2.4 s). After changing the name check:
  `docker compose exec ingestion python -c "from tasks import recheck_summary_flags as r; print(r(N))"`.
- **Bulk ingest:** POST /ingest/from-source with `extract_entities: false` (default) and a
  `source_path` under `/novels` (host `NOVEL_DATA_DIR`). ~0.12 s/chapter (300 chapters in 35 s:
  embedding ~80 ms, Chroma writes ~45 ms; one chapter per embedding call took 52.6 s), so a
  3,000-chapter book is ~6 min. Chapters are searchable as they land; the task's progress meta
  has `searchable_up_to`, which the web UI shows. Extraction ~20s/chapter.
- **End-to-end tests:** run in a separate project so the dev data stays out:
  `NOVEL_DATA_DIR=<dir> docker compose -p novel-e2e -f docker-compose.yml -f docker-compose.ci.yml up -d --build --wait`,
  then `RUN_INTEGRATION=1 NOVEL_DATA_DIR=<dir> pytest -m integration` (stop the dev stack
  first: same ports). Tear down with `down`, then `docker volume rm novel-e2e_...` by name.

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
  `[Chapter N: title]`. Their cache (`range_summaries`) is keyed by (novel_id, start, end).
- **The prompt is part of the spoiler defense.** A model may know a famous book: at chapter
  5 of the Hound it named the culprit from memory 7/20 times until the rules went into the
  system message (0/40). Keep `system_prompt()` in `llm_client.py`.
- **EPUB chapters come from the spine plus the table of contents**, front and back matter
  dropped by title, Project Gutenberg boilerplate clipped at its START/END markers.
- Never `pgrep -f`/`pkill -f` a pattern containing literal text from your own command (it
  matches the shell running it and kills it, exit 144). Use a bracket: `800[3]`, `vll[m]`.
- In `tasks.py`, SQLAlchemy `text` is imported as `sa_text` to avoid shadowing.
- **Never `docker compose down -v` on the dev project:** its volumes hold the ingested novels
  (pg_dump backups live in `backups/`).
- **Containers cannot reach the host here** (a firewall drops container-to-host traffic), so
  the model runs as the compose `vllm` service, never a host process via host.docker.internal.
- Dockerfiles copy code with `--chmod=755`: the host umask (007) makes files 660, unreadable
  to the containers' non-root user.
- The worker and vLLM run as `HOST_UID:HOST_GID` (read mounted novels; cache files stay the
  user's). That UID has no passwd entry, so vLLM needs `USER` set (`getpass.getuser()`), and
  its cache mounts at `/vllm-cache`: a mount under `/tmp/.cache` made Docker create that
  directory as root and FlashInfer could not write beside it.
- **No function-level imports of local modules in Celery tasks:** Celery puts the working
  directory on `sys.path` only while importing the app. The host worker hid this because
  `~/.bashrc` builds PYTHONPATH with a trailing `:` (an empty entry means the current directory).
- Requirements are lockfiles: edit `requirements.in`, then `uv pip compile requirements.in
  -o requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28`.
- Chroma is pinned by digest: the tag `1.0.0` names a different image than the one that wrote
  the data.
- vLLM has `restart: "no"`: on shared GPUs it must not reclaim a GPU after a reboot.
- **`chapter_summaries` is per chapter; `range_summaries` caches /summarize answers** (it was
  named chapter_summaries until migration 006). Bump `SUMMARY_PROMPT_VERSION` in
  `llm_client.py` whenever the summary prompt changes: scheduling then rewrites older ones.
- Keep `summary-worker` at concurrency 1: each task already has `SUMMARY_PARALLEL` model calls
  in flight, and two tasks could summarize the same chapters. Chapters waiting for a summary
  are tracked in Redis (`summaries:pending:<novel>`) so repeated scheduling adds no duplicates.
- Summaries record names their chapter (title included) never mentions (`unverified_names`):
  possible additions from the model's memory of the book. Heading words ("Chapter 12 Summary
  (Part 1 of 2)") are ignored. On Shadow Slave 1-1391 it flags 35 of 1,390: one real leak
  (chapter 33's summary used "Underworld", first in the book at chapter 250), four words the
  book never uses ("Cassandra", "Fawkes", "Sightless", "Transcender"), and 30 harmless (names
  from earlier chapters). The Hound: 6 of 15, including an invented name (chapter 11 calls Mr.
  Frankland "Captain John Sebastian Morland Frankland") and the book's title in chapter 1's
  summary. Checking against the book up to the chapter instead (a per-novel index of where each
  word first appears) would leave 6 and 3 flags with every real one (the extra Shadow Slave flag
  is "Antarctic" against the text's "Antarctica"), and would also answer "is this name in what
  I've read?".
- **Chapter summaries get who-did-what wrong about 1 time in 5.** Read against their chapters:
  Shadow Slave 19 of 25 accurate, the 5 wrong ones mostly crediting an event to the wrong
  character; the Hound has the same errors. Use them for recaps and for finding where something
  happened, not as the only source for a specific answer. Long chapters are summarized in
  sections of at most 2,500 words: in one pass, 3,500-word chapters lost their endings.
- **Chunks are longer than the embedding model reads.** `CHUNK_SIZE` is 400 words (~500
  tokens); all-MiniLM-L6-v2 reads the first 256 tokens and ignores the rest. Measured on Shadow
  Slave 1-64: 89% of chunks are cut, so vector search sees ~54% of the text (replacing a chunk's
  tail leaves its vector unchanged). The model answering still gets whole chunks. Changing it
  means re-ingesting; measure retrieval with the eval set first.
- **Shadow Slave's index predates the paragraph-spacing fix in `clean_html`**: its chunks glue
  paragraphs ("himself.After", ~6 per chunk) and each chapter's title to its first word
  ("Turf WarCaster"). The Hound's index is clean. Re-ingest Shadow Slave to fix (CPU only).
- Never pipe `scripts/start_vllm.sh` into `head` or similar: the closed pipe kills the script
  before `docker compose up` runs, and the model silently never starts.
- **Evaluation scores in code, not with an LLM judge.** Mistral Nemo graded "not revealed" as
  correct and passed invented answers. Fact answers use `key_terms`; leaks use names first
  mentioned after the reader's chapter; reveal patterns only flag answers for review (they
  overfit when written against graded answers). Check any scoring change with
  `run_eval.py --rejudge <results> --labels eval/labels/hound.json`.
- The answer prompt says "couldn't find it in the chapters the reader has read", never "the
  story has not revealed that yet": that was false whenever retrieval missed, and "yet" hints
  that a reveal is coming.
- Map-reduce over many chapters with Nemo has poor precision: it reports "relevant" notes for
  every block, then merges them into confident lists. Search first, then read what was found.

## Roadmap

Evaluation harness (RAGAS metrics + deterministic spoiler-leakage check), populated character
graph, hybrid search with reranking, SSE streaming, tracing via Langfuse/OpenTelemetry.

**Keep this file updated as work progresses** so a fresh session can resume from here.
