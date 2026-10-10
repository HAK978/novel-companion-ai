# Novel Companion AI — working notes

Spoiler-aware reading companion for long web novels, built as FastAPI microservices:
Celery/Redis ingestion into vector embeddings, a summary of every chapter written in the
background, and Q&A and recaps backed by RAG scoped to the reader's progress. Character
extraction exists but is off by default, so the PostgreSQL character graph is empty and
character questions are answered through search.

## Architecture

- **Gateway** (8000) — `services/gateway/main.py`. Single entry point, routes via
  httpx.AsyncClient, CORS enabled. Endpoints: /novels CRUD, /ingest, /ingest/from-source,
  /query, /characters/list, /characters/{name}, /summarize, /catch-me-up, /progress,
  /health (aggregated). Answers and range summaries go through `_generate_checked`, which
  has retrieval take out names the reader has not reached; /query reports `spoiler_check`
  ("withheld" or "clean"), never the names. Asking the model to rewrite without them was
  tried first and dropped: it kept the name in 5 of 12 tries and complying rewrites got
  worse. /summarize and /catch-me-up read the stored summary of every chapter in the range,
  in order, when all exist and the range is at most 60 chapters (`_chapter_summaries`,
  `RECAP_FROM_SUMMARIES_MAX`); otherwise they search. /query sends recent-events questions
  ("recently", "catch me up", "the last N chapters": `_recent_range`) to the summaries of the
  last 10 (or N) chapters: search has no sense of time, and at chapter 1291 it offered
  chapters 527 and 398 as "the last few".
- **Ingestion** (8001) — `services/ingestion/`. Celery tasks (`tasks.py`, core logic in
  `_process_chapter()`): clean HTML → chunk (~400 words, sentence-boundary, `chunking.py`)
  → split each chunk into windows of ≤170 words (`split_windows`, lossless) → embed the windows
  (all-MiniLM-L6-v2, Chroma's ONNX build; bulk ingests embed 32 chapters per call in
  `_ingest_chapters()`) → store the windows in ChromaDB (`chunk_index`, `window_index`) →
  the chapter's words in `book_words` (where each word first appears, and first appears in
  lowercase; `words.py`) → chapter record in
  PostgreSQL → optional entity extraction via generation service. Source adapters in
  `adapters/` (local_json handles nested dirs via rglob, epub). Chapter summaries: the
  `summary-worker` (Celery queue `summaries`) writes one per chapter through generation's
  `/chapter-summary`, up to the furthest reader + `SUMMARY_LOOKAHEAD` (100), newest first back
  from the reader, then ahead. Scheduled as soon as an ingest has indexed the reader's chapter,
  again when it ends, and on every progress update (`POST /summaries`, `GET /summaries/{novel_id}`).
- **Retrieval** (8002) — `services/retrieval/`. ChromaDB similarity search over windows that
  returns the whole chunks they belong to, each once, ranked by its best window
  (`search.py`); spoiler filter `chapter_number <= current_chapter`, optional `min_chapter`
  floor, adaptive n_results, character lookup from PostgreSQL. `/withhold-unread-names`:
  takes out of a text the names the book first uses after the reader's chapter (words it
  never writes in lowercase, from `book_words`), ignoring the reader's own words. `names.py`:
  from a longer name when a name is left ("his wife, Beryl Garcia" → "his wife, Beryl"),
  otherwise the whole sentence.
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
chapter_summaries (one per chapter), range_summaries (cache of /summarize answers),
book_words (where each word first appears), reading_progress,
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
  A novel ingested before `book_words` existed needs `rebuild_word_index(N)` the same way
  (from its indexed chapters; ~80 s for Shadow Slave), then a recheck.
- **Bulk ingest:** POST /ingest/from-source with `extract_entities: false` (default) and a
  `source_path` under `/novels` (host `NOVEL_DATA_DIR`). ~0.22 s/chapter: Shadow Slave's 3,026
  chapters in 11 min (whole chunks before windows took ~0.12 s; embedding one chapter per call,
  ~0.18 s). Chapters are searchable as they land; the task's progress meta has
  `searchable_up_to`, which the web UI shows. Extraction ~20s/chapter.
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
- Range queries (`/summarize`, `/catch-me-up`) that fall back to search pass `min_chapter`;
  context chunks are labelled `[Chapter N: title]`. Their cache (`range_summaries`) is keyed
  by (novel_id, start, end) and outlives changes to how recaps are made: clear the table
  after changing that, or old recaps keep being served.
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
- Summaries record names the book has not used by their chapter, or never uses
  (`unverified_names`, checked against `book_words`): additions from the model's memory of
  the book. Heading words ("Chapter 12 Summary (Part 1 of 2)") and sentence-opening words are
  ignored. Shadow Slave 1-1391: 6 of 1,390 flagged, one real leak (chapter 33's summary used
  "Underworld", first in the book at chapter 250), four words the book never uses
  ("Cassandra", "Fawkes", "Sightless", "Transcender"), and "Antarctic" against the text's
  "Antarctica". Hound: 3 of 15, an invented name (chapter 11 calls Mr. Frankland "Captain John
  Sebastian Morland Frankland"), the book's title in chapter 1's summary, "Inspector" for
  Lestrade. (Checked against each chapter alone it was 33 and 6: names from earlier chapters.)
- **Name check limits.** `book_words` keys words by `words.stem`, identical copies in
  ingestion and retrieval (tests/test_words.py fails on drift); names ending in s reduce too
  ("Nephis" → "nephi"), harmlessly, since both sides reduce alike. Measured: on the Hound's
  chapter-14 question the model added "Garcia" in 7 of 24 answers; all were withheld, and 141
  evaluated answers named no one the reader had not met (3 had, before). Only names are checked
  (words the book never writes in lowercase): a hint without a name gets through (at chapter
  11, "the brother, Rodger" hints at Stapleton's identity; Rodger is named in chapter 2).
- **Chapter summaries get who-did-what wrong about 1 time in 5.** Read against their chapters:
  Shadow Slave 19 of 25 accurate, the 5 wrong ones mostly crediting an event to the wrong
  character; the Hound has the same errors. Use them for recaps and for finding where something
  happened, not as the only source for a specific answer. Long chapters are summarized in
  sections of at most 2,500 words: in one pass, 3,500-word chapters lost their endings.
- **Chunks are searched through windows.** all-MiniLM-L6-v2 reads the first 256 tokens of a
  text and ignores the rest: 400-word chunks (~500 tokens) left ~46% of the book invisible to
  search. Each chunk is stored as windows of ≤170 words (the most that always fit, measured);
  a search ranks windows and returns whole chunks. Measured (eval/experiments/chunk_windows.py):
  the answer was retrieved for 17 of 25 Hound fact questions instead of 12, and 140 of 262
  Shadow Slave questions instead of 110; hand-graded answers 16.8 → 20.5 of 30, fewer wrong.
  170-word chunks returned as 12 passages retrieved as well but answered worse (fragments).
  Live after re-ingesting: Hound 16 of 25, Shadow Slave 140 of 262. `WINDOW_SIZE` must stay
  within 256 tokens; changing either size means re-ingesting. Collections stored before
  windows still work (each chunk counts as one window). `split_windows` must lose nothing
  (`_chapter_text` and search rebuild chunks from windows): a merge bug once dropped text
  only when a chunk's last window was under 50 characters, so test every length.
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

Next, in order: hybrid keyword + vector search (names embed poorly, and BM25 beat vector
search on a name-heavy question); an unknown-person check from `book_words` (asked about
someone who never appears, the app still answers); recaps over 60 chapters from summaries of
summaries; the character graph through schema-constrained extraction; streaming responses
and tracing. The detailed plan is `ROADMAP.md`, local and gitignored.

`AGENTS.md` points other coding agents (Codex) to this file and repeats its hard rules; keep
the two in step.

**Keep this file updated as work progresses** so a fresh session can resume from here.
