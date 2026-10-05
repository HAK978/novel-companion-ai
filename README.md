# Novel Companion AI

[![CI](https://github.com/HAK978/novel-companion-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/HAK978/novel-companion-ai/actions/workflows/ci.yml)

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
`chapter_number <= current_chapter` before searching, so the model never sees unread text.
That filter is enforced server-side, not left to the prompt.

Retrieval alone isn't enough for a well-known book, though: a model may already know how it
ends. Asked at chapter 5 of *The Hound of the Baskervilles* who "turns out to be the villain",
the model named the culprit from memory in 7 of 20 answers. The reader's position and a
no-outside-knowledge rule now go to the model as a system message, which brought that to
0 of 40 on the same questions.

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
- **Generation** — prompts and calls to the language model: any OpenAI-compatible endpoint,
  vLLM by default

## Stack

FastAPI · Celery · Redis · PostgreSQL 16 · ChromaDB · `all-MiniLM-L6-v2` embeddings (ONNX,
pinned in code) · vLLM serving Mistral Nemo 12B · Next.js + TypeScript + Tailwind ·
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

Claude Code picks it up from `.mcp.json` when started in this repository (the server needs
`pip install -r services/mcp/requirements.txt`). To use it from any directory:

```bash
claude mcp add novel-companion --scope user \
  --env GATEWAY_URL=http://localhost:8000 \
  -- python3 /path/to/services/mcp/server.py
```

## Running it

Needs Docker with Compose. The language model can be any OpenAI-compatible endpoint: by
default Compose serves Mistral Nemo with vLLM on an NVIDIA GPU, or set `LLM_BASE_URL`,
`LLM_MODEL` and `LLM_API_KEY` in `.env` to use a hosted model and skip the GPU.

```bash
cp .env.example .env
docker compose up -d                  # everything except the model server
docker compose --profile gpu up -d    # also serve the model with vLLM
```

The first start builds the images. The web UI is then at http://localhost:3000, and
`curl localhost:8000/health` reports on every service. Ports are published on localhost only.
Services come back on their own after a reboot, but the model server does not: on a shared
machine, `scripts/start_vllm.sh` starts it on an idle GPU and `docker compose stop vllm`
releases it.

Before any service starts, a one-shot `migrate` container applies pending `migrations/*.sql`
in order and records each in a `schema_migrations` table, so this always answers "is the
database up to date?":

```bash
docker compose run --rm migrate python scripts/migrate.py --status
```

Add a novel and ingest it. The ingestion worker reads novels from `data/novels` (or
`NOVEL_DATA_DIR` in `.env`), mounted at `/novels`, so source paths start there:

```bash
cp -r ~/my-novel data/novels/

curl -X POST localhost:8000/novels -H 'Content-Type: application/json' \
  -d '{"title": "My Novel", "source_type": "local_json"}'

curl -X POST localhost:8000/ingest/from-source -H 'Content-Type: application/json' \
  -d '{"novel_id": 1, "source_type": "local_json", "source_path": "/novels/my-novel"}'
```

Adapters handle directories of JSON chapter files and EPUB (`"source_type": "epub"` with a
path like `/novels/book.epub`). Ingestion runs through Celery; poll `/ingest/status/{task_id}`
for progress. Roughly a second per chapter.

Ask something:

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "Who is the protagonist travelling with?", "novel_id": 1, "current_chapter": 500}'
```

## Layout

```
services/       gateway, ingestion, retrieval, generation, mcp
migrations/     PostgreSQL schema
scripts/        database and embedding migrations, model server start
tests/          unit, API, database and end-to-end tests
frontend/       Next.js app
training/       dataset generation for fine-tuning
```

## Status

Working: ingestion at scale (tested on a 3,000-chapter web serial, and on a 15-chapter EPUB
with chapters three times as long), question answering, summarization, character recall,
progress tracking, MCP server, web UI. Novels are fully isolated: queries, summaries,
progress and deletion for one never touch another.

Next: an evaluation harness, request tracing, streaming responses, hybrid search with
reranking, and fine-tuning on a distilled dataset.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs in tiers:

- **Unit and API tests** use stubbed HTTP clients and need nothing running.
- **Vector-store tests** run an embedded ChromaDB in-process, so behaviors like re-ingestion
  are checked against the real library rather than a stub.
- **Database tests** (`-m db`) apply the migrations to a throwaway PostgreSQL database and run
  real SQL. They use the Compose Postgres locally and are skipped without one; CI sets
  `REQUIRE_DB=1`, so there an unreachable database fails the run instead of skipping it.
  The test database name must end in `_test`; anything else is refused.

- **End-to-end tests** (`-m integration`) drive the whole stack in Compose through the
  gateway. A stub model server stands in for vLLM and answers with the chapters it was
  shown, so the spoiler rule is checked across every service. CI runs them on each push.
  Locally (they create and delete their own novel):

```bash
docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --build --wait
RUN_INTEGRATION=1 NOVEL_DATA_DIR=data/novels pytest -m integration
docker compose up -d    # back to the real model
```

`NOVEL_DATA_DIR` must name the directory the stack mounts.

## Evaluation

`eval/` holds a question set for *The Hound of the Baskervilles* (public domain) and the
scripts that made and score it. `generate_set.py` drafts questions for any book with an LLM
and keeps only those it can check (answers must quote the chapter verbatim; invented names
must be absent from the whole book); the set was then reviewed by hand
(`sets/hound.review.json`). `run_eval.py` sends each question through `/query` at its
reading position and scores retrieval in code and answers with an LLM judge;
`--no-retrieval` asks the model directly instead, as a chatbot would be asked.

```bash
python eval/run_eval.py --set eval/sets/hound.jsonl --novel-id 4
```

## Data

No novel text is included here. Chapter content belongs to its authors; the ingestion
adapters read from a local directory you point them at.
