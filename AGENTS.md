# AGENTS.md

Instructions for coding agents (Codex and others) working in this repository.

**Read `CLAUDE.md` before changing anything.** It is the project's working notes, kept current
for every agent: the architecture, how to run and test the stack, and a list of gotchas that
each cost a bug once. This file repeats the rules that are expensive to break and says how to
share the repository with Claude Code, which also works here.

## Where the rest of the context lives

- **Why things are the way they are: `git log`.** Commit messages record the problem, what was
  measured, and what was tried and rejected. Read the history of a file before redesigning it.
- **Experiment results:** the docstrings of `eval/experiments/*.py`.
- **The detailed plan: `ROADMAP.md`,** local and gitignored, in the main checkout only. A git
  worktree is a fresh checkout of committed files, so it also lacks `.env`, `data/`,
  `eval/experiments/out/` and `frontend/node_modules` (run `npm ci` there before the frontend
  checks).

## Rules

- **Never run `docker compose down -v`.** The dev project's volumes hold the ingested novels;
  re-ingesting the largest takes 11 minutes, and backups live in `backups/`.
- **The GPUs are shared with other people.** Serve the model only with `scripts/start_vllm.sh`,
  which picks an idle GPU (never pipe it into `head`: that kills it before the model starts).
  Stop it with `docker compose stop vllm` as soon as you are done, and never leave it running
  unattended.
- **Commits carry no AI attribution:** no Co-authored-by or session trailers.
- **Schema changes go in a new file under `migrations/`,** applied by `scripts/migrate.py`.
- **Requirements are lockfiles:** edit `requirements.in`, then regenerate `requirements.txt`
  as described in `CLAUDE.md`.
- **`ROADMAP.md` is local and gitignored on purpose.** Never commit it.
- Add a technology only when it solves a problem the project actually has.

## Before you finish

Run what CI runs:

```bash
pytest                                        # unit, API, vector-store and database tests
ruff check services/ tests/ scripts/ eval/    # CI pins ruff 0.16.7
```

If you touched `frontend/`: `npx tsc --noEmit` and `npx eslint src` in that directory.

Passing tests are not enough for changes to retrieval, chunking or prompts: measure them on
the evaluation set (`eval/run_eval.py`, see the Evaluation section of `README.md`; it needs
the model served), and read the answers that changed, because the key-term scores have
misjudged answers in both directions.

## Working alongside Claude Code

- Work on your own branch or git worktree, and don't edit files another agent is changing at
  the same time.
- Never run `docker compose` from a worktree or branch checkout. The compose project name is
  fixed (`name: novel-companion-ai`), so it would rebuild the live stack from your unmerged
  code. Deploying is done from the main checkout after merging.
- The database tests share one test database (`novel_companion_test`): don't run them while
  another agent is.
- When you change how something works, update `CLAUDE.md` in the same commit, so the next
  agent of either kind starts from the truth.
