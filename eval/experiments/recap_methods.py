"""Three ways to answer questions that span a long stretch of a book.

  app        what the app does today: /summarize for a range recap, /query otherwise
             (a handful of retrieved chunks)
  summaries  every chapter summarized once, as ingestion could do it; at question time the
             model reads those summaries
  full       every chapter read at question time

Both of the last two read with the question in mind and in pieces that fit the model's
context (map), then combine their notes (reduce). The only difference is what they read:
summaries written in advance, or the full text. Everything stays at or before the reader's
chapter, and the same model and prompts are used throughout.

    python eval/experiments/recap_methods.py --source ~/novel-data/shadow-slave \
        --novel-id 3 --start 992 --end 1291

Outputs (book-derived text, kept out of git) go to eval/experiments/out/.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "ingestion"))
from adapters import get_adapter  # noqa: E402
from chunking import clean_html  # noqa: E402

LLM = "http://localhost:8004"
MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
GATEWAY = "http://localhost:8000"
OUT = ROOT / "eval" / "experiments" / "out"
BLOCK_TOKENS = 10_000  # text per map call; the server allows 16,384 including the reply

SYSTEM = ("You are a spoiler-free reading companion. The reader has read up to chapter {end} "
          "of this book and nothing beyond it. Use only the text you are given. You may "
          "recognize this book: never use that knowledge.")

SUMMARY_PROMPT = """Summarize chapter {n} of a novel in at most 120 words: the key events, who \
was involved, and anything that changed or was revealed. Use only this text.

{text}"""

MAP_PROMPT = """A reader asked: "{question}"

Below are {kind} {first} to {last} of the novel. Write notes on everything in them that helps \
answer the question: events, people, decisions, revelations, in order, with chapter numbers \
in brackets. Use only this text. At most 250 words. If nothing here is relevant, reply only \
NOTHING RELEVANT.

{text}"""

CONDENSE_PROMPT = """A reader asked: "{question}"

Below are notes taken while reading chapters {first} to {last} in order. Merge them into one \
set of notes that keeps every point relevant to the question, in story order, with chapter \
numbers. At most 400 words.

{text}"""

ANSWER_PROMPT = """A reader asked: "{question}"

Below are notes taken while reading chapters {first} to {last} in order. Answer the question \
from these notes only, in story order, citing chapter numbers. At most 350 words.

{text}"""


class Meter:
    def __init__(self):
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def add(self, usage):
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)


def chat(prompt: str, end: int, meter: Meter, max_tokens: int) -> str:
    resp = httpx.post(f"{LLM}/v1/chat/completions", timeout=900, json={
        "model": MODEL, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": SYSTEM.format(end=end)},
                     {"role": "user", "content": prompt}],
    })
    resp.raise_for_status()
    body = resp.json()
    meter.add(body.get("usage", {}))
    return body["choices"][0]["message"]["content"].strip()


def count_tokens(text: str) -> int:
    resp = httpx.post(f"{LLM}/tokenize", timeout=60, json={"model": MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["count"]


def blocks(docs: list[tuple[int, str]], tokens: dict[int, int]) -> list[list[tuple[int, str]]]:
    """Consecutive documents grouped so each group fits in one call."""
    out, current, size = [], [], 0
    for n, text in docs:
        if current and size + tokens[n] > BLOCK_TOKENS:
            out.append(current)
            current, size = [], 0
        current.append((n, text))
        size += tokens[n]
    return [*out, current] if current else out


def parallel(fn, items, workers=32):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def map_reduce(question: str, docs, tokens, kind: str, end: int, meter: Meter) -> str:
    groups = blocks(docs, tokens)

    def read(group):
        text = "\n\n".join(f"[Chapter {n}]\n{t}" for n, t in group)
        notes = chat(MAP_PROMPT.format(question=question, kind=kind, first=group[0][0],
                                       last=group[-1][0], text=text), end, meter, 350)
        return group[0][0], group[-1][0], notes

    notes = [n for n in parallel(read, groups) if "NOTHING RELEVANT" not in n[2][:40]]
    print(f"    {len(groups)} reads, {len(notes)} with relevant notes", file=sys.stderr)
    # combine notes until they fit in one call, keeping story order
    while True:
        joined = "\n\n".join(f"[Chapters {a}-{b}]\n{t}" for a, b, t in notes)
        if len(notes) <= 1 or count_tokens(joined) <= BLOCK_TOKENS:
            break
        size = max(2, len(notes) // max(2, count_tokens(joined) // (BLOCK_TOKENS - 2000) + 1))
        batches = [notes[i:i + size] for i in range(0, len(notes), size)]

        def condense(batch):
            text = "\n\n".join(f"[Chapters {a}-{b}]\n{t}" for a, b, t in batch)
            return batch[0][0], batch[-1][1], chat(CONDENSE_PROMPT.format(
                question=question, first=batch[0][0], last=batch[-1][1], text=text), end, meter, 600)

        notes = parallel(condense, batches)
        print(f"    condensed to {len(notes)} sets of notes", file=sys.stderr)
    if not notes:
        return "Nothing relevant was found."
    return chat(ANSWER_PROMPT.format(question=question, first=docs[0][0], last=docs[-1][0],
                                     text=joined), end, meter, 700)


def first_appearances(chapters: dict[int, str]) -> dict[str, int]:
    first = {}
    for n in sorted(chapters):
        for w in set(re.findall(r"\b[A-Z][a-z]{2,}\b", chapters[n])):
            first.setdefault(w, n)
    return first


def name_flags(answer: str, first: dict[str, int], end: int) -> dict:
    names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", answer or ""))
    return {"after_reader": sorted(w for w in names if first.get(w, 0) > end),
            "not_in_book": sorted(w for w in names if w not in first)}


def app_answer(question: str, args, recap: bool) -> dict:
    started = time.perf_counter()
    if recap:
        resp = httpx.post(f"{GATEWAY}/summarize", timeout=600, json={
            "novel_id": args.novel_id, "start_chapter": args.start, "end_chapter": args.end,
            "current_chapter": args.end})
        body = resp.json()
        answer, extra = body.get("summary") or body.get("error"), {"cached": body.get("cached")}
    else:
        resp = httpx.post(f"{GATEWAY}/query", timeout=600, json={
            "query": question, "novel_id": args.novel_id, "current_chapter": args.end})
        body = resp.json()
        answer = body.get("answer")
        extra = {"source_chapters": sorted({s["chapter_number"] for s in body.get("sources", [])})}
    return {"answer": answer, "seconds": round(time.perf_counter() - started, 1), **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-type", default="local_json")
    ap.add_argument("--novel-id", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--questions", required=True, help="JSON file: [{id, question, recap}]")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    every = {c["number"]: clean_html(c["content"])
             for c in get_adapter(args.source_type, args.source).fetch_all()}
    first = first_appearances(every)
    docs = [(n, every[n]) for n in range(args.start, args.end + 1) if every.get(n)]
    tokens = dict(zip([n for n, _ in docs], parallel(count_tokens, [t for _, t in docs]), strict=True))
    print(f"{len(docs)} chapters, {sum(tokens.values()):,} tokens", file=sys.stderr)

    # One-time cost of the summaries method: what ingestion would pay once per chapter
    cache = OUT / f"summaries-{args.start}-{args.end}.json"
    build = Meter()
    started = time.perf_counter()
    if cache.exists():
        summaries = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        build_seconds = None
    else:
        texts = parallel(lambda d: chat(SUMMARY_PROMPT.format(n=d[0], text=d[1]), args.end,
                                        build, 220), docs)
        summaries = {n: s for (n, _), s in zip(docs, texts, strict=True)}
        cache.write_text(json.dumps(summaries, indent=1))
        build_seconds = round(time.perf_counter() - started, 1)
    summary_docs = [(n, summaries[n]) for n, _ in docs]
    summary_tokens = dict(zip([n for n, _ in summary_docs],
                              parallel(count_tokens, [s for _, s in summary_docs]), strict=True))
    print(f"summaries: {sum(summary_tokens.values()):,} tokens, built in {build_seconds}s",
          file=sys.stderr)

    results = []
    for q in json.loads(Path(args.questions).read_text()):
        print(f"{q['id']}: {q['question']}", file=sys.stderr)
        row = {"id": q["id"], "question": q["question"], "methods": {}}
        row["methods"]["app"] = app_answer(q["question"], args, q.get("recap", False))
        for name, material, toks, kind in [("summaries", summary_docs, summary_tokens,
                                            "summaries of chapters"),
                                           ("full", docs, tokens, "chapters")]:
            meter, started = Meter(), time.perf_counter()
            answer = map_reduce(q["question"], material, toks, kind, args.end, meter)
            row["methods"][name] = {
                "answer": answer, "seconds": round(time.perf_counter() - started, 1),
                "calls": meter.calls, "prompt_tokens": meter.prompt_tokens,
                "completion_tokens": meter.completion_tokens}
            print(f"  {name}: {row['methods'][name]['seconds']}s, {meter.calls} calls",
                  file=sys.stderr)
        for m in row["methods"].values():
            m["names"] = name_flags(m["answer"], first, args.end)
        results.append(row)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT / f"recap-{args.start}-{args.end}-{stamp}.json"
    out.write_text(json.dumps({
        "range": [args.start, args.end], "chapter_tokens": sum(tokens.values()),
        "summary_build": {"seconds": build_seconds, "calls": build.calls,
                          "prompt_tokens": build.prompt_tokens,
                          "completion_tokens": build.completion_tokens,
                          "summary_tokens": sum(summary_tokens.values())},
        "results": results}, indent=2))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
