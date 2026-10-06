"""Chunks longer than the embedding model reads: compare two fixes against today's chunks.

The pinned model (all-MiniLM-L6-v2) reads the first 256 tokens of a text and ignores the rest.
Chunks are 400 words (~500 tokens), so vector search sees about half of the book. 170 words is
the largest chunk size that always fits, measured on the Hound and Shadow Slave 1-300.

Variants, each built from the source books in an in-memory Chroma with the pinned model:
  base     400-word chunks, as ingested today; 5 passages (10 for broad questions)
  small    170-word chunks, as many as fill about the same text: 12 (24)
  small-5  170-word chunks, 5 (10): as many passages, less than half the text
  windows  170-word windows inside today's 400-word chunks: the search ranks windows, and the
           model gets the chunks they belong to, 5 (10) distinct ones
  neighbors  170-word chunks; each hit comes with the chunks either side of it in its chapter,
           so the matching text sits mid-passage: 4 (8) passages of up to 3 chunks

Retrieval, with no language model:
  hound         run_eval's answer_in_context over the set's fact and recent questions
  shadow slave  questions from training/data_v2/qa_pairs.jsonl whose passage opens a chapter
                up to --max-chapter, asked at that chapter. A hit: a returned passage covers at
                least half of the passage's sentence that shares the most words with the
                reference answer
Answers (--runs N, with the model served): the Hound set through generation's /generate, sent
as the gateway sends it, N times per variant, scored with run_eval.score.

Results (2026-10-06; 'base' matched the live app's retrieval on all 47 questions):
  answer retrieved   Hound fact (of 25)   Shadow Slave (of 262)
  base               12                   110
  small              14                   141
  small-5            13                   111
  windows            17                   140
  neighbors          16                   133
On Shadow Slave, base found 12 that windows missed and windows 42 that base missed (McNemar
p < 0.001). Answers, hand-graded over the 30 fact and recent questions (right 1, partly
right 0.5; mean of 3 runs): base 16.8, small 19.7, windows 20.5, neighbors 21.0; wrong
answers 5.3, 5.7, 3.7, 3.7. Key terms misled both ways: two "correct" base answers were
wrong (they quoted the right name while giving another), and partial answers counted as
wrong. Small and neighbors leaked a later name ("Beryl Garcia" at chapter 14) from the
model's memory in some runs; base and windows never did. Windows shipped: best retrieval,
and the model still gets the same five 400-word passages.

Run in the ingestion-worker container, which has the pinned model, the chunker and the books:
    docker compose cp eval ingestion-worker:/tmp/eval
    docker compose cp training/data_v2/qa_pairs.jsonl ingestion-worker:/tmp/qa_pairs.jsonl
    docker compose exec -e PYTHONPATH=/app ingestion-worker \\
        python /tmp/eval/experiments/chunk_windows.py --qa /tmp/qa_pairs.jsonl --runs 3
"""

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import chromadb
import httpx

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import run_eval  # noqa: E402
import tasks  # noqa: E402  (the ingestion service's module, on PYTHONPATH)
from adapters import get_adapter  # noqa: E402
from chunking import chunk_text, clean_html  # noqa: E402
from search_then_read import STOPWORDS  # noqa: E402
from vector_store import open_collection  # noqa: E402

BASE_WORDS, SMALL_WORDS = 400, 170
# passages per question, and per broad question
VARIANTS = {"base": (5, 10), "small": (12, 24), "small-5": (5, 10), "windows": (5, 10),
            "neighbors": (4, 8)}
INDEX_OF = {"base": "base", "small": "small", "small-5": "small", "windows": "windows",
            "neighbors": "small"}
# the retrieval service's test for broad questions (services/retrieval/search.py)
COMPLEX = re.compile(
    r"\b(?:summari[sz]e|explain|arcs?|stor(?:y|ies)|what happened|tell me about)\b", re.I)


def strip(text: str) -> str:
    # letters and digits only: the question pool's passages came from an older cleaner, which
    # treated spacing, quotes and dashes differently
    return re.sub(r"[^A-Za-z0-9]", "", text)


class Book:
    def __init__(self, source_type: str, source: str, max_chapter: int | None = None):
        chapters = get_adapter(source_type, source).fetch_all()
        self.chapters = [(c["number"], c.get("title") or f"Chapter {c['number']}",
                          clean_html(c["content"])) for c in chapters
                         if max_chapter is None or c["number"] <= max_chapter]
        self.stripped = {n: strip(text) for n, _, text in self.chapters}

    def span(self, chapter: int, text: str) -> tuple[int, int] | None:
        s = strip(text)
        pos = self.stripped[chapter].find(s)
        return (pos, pos + len(s)) if pos >= 0 else None


class Index:
    """One way of cutting a book into searchable units, in an in-memory collection."""

    def __init__(self, client, name: str, book: Book, kind: str):
        self.kind, self.parents, self.units = kind, {}, {}
        ids, docs, metas = [], [], []
        for number, title, text in book.chapters:
            meta = {"chapter_number": number, "chapter_title": title}
            if kind == "small":
                units = [(i, c) for i, c in enumerate(chunk_text(text, SMALL_WORDS))]
            else:
                parents = chunk_text(text, BASE_WORDS)
                if kind == "base":
                    units = list(enumerate(parents))
                else:  # windows: units point at the chunk they came from
                    units = []
                    for i, parent in enumerate(parents):
                        self.parents[(number, i)] = parent
                        units += [(i, w) for w in chunk_text(parent, SMALL_WORDS)]
            for j, (parent, unit) in enumerate(units):
                self.units[(number, j)] = unit
                ids.append(f"{number}-{j}")
                docs.append(unit)
                metas.append({**meta, "parent": parent})
        self.col = open_collection(client, name)
        vectors = []
        for start in range(0, len(docs), 512):
            vectors += tasks._embed(docs[start:start + 512])
        for start in range(0, len(docs), 1000):
            end = start + 1000
            self.col.add(ids=ids[start:end], documents=docs[start:end],
                         metadatas=metas[start:end], embeddings=vectors[start:end])
        self.size = len(docs)

    def around(self, query: str, chapter: int, k: int) -> list[dict]:
        """Each hit with the units either side of it, skipping units already given."""
        res = self.col.query(query_texts=[query], n_results=k * 6,
                             where={"chapter_number": {"$lte": chapter}}, include=["metadatas"])
        out, given = [], set()
        for m in res["metadatas"][0]:
            c, i = m["chapter_number"], m["parent"]
            if (c, i) in given:
                continue
            run = [j for j in (i - 1, i, i + 1) if (c, j) in self.units and (c, j) not in given]
            given.update((c, j) for j in run)
            out.append({"chapter_number": c, "chapter_title": m["chapter_title"],
                        "text": " ".join(self.units[(c, j)] for j in run)})
            if len(out) == k:
                break
        return out

    def search(self, query: str, chapter: int, k: int) -> list[dict]:
        where = {"chapter_number": {"$lte": chapter}}
        if self.kind != "windows":
            res = self.col.query(query_texts=[query], n_results=k, where=where,
                                 include=["documents", "metadatas"])
            return [{"chapter_number": m["chapter_number"], "chapter_title": m["chapter_title"],
                     "text": d} for d, m in zip(res["documents"][0], res["metadatas"][0], strict=True)]
        res = self.col.query(query_texts=[query], n_results=k * 8, where=where,
                             include=["metadatas"])
        out, seen = [], set()
        for m in res["metadatas"][0]:
            key = (m["chapter_number"], m["parent"])
            if key not in seen:
                seen.add(key)
                out.append({"chapter_number": key[0], "chapter_title": m["chapter_title"],
                            "text": self.parents[key]})
            if len(out) == k:
                break
        return out


def passages(indexes, variant: str, query: str, chapter: int) -> list[dict]:
    simple, broad = VARIANTS[variant]
    index, k = indexes[INDEX_OF[variant]], broad if COMPLEX.search(query) else simple
    return index.around(query, chapter, k) if variant == "neighbors" else index.search(query, chapter, k)


def build(book: Book, label: str) -> dict:
    client = chromadb.EphemeralClient()
    indexes = {}
    for kind in ("base", "small", "windows"):
        start = time.time()
        indexes[kind] = Index(client, f"{label}-{kind}", book, kind)
        print(f"{label} {kind}: {indexes[kind].size} units in {time.time() - start:.0f} s",
              flush=True)
    return indexes


def shadow_slave_items(path: str, book: Book) -> tuple[list[dict], int]:
    items, unplaced = [], 0
    for line in open(path):
        row = json.loads(line)
        passage = row["input"].split("Passage:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        question = row["input"].rsplit("Question: ", 1)[1].split("\n")[0].strip()
        m = re.match(r"Chapter (\d+)", passage)
        if not m or int(m.group(1)) not in book.stripped:
            continue
        chapter = int(m.group(1))
        answer_words = {w for w in re.findall(r"[a-z']+", row["output"].lower())
                        if len(w) > 2 and w not in STOPWORDS}
        # sentences; the old cleaner glued some together ("himself.After")
        sentences = re.split(r"(?<=[.!?])[\"'\u201d\u2019]?\s*(?=[A-Z\u201c\"])", passage)
        scored = [(len(answer_words & set(re.findall(r"[a-z']+", s.lower()))), s)
                  for s in sentences[1:]]  # the first carries the chapter title
        best, sentence = max(scored, default=(0, ""), key=lambda x: x[0])
        span = book.span(chapter, sentence) if best >= 3 else None
        if span is None:
            unplaced += 1
            continue
        items.append({"question": question, "chapter": chapter, "evidence": span,
                      "sentence": sentence, "answer": row["output"]})
    return items, unplaced


def covers(book: Book, results: list[dict], chapter: int, evidence: tuple[int, int]) -> bool:
    for r in results:
        if r["chapter_number"] != chapter:
            continue
        span = book.span(chapter, r["text"])
        if span:
            overlap = min(span[1], evidence[1]) - max(span[0], evidence[0])
            if overlap >= (evidence[1] - evidence[0]) / 2:
                return True
    return False


def generate(args, item, results) -> tuple[str, float]:
    # exactly as the gateway's /query builds the request
    context = [f"[Chapter {r['chapter_number']}: {r.get('chapter_title', '')}]\n{r['text']}"
               for r in results]
    start = time.time()
    resp = httpx.post(f"{args.generation}/generate", timeout=300, json={
        "query": item["question"], "context_chunks": context, "conversation_context": "",
        "current_chapter": item["current_chapter"]})
    resp.raise_for_status()
    return resp.json().get("answer") or "", round(time.time() - start, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(HERE.parent / "sets" / "hound.jsonl"))
    ap.add_argument("--hound", default="/novels/hound-of-the-baskervilles/hound.epub")
    ap.add_argument("--shadow-slave", default="/novels/shadow-slave")
    ap.add_argument("--qa", help="training/data_v2/qa_pairs.jsonl, for the Shadow Slave check")
    ap.add_argument("--max-chapter", type=int, default=600)
    ap.add_argument("--retrieval", default="http://retrieval:8002",
                    help="the live service, to check 'base' matches what the app retrieves")
    ap.add_argument("--live-novel", type=int, default=4)
    ap.add_argument("--generation", default="http://generation:8003")
    ap.add_argument("--runs", type=int, default=0, help="answer runs per variant (needs the model)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--out", default="/tmp")
    args = ap.parse_args()

    report = {"variants": VARIANTS, "words": {"base": BASE_WORDS, "small": SMALL_WORDS}}
    items = [json.loads(line) for line in open(args.set)]
    hound = Book("epub", args.hound)
    indexes = build(hound, "hound")
    found = {v: [passages(indexes, v, i["question"], i["current_chapter"]) for i in items]
             for v in VARIANTS}

    # 'base' should be what the app retrieves today
    same = 0
    for item, mine in zip(items, found["base"], strict=True):
        live = httpx.post(f"{args.retrieval}/search", timeout=60, json={
            "query": item["question"], "current_chapter": item["current_chapter"],
            "n_results": 5, "collection_name": f"novel_{args.live_novel}"}).json()["results"]
        same += [r["text"] for r in live] == [r["text"] for r in mine]
    report["base_matches_live_app"] = f"{same} of {len(items)} questions"
    print("base matches the live app on", report["base_matches_live_app"], flush=True)

    report["hound"] = {}
    for variant, results in found.items():
        scored = [i for i in items if i["category"] in ("fact", "recent")]
        hits = {cat: sum(run_eval.in_context(i, r) for i, r in zip(items, results, strict=True)
                         if i["category"] == cat) for cat in ("fact", "recent")}
        words = statistics.mean(sum(len(r["text"].split()) for r in res) for res in results)
        report["hound"][variant] = {
            "answer_in_context": {c: f"{hits[c]} of {sum(i['category'] == c for i in scored)}"
                                  for c in hits},
            "context_words": round(words)}
        print(f"hound {variant}: {report['hound'][variant]}", flush=True)

    if args.qa:
        ss = Book("local_json", args.shadow_slave, args.max_chapter)
        qa, unplaced = shadow_slave_items(args.qa, ss)
        print(f"shadow slave: {len(qa)} questions placed, {unplaced} skipped", flush=True)
        ss_indexes = build(ss, "ss")
        report["shadow_slave"] = {"questions": len(qa), "skipped": unplaced}
        per_question = {}
        for variant in VARIANTS:
            hits = [covers(ss, passages(ss_indexes, variant, q["question"], q["chapter"]),
                           q["chapter"], q["evidence"]) for q in qa]
            per_question[variant] = hits
            report["shadow_slave"][variant] = f"{sum(hits)} of {len(qa)}"
            print(f"shadow slave {variant}: {sum(hits)} of {len(qa)}", flush=True)
        # where two variants disagree, which one found it
        for a, b in (("base", "small"), ("base", "windows"), ("base", "neighbors"),
                     ("windows", "neighbors")):
            only_a = sum(x and not y for x, y in zip(per_question[a], per_question[b], strict=True))
            only_b = sum(y and not x for x, y in zip(per_question[a], per_question[b], strict=True))
            report["shadow_slave"][f"{a} vs {b}"] = f"only {a} {only_a}, only {b} {only_b}"

    if args.runs:
        names = run_eval.name_index(argparse.Namespace(source=args.hound, source_type="epub"))
        reveals_path = Path(args.set).with_suffix(".reveals.json")
        reveals = json.loads(reveals_path.read_text()) if reveals_path.exists() else []
        score_args = argparse.Namespace(judge=False)
        report["answers"], report["rows"] = {}, {}
        for variant in VARIANTS:
            if variant == "small-5":
                continue
            report["answers"][variant], report["rows"][variant] = [], []
            for run in range(args.runs):
                def answer(pair):
                    item, results = pair
                    text, seconds = generate(args, item, results)
                    row = run_eval.score(score_args, item, text, results, reveals, names)
                    return {**row, "seconds": seconds}
                with ThreadPoolExecutor(args.parallel) as pool:
                    rows = list(pool.map(answer, zip(items, found[variant], strict=True)))
                summary = run_eval.summarize(rows)
                report["answers"][variant].append(summary)
                report["rows"][variant].append(rows)
                print(f"answers {variant} run {run + 1}: {json.dumps(summary)}", flush=True)

    out = Path(args.out) / f"chunk-windows-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
