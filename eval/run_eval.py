"""Run an evaluation set through the live app and score it.

    python eval/run_eval.py --set eval/sets/hound.jsonl --novel-id 4 \
        --source ~/novel-data/hound-of-the-baskervilles/hound.epub --source-type epub

Each question goes through the gateway's /query at its reading position, as a reader's would.
Scoring is done in code wherever it can be, because an LLM judge proved unreliable:
  leak           every answer, whatever its category: it names someone or something the book
                 first mentions after the reader's chapter (needs --source)
  leak review    it matches a pattern for a reveal after the reader's chapter
                 (<set>.reveals.json): a flag for a person to check, not a verdict
  passages safe  no retrieved passage comes from after the reader's chapter
  fact, recent   CORRECT if the answer contains one of the item's key terms, otherwise
                 DECLINED if it says it cannot answer, otherwise WRONG; and whether an accepted
                 answer or the evidence was in the retrieved text
  unanswerable   whether the answer says it cannot find the person (strict: an answer that
                 neither says so nor invents anything still fails)

Measured against hand grades (eval/labels/hound.json): key terms agreed on 59 of the 60
answers they were written against, but on fresh answers 5 of 64 passes were wrong or only
partly right, because a wrong answer can still contain the right name ("Selden is actually
Sherlock Holmes in disguise"), so spot-check what passes. The name check raised no false
alarm in 188 fresh answers; the reveal patterns were right on only 2 of 8 flags (they overfit
the answers they were written against), so they only flag answers for review. Mistral Nemo as
a judge agreed on leaks 30/47 and on invented people 4/10, so --judge (an LLM cross-check) is
off by default.

--rejudge RESULTS re-scores stored answers without calling the app (--in-place rewrites that
file, keeping stored verdicts in step with the scoring); --labels reports agreement with hand
grades. --no-retrieval asks the model directly, as a chatbot would be
asked. The app answers at temperature 0.3, so runs differ slightly.
Results go to eval/results/<set>[-no-retrieval][-rejudge]-<timestamp>.json.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

DECLINES = re.compile(
    r"not (been )?(explicitly )?(stated|mentioned|revealed|given|provided|specified|identified)|"
    r"has not revealed|hasn't revealed|(does|do)(n't| not) (say|mention|specify)|"
    r"(couldn't|could not|can't|cannot) (find|be determined)|no mention of|"
    r"not possible to (determine|say|tell)", re.I)
FLAGS_UNKNOWN = re.compile(
    r"not (a )?character|does(n't| not) appear|is(n't| not) mentioned|never mentioned|"
    r"no (character|one|person) (named|called)|(couldn't|could not|can't|cannot) find|"
    r"no information (about|on)", re.I)

# Optional LLM judge, reasoning before the verdict
VERDICT = "First give one sentence of reasoning. Then on a new line write only the verdict, one of: "
JUDGES = {
    "correct": """Grade an answer to a question about a novel.

Question: {question}
Acceptable answers (any one is enough): {accepted}
Answer to grade: {answer}

CORRECT: it gives an acceptable answer; extra detail is fine unless it contradicts one.
PARTIAL: it gets part of an acceptable answer, or is vaguer than one.
WRONG: it gives a different answer.
DECLINED: it gives no answer, for example it says the information is not stated or not revealed.
{verdict}CORRECT, PARTIAL, WRONG or DECLINED.""",
    "leak": """A reader has read up to chapter {chapter} of a novel. These facts are revealed only \
later in the book:
{facts}

Here is an answer the reader was shown:
{answer}

Does the answer disclose, confirm or strongly hint at any of those facts? A guess that matches one
counts. Ignore anything else the answer gets wrong.
{verdict}LEAK or SAFE.""",
    "abstain": """{name} does not appear anywhere in the novel. A reader asked: {question}

Answer: {answer}

INVENTED: the answer tells the reader anything about {name}, even by giving them another
character's actions or traits, or while hedging.
ABSTAINED: it only says it cannot find {name} or does not know.
{verdict}ABSTAINED or INVENTED.""",
}
VERDICTS = {"correct": ("CORRECT", "PARTIAL", "WRONG", "DECLINED"), "leak": ("LEAK", "SAFE"),
            "abstain": ("ABSTAINED", "INVENTED")}

NO_RETRIEVAL_SYSTEM = """The user is reading {title} and has reached chapter {chapter}. Answer
their question from your knowledge of the book. Do not reveal anything that happens after
chapter {chapter}. If you do not know, say so."""


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def judge(args, kind: str, **fields) -> str:
    resp = httpx.post(f"{args.judge_url}/chat/completions", timeout=120, json={
        "model": args.judge_model, "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user",
                      "content": JUDGES[kind].format(verdict=VERDICT, **fields)}],
    })
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].upper()
    found = re.findall(r"\b(" + "|".join(VERDICTS[kind]) + r")\b", text)
    return found[-1] if found else ""


def ask_app(args, item) -> dict:
    resp = httpx.post(f"{args.gateway}/query", timeout=300, json={
        "query": item["question"], "novel_id": args.novel_id,
        "current_chapter": item["current_chapter"],
    })
    resp.raise_for_status()
    body = resp.json()
    return {"answer": body.get("answer") or "",
            "sources": [{"chapter_number": s["chapter_number"], "text": s["text"]}
                        for s in body.get("sources", [])]}


def ask_model_directly(args, item) -> dict:
    resp = httpx.post(f"{args.judge_url}/chat/completions", timeout=300, json={
        "model": args.judge_model, "temperature": 0, "max_tokens": 600,
        "messages": [
            {"role": "system", "content": NO_RETRIEVAL_SYSTEM.format(
                title=args.title, chapter=item["current_chapter"])},
            {"role": "user", "content": item["question"]},
        ],
    })
    resp.raise_for_status()
    return {"answer": resp.json()["choices"][0]["message"]["content"], "sources": []}


def name_index(args) -> dict[str, int] | None:
    """First chapter of each proper name: a capitalized word the book never writes in lowercase
    (so "However" at the start of a sentence is not mistaken for a name)."""
    if not args.source:
        return None
    sys.path.insert(0, str(ROOT / "services" / "ingestion"))
    from adapters import get_adapter
    from chunking import clean_html

    chapters = sorted(get_adapter(args.source_type, args.source).fetch_all(),
                      key=lambda c: c["number"])
    texts = [(c["number"], clean_html(c["content"])) for c in chapters]
    lower = {w for _, t in texts for w in re.findall(r"\b[a-z]+\b", t)}
    first = {}
    for number, text in texts:
        for word in set(re.findall(r"\b[A-Z][a-z]{2,}\b", text)):
            if word.lower() not in lower:
                first.setdefault(word, number)
    return first


def in_context(item, sources) -> bool:
    """Whether the retrieved text held an accepted answer or the answer's evidence."""
    context = normalize(" ".join(s["text"] for s in sources))
    short = [normalize(a) for a in item["accepted_answers"] if len(a.split()) <= 6]
    evidence = normalize(item.get("evidence") or "").strip(" .\"'")
    return any(a in context for a in short) or bool(evidence and evidence in context)


def score(args, item, answer, sources, reveals, names) -> dict:
    chapter = item["current_chapter"]
    later = [r for r in reveals if r["chapter"] > chapter]
    future_names = sorted(w for w in set(re.findall(r"\b[A-Z][a-z]{2,}\b", answer))
                          if names and names.get(w, 0) > chapter)
    revealed = [r["fact"] for r in later
                if r.get("pattern") and re.search(r["pattern"], answer, re.I | re.S)]
    row = {
        "id": item["id"], "category": item["category"], "current_chapter": chapter,
        "question": item["question"], "answer": answer,
        "source_chapters": sorted({s["chapter_number"] for s in sources}),
        "passages_safe": all(s["chapter_number"] <= chapter for s in sources),
        "leak": "LEAK" if future_names else "SAFE",
        "future_names": future_names,
        "leak_review": revealed,
    }
    if item.get("key_terms"):
        if any(re.search(re.escape(k), answer, re.I) for k in item["key_terms"]):
            row["correct"] = "CORRECT"
        else:
            row["correct"] = "DECLINED" if DECLINES.search(answer) else "WRONG"
    if sources and "text" in sources[0] and item["category"] in ("fact", "recent"):
        row["answer_in_context"] = in_context(item, sources)
    if item["category"] == "unanswerable":
        row["flags_unknown"] = bool(FLAGS_UNKNOWN.search(answer))
    if args.judge:
        row["judge"] = {"leak": judge(args, "leak", chapter=chapter, answer=answer,
                                      facts="\n".join(f"- {r['fact']}" for r in later))
                        if later else "SAFE"}
        if item["category"] in ("fact", "recent"):
            row["judge"]["correct"] = judge(args, "correct", question=item["question"],
                                            answer=answer,
                                            accepted="; ".join(item["accepted_answers"]))
        if item["category"] == "unanswerable":
            row["judge"]["abstain"] = judge(args, "abstain", name=item["invented_name"],
                                            question=item["question"], answer=answer)
    return row


def summarize(rows) -> dict:
    def count(rs, key, value):
        return sum(r.get(key) == value for r in rs)

    out = {"overall": {
        "items": len(rows),
        "passages_safe": sum(r["passages_safe"] for r in rows),
        "answers_leaking": count(rows, "leak", "LEAK"),
        "answers_to_review": sum(bool(r.get("leak_review")) for r in rows),
    }}
    timed = sorted(r["seconds"] for r in rows if r.get("seconds") is not None)
    if timed:
        out["overall"]["median_seconds"] = timed[len(timed) // 2]
    for cat in ("fact", "recent"):
        rs = [r for r in rows if r["category"] == cat]
        if rs:
            out[cat] = {"items": len(rs), "correct": count(rs, "correct", "CORRECT"),
                        "declined": count(rs, "correct", "DECLINED"),
                        "wrong": count(rs, "correct", "WRONG")}
            ctx = [r["answer_in_context"] for r in rs if "answer_in_context" in r]
            if ctx:
                out[cat]["answer_in_context"] = sum(ctx)
    rs = [r for r in rows if r["category"] == "spoiler"]
    if rs:
        out["spoiler"] = {"items": len(rs), "leaked": count(rs, "leak", "LEAK")}
    rs = [r for r in rows if r["category"] == "unanswerable"]
    if rs:
        out["unanswerable"] = {"items": len(rs),
                               "flags_unknown": sum(r["flags_unknown"] for r in rs)}
    return out


def agreement(rows, labels) -> dict:
    """How often code and (if run) the judge match the hand grades. A hand PARTIAL accepts either
    CORRECT or WRONG, since key terms cannot see partial answers."""
    def same(dim, got, hand):
        return got == hand or (dim == "correct" and hand == "PARTIAL" and got in ("CORRECT", "WRONG"))

    report = {}
    for source, dims in (("code", ("correct", "leak")), ("judge", ("correct", "leak", "abstain"))):
        for dim in dims:
            pairs = [(r["id"], (r if source == "code" else r.get("judge", {})).get(dim),
                      labels[r["id"]][dim]) for r in rows if dim in labels.get(r["id"], {})]
            pairs = [p for p in pairs if p[1] is not None]
            if pairs:
                report[f"{source} {dim}"] = {
                    "agree": sum(same(dim, g, h) for _, g, h in pairs), "of": len(pairs),
                    "disagreements": [f"{i}: {source} {g}, hand {h}"
                                      for i, g, h in pairs if not same(dim, g, h)]}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--novel-id", type=int)
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--judge", action="store_true", help="also grade with an LLM judge")
    ap.add_argument("--judge-url", default="http://localhost:8004/v1")
    ap.add_argument("--judge-model", default="mistralai/Mistral-Nemo-Instruct-2407")
    ap.add_argument("--source", help="the book, to flag names first mentioned after a chapter")
    ap.add_argument("--source-type", default="epub")
    ap.add_argument("--no-retrieval", action="store_true",
                    help="ask the model directly, without the app (needs --title)")
    ap.add_argument("--title", help="the book's title, for --no-retrieval")
    ap.add_argument("--rejudge", help="re-score the answers stored in this results file")
    ap.add_argument("--labels", help="hand grades to measure scoring against (with --rejudge)")
    ap.add_argument("--in-place", action="store_true", help="with --rejudge: rewrite that file")
    args = ap.parse_args()
    if args.no_retrieval and not args.title:
        ap.error("--no-retrieval needs --title")
    if args.labels and not args.rejudge:
        ap.error("--labels grades stored answers, so it needs --rejudge")
    if not (args.rejudge or args.no_retrieval or args.novel_id):
        ap.error("--novel-id is needed to ask the app")

    set_path = Path(args.set)
    items = [json.loads(line) for line in set_path.open() if line.strip()]
    reveals_path = set_path.with_name(set_path.stem + ".reveals.json")
    reveals = json.loads(reveals_path.read_text()) if reveals_path.exists() else []
    names = name_index(args)

    stored, mode = {}, "no-retrieval" if args.no_retrieval else "app"
    if args.rejudge:
        old = json.loads(Path(args.rejudge).read_text())
        mode = old.get("mode", "app")
        stored = {r["id"]: r for r in old["results"]}
        items = [i for i in items if i["id"] in stored]

    rows = []
    for n, item in enumerate(items, 1):
        started = time.perf_counter()
        if args.rejudge:
            r = stored[item["id"]]
            reply = {"answer": r["answer"], "sources": r.get("sources") or
                     [{"chapter_number": c} for c in r.get("source_chapters", [])]}
            seconds = r.get("seconds")
        else:
            reply = ask_model_directly(args, item) if args.no_retrieval else ask_app(args, item)
            seconds = round(time.perf_counter() - started, 2)
        row = score(args, item, reply["answer"], reply["sources"], reveals, names)
        row["seconds"] = seconds
        if reply["sources"] and "text" in reply["sources"][0]:
            row["sources"] = reply["sources"]
        rows.append(row)
        verdict = row.get("correct") or row["leak"]
        print(f"[{n}/{len(items)}] {item['id']} {item['category']}: {verdict}", file=sys.stderr)

    summary = summarize(rows)
    result = {"set": args.set, "mode": mode, "judge": args.judge_model if args.judge else None,
              "answers_from": args.rejudge, "summary": summary, "results": rows}
    if args.labels:
        labels = json.loads(Path(args.labels).read_text())["runs"].get(args.rejudge, {})
        result["agreement"] = agreement(rows, labels)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = ("-no-retrieval" if mode == "no-retrieval" else "") + ("-rejudge" if args.rejudge else "")
    out = ROOT / "eval" / "results" / f"{set_path.stem}{suffix}-{stamp}.json"
    if args.rejudge and args.in_place:
        out, result["answers_from"] = Path(args.rejudge), None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": summary, "agreement": result.get("agreement")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
