"""Run an evaluation set through the live app and score it.

    python eval/run_eval.py --set eval/sets/hound.jsonl --novel-id 4

Each question goes through the gateway's /query exactly as a reader's would. Scores:
  spoiler-safe sources  no retrieved passage comes after the reader's chapter (code, all items)
  context recall        the chapter holding the answer was retrieved (code, fact/recent)
  correct               the answer agrees with the reference (LLM judge, fact/recent)
  no leak               the answer does not disclose a later reveal (LLM judge, spoiler)
  abstained             the answer says the book does not tell, instead of inventing (judge)
  declined              an answerable question was answered "not stated" (code, fact/recent)

The judge is the model the app itself uses unless --judge-url/--judge-model say otherwise;
a model grading its own answers is lenient, so compare runs with the same judge only.
Results go to eval/results/<set>-<timestamp>.json.

--no-retrieval asks the model directly instead, the way a reader would ask a chatbot: told
the book and their chapter, asked not to spoil, answering from what it already knows. The
same judges score it, so the two runs show what retrieval adds.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

JUDGES = {
    "correct": """Grade an answer to a question about a novel against the reference answer.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Is the answer correct? It may add detail, but must agree with the reference on the point
asked. An answer that says the information is not given, not stated or not yet revealed is
WRONG, whatever else it guesses. {verdict}CORRECT, PARTIAL or WRONG.""",
    "leak": """A reader has not yet reached the part of a novel where this is revealed:
{reference}

Here is an answer the reader was shown:
{answer}

Does the answer disclose that reveal, or strongly hint at it? It counts as a leak even if the
answer claims the reveal happened earlier than it does. {verdict}LEAK or SAFE.""",
    "abstain": """A reader asked about something that does not appear in the novel at all.

Question: {question}
Answer: {answer}

This person or event is not in the book. Did the answer say so, or that it does not know?
If it describes them as if they were real, even partly or with hedging, it INVENTED.
{verdict}ABSTAINED or INVENTED.""",
}
# a bare one-word verdict was unreliable (it passed invented characters as abstentions);
# a sentence of reasoning first, then the verdict, agreed with hand labels
VERDICT = ("First give one sentence of reasoning. Then on a new line write only the "
           "verdict, one of: ")
JUDGE_FOR = {"fact": "correct", "recent": "correct", "spoiler": "leak", "unanswerable": "abstain"}
PASS = {"correct": "CORRECT", "leak": "SAFE", "abstain": "ABSTAINED"}
# Checked in code, not by the judge: a judge of the app's own model let these pass as correct
DECLINES = re.compile(r"not (been )?(explicitly )?(stated|mentioned|revealed|given|provided|"
                      r"specified)|has not revealed|does not (say|mention|specify)", re.I)


NO_RETRIEVAL_SYSTEM = """The user is reading {title} and has reached chapter {chapter}. Answer
their question from your knowledge of the book. Do not reveal anything that happens after
chapter {chapter}. If you do not know, say so."""


def ask_model_directly(args, item):
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


def judge(args, kind, item, answer):
    prompt = JUDGES[kind].format(question=item["question"], answer=answer,
                                 reference=item["reference_answer"], verdict=VERDICT)
    resp = httpx.post(f"{args.judge_url}/chat/completions", timeout=120, json={
        "model": args.judge_model, "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].upper()
    labels = re.findall(r"\b(CORRECT|PARTIAL|WRONG|LEAK|SAFE|ABSTAINED|INVENTED)\b", text)
    return labels[-1] if labels else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--novel-id", type=int, required=True)
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--judge-url", default="http://localhost:8004/v1")
    ap.add_argument("--judge-model", default="mistralai/Mistral-Nemo-Instruct-2407")
    ap.add_argument("--no-retrieval", action="store_true",
                    help="ask the model directly, without the app (needs --title)")
    ap.add_argument("--title", help="the book's title, for --no-retrieval")
    args = ap.parse_args()
    if args.no_retrieval and not args.title:
        ap.error("--no-retrieval needs --title")

    items = [json.loads(line) for line in open(args.set) if line.strip()]
    results = []
    for n, item in enumerate(items, 1):
        started = time.perf_counter()
        if args.no_retrieval:
            body = ask_model_directly(args, item)
        else:
            resp = httpx.post(f"{args.gateway}/query", timeout=300, json={
                "query": item["question"], "novel_id": args.novel_id,
                "current_chapter": item["current_chapter"],
            })
            resp.raise_for_status()
            body = resp.json()
        seconds = time.perf_counter() - started
        sources = sorted({s["chapter_number"] for s in body.get("sources", [])})
        kind = JUDGE_FOR[item["category"]]
        verdict = judge(args, kind, item, body["answer"])
        row = {
            "id": item["id"], "category": item["category"],
            "current_chapter": item["current_chapter"], "question": item["question"],
            "answer": body["answer"], "source_chapters": sources, "seconds": round(seconds, 2),
            "sources_spoiler_safe": all(c <= item["current_chapter"] for c in sources),
            "verdict": verdict, "passed": verdict == PASS[kind],
            "declined": bool(DECLINES.search(body["answer"])),
        }
        if item["expected_source_chapters"] and not args.no_retrieval:
            row["context_recall"] = any(c in sources for c in item["expected_source_chapters"])
        results.append(row)
        print(f"[{n}/{len(items)}] {item['id']} {item['category']}: {verdict}")

    summary = {"overall": {
        "items": len(results),
        "sources_spoiler_safe": sum(r["sources_spoiler_safe"] for r in results) / len(results),
        "median_seconds": sorted(r["seconds"] for r in results)[len(results) // 2],
    }}
    for cat in ("fact", "recent", "spoiler", "unanswerable"):
        rows = [r for r in results if r["category"] == cat]
        if not rows:
            continue
        summary[cat] = {"items": len(rows),
                        PASS[JUDGE_FOR[cat]].lower(): sum(r["passed"] for r in rows) / len(rows)}
        if cat in ("fact", "recent"):
            summary[cat]["declined"] = sum(r["declined"] for r in rows) / len(rows)
        recall = [r["context_recall"] for r in rows if "context_recall" in r]
        if recall:
            summary[cat]["context_recall"] = sum(recall) / len(recall)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "-no-retrieval" if args.no_retrieval else ""
    out = ROOT / "eval" / "results" / f"{Path(args.set).stem}{mode}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set": args.set, "judge": args.judge_model,
                               "mode": "no-retrieval" if args.no_retrieval else "app",
                               "summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
