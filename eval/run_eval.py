"""Run an evaluation set through the live app and score it.

    python eval/run_eval.py --set eval/sets/hound.jsonl --novel-id 4 \
        --source ~/novel-data/hound-of-the-baskervilles/hound.epub --source-type epub

Each item goes to the app at its reading position, as a reader's question would: through
/query, or for recap items through the endpoint the item names (/summarize, /catch-me-up),
with the recap written fresh rather than read from the cache. Scoring is done in code wherever
it can be, because an LLM judge proved unreliable:
  name leak      every answer: it names someone or something the book first mentions after
                 the reader's chapter. LEAK or NONE FOUND, and NOT CHECKED without --source.
                 NONE FOUND is not "no spoiler": an answer can give away a later plot point
                 about a character the reader already knows, which no name check can see
  leak review    it matches a pattern for a reveal after the reader's chapter
                 (<set>.reveals.json): a flag for a person to check, not a verdict
  sources        whether every retrieved passage comes from the reader's chapters (by chapter
                 number), and with --source whether a source names anything unread, checked
                 as strictly as the app checks chapter summaries
  fact, recent   CORRECT if the answer contains one of the item's key terms, otherwise
                 DECLINED if it says it cannot answer, otherwise WRONG; and whether an accepted
                 answer or the evidence was in the retrieved text. "recent" questions are about
                 the reader's latest chapter but are not phrased as recaps
  recap          whether the app read the chapter summaries of the expected range (routed),
                 and which of the item's terms (people, places and events of the range) the
                 recap mentions
  unanswerable   whether the answer says it cannot find the person (strict: an answer that
                 neither says so nor invents anything still fails)

So the name check never makes a run spoiler-free. "to_read" lists the answers to read at
least: the spoiler questions, the flagged answers, and a random sample of the rest, since a
plot leak can sit in an answer nothing flagged (three "Rodger Stapleton" leaks were in fact
questions). Grades go in eval/labels/, and --labels counts them.

Measured against hand grades (eval/labels/hound.json): key terms agreed on 59 of the 60
answers they were written against, but on fresh answers 5 of 64 passes were wrong or only
partly right, because a wrong answer can still contain the right name ("Selden is actually
Sherlock Holmes in disguise"), so spot-check what passes. The name check raised no false
alarm in 188 fresh answers, but it found only 7 of the 21 labelled leaks, and 1 of the 13 in
the app's own answers: the others give away plot points about what the reader already knows
("Rodger Stapleton", or at chapter 5 "the hound is a real animal"). The reveal patterns were
right on only 2 of 8 flags (they overfit the answers they were written against), so they only
flag answers for review. Mistral Nemo as a judge agreed on leaks 30/47 and on invented people
4/10, so --judge (an LLM cross-check) is off by default.

--rejudge RESULTS re-scores stored answers without calling the app (--in-place rewrites that
file, keeping stored verdicts in step with the scoring); --labels reports agreement with hand
grades. --no-retrieval asks the model directly, as a chatbot would be asked. The app answers
at temperature 0.3, so runs differ slightly. Each result records the code revision, a hash of
the set, the model and how far the novel's chapter summaries had got.
Results go to eval/results/<set>[-no-retrieval][-rejudge]-<timestamp>.json.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
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

NOT_CHECKED, NONE_FOUND = "NOT CHECKED", "NONE FOUND"
CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")
# nothing but quotes, brackets or list marks since the last sentence end (or the start)
OPENS_SENTENCE = re.compile(r"(^|[.!?:;\n])[^A-Za-z]*$")
HEADING_WORDS = {"chapter", "part", "summary"}
# /catch-me-up reads the reading position stored for a user; the eval keeps its own
EVAL_READER = "eval"

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


def post(args, path: str, payload: dict) -> dict:
    resp = httpx.post(f"{args.gateway}{path}", timeout=300, json=payload)
    resp.raise_for_status()
    return resp.json()


def ask_app(args, item) -> dict:
    """Ask the app as the reader would: /query, or the recap endpoint the item names. Recaps
    skip the cache, which holds whatever an earlier version of the code wrote."""
    chapter = item["current_chapter"]
    endpoint = item.get("endpoint", "query")
    if endpoint == "summarize":
        body = post(args, "/summarize", {
            "novel_id": args.novel_id, "start_chapter": item["start_chapter"],
            "end_chapter": chapter, "current_chapter": chapter, "use_cache": False})
        return {"answer": body.get("summary") or "", "sources": [], "route": body.get("source"),
                "recap_range": [body.get("start_chapter"), body.get("end_chapter")],
                "error": body.get("error")}
    if endpoint == "catch-me-up":
        post(args, "/progress", {"novel_id": args.novel_id, "user_id": EVAL_READER,
                                 "current_chapter": chapter})
        body = post(args, "/catch-me-up", {"novel_id": args.novel_id, "user_id": EVAL_READER,
                                           "use_cache": False})
        span = body.get("summary_range") or {}
        return {"answer": body.get("summary") or "", "sources": [], "route": body.get("source"),
                "recap_range": [span.get("start"), span.get("end")], "error": body.get("error")}
    body = post(args, "/query", {"query": item["question"], "novel_id": args.novel_id,
                                 "current_chapter": chapter})
    sources = [{"chapter_number": s["chapter_number"], "text": s["text"]}
               for s in body.get("sources", [])]
    chapters = sorted({s["chapter_number"] for s in sources})
    return {"answer": body.get("answer") or "", "sources": sources,
            # whether the app found a name the reader has not reached and withheld it
            "spoiler_check": body.get("spoiler_check"),
            # chapter summaries, for a question about recent events, or searched passages
            "route": body.get("source"),
            "recap_range": [chapters[0], chapters[-1]] if chapters else None,
            "model_used": body.get("model_used")}


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
    return {"answer": resp.json()["choices"][0]["message"]["content"], "sources": [],
            "model_used": args.judge_model}


def book_index(args) -> dict | None:
    """The chapter in which the book first uses each word, to tell what a reader has not
    reached: "names", capitalized words the book never writes in lowercase (so "However"
    opening a sentence is not taken for a name), and "words", every word in any case."""
    if not args.source:
        return None
    sys.path.insert(0, str(ROOT / "services" / "ingestion"))
    from adapters import get_adapter
    from chunking import clean_html

    chapters = sorted(get_adapter(args.source_type, args.source).fetch_all(),
                      key=lambda c: c["number"])
    texts = [(c["number"], clean_html(c["content"])) for c in chapters]
    lower = {w for _, t in texts for w in re.findall(r"\b[a-z]+\b", t)}
    names, words = {}, {}
    for number, text in texts:
        for word in set(CAPITALIZED.findall(text)):
            if word.lower() not in lower:
                names.setdefault(word, number)
        for word in set(re.findall(r"[a-z]+", text.lower())):
            words.setdefault(word, number)
    return {"names": names, "words": words}


def unread_in(text: str, chapter: int, book: dict, strict: bool = False) -> list[str]:
    """Words in `text` the book first uses after `chapter`: proper names, and when strict (as
    the app checks chapter summaries, whose model may know a place the book has not named
    yet) any capitalized word that does not open a sentence."""
    found = {w for w in CAPITALIZED.findall(text) if book["names"].get(w, 0) > chapter}
    if strict:
        found |= {m.group() for m in CAPITALIZED.finditer(text)
                  if m.group().lower() not in HEADING_WORDS
                  and book["words"].get(m.group().lower(), 0) > chapter
                  and not OPENS_SENTENCE.search(text[:m.start()])}
    return sorted(found)


def in_context(item, sources) -> bool:
    """Whether the retrieved text held an accepted answer or the answer's evidence."""
    context = normalize(" ".join(s["text"] for s in sources))
    short = [normalize(a) for a in item["accepted_answers"] if len(a.split()) <= 6]
    evidence = normalize(item.get("evidence") or "").strip(" .\"'")
    return any(a in context for a in short) or bool(evidence and evidence in context)


def score(args, item, reply, reveals, book) -> dict:
    """Score what the app returned for an item (`reply`, as from ask_app). `book` is the word
    index from book_index, or None when no book was given: then nothing is checked for names,
    and the row says NOT CHECKED rather than passing."""
    chapter, answer, sources = item["current_chapter"], reply["answer"], reply["sources"]
    later = [r for r in reveals if r["chapter"] > chapter]
    revealed = [r["fact"] for r in later
                if r.get("pattern") and re.search(r["pattern"], answer, re.I | re.S)]
    row = {
        "id": item["id"], "category": item["category"], "current_chapter": chapter,
        "question": item["question"], "answer": answer,
        "source_chapters": sorted({s["chapter_number"] for s in sources}),
        # by chapter number only: a chapter summary from the reader's chapters can still name
        # something from later, which is what sources_unread looks for
        "passages_within_progress": all(s["chapter_number"] <= chapter for s in sources),
        "name_leak": NOT_CHECKED,
        "leak_review": revealed,
    }
    for key in ("route", "error"):
        if reply.get(key):
            row[key] = reply[key]
    if book is not None:
        row["unread_names"] = unread_in(answer, chapter, book)
        row["name_leak"] = "LEAK" if row["unread_names"] else NONE_FOUND
        texts = [s["text"] for s in sources if s.get("text")]
        if texts:
            row["sources_unread"] = sorted({w for t in texts
                                            for w in unread_in(t, chapter, book, strict=True)})
    if item["category"] == "recap":
        row["covers_found"] = [t for t in item["covers"]
                               if re.search(r"\b" + re.escape(t), answer, re.I)]
        row["covers_missing"] = [t for t in item["covers"] if t not in row["covers_found"]]
        row["recap_range"] = reply.get("recap_range")
        row["routed"] = (reply.get("route") == "chapter summaries"
                         and row["recap_range"] == item["expected_range"])
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


# Unflagged answers read as well, drawn from the run's own answers, so re-scoring a run
# draws the same ones while each new run draws others
UNFLAGGED_TO_READ = 5


def to_read(rows) -> list[str]:
    """The answers a person must read before trusting a run on spoilers: the spoiler questions,
    the answers matching a reveal pattern, and a random sample of the rest."""
    listed = [r["id"] for r in rows if r["category"] == "spoiler" or r.get("leak_review")]
    rest = sorted(r["id"] for r in rows if r["id"] not in listed)
    seed = hashlib.sha256("\n".join(r["answer"] for r in rows).encode()).hexdigest()
    return listed + sorted(random.Random(seed).sample(rest, min(UNFLAGGED_TO_READ, len(rest))))


def summarize(rows) -> dict:
    def count(rs, key, value):
        return sum(r.get(key) == value for r in rs)

    checked = any(r["name_leak"] != NOT_CHECKED for r in rows)

    def name_leaks(rs):
        return count(rs, "name_leak", "LEAK") if checked else NOT_CHECKED

    out = {"overall": {"items": len(rows)}}
    timed = sorted(r["seconds"] for r in rows if r.get("seconds") is not None)
    if timed:
        out["overall"]["median_seconds"] = timed[len(timed) // 2]
    out["spoilers"] = {
        "passages_within_progress": sum(r["passages_within_progress"] for r in rows),
        "sources_naming_unread": (sum(bool(r.get("sources_unread")) for r in rows)
                                  if checked else NOT_CHECKED),
        "answers_naming_unread": name_leaks(rows),
        # what no name check sees: a later plot point about someone the reader knows
        "plot_leaks": "not checked in code: read the answers in to_read (with a sample of "
                      "unflagged ones), and more of the rest before any claim of none",
        "flagged_for_review": [r["id"] for r in rows if r.get("leak_review")],
        "to_read": to_read(rows),
    }
    by_app = [r["spoiler_check"] for r in rows if r.get("spoiler_check")]
    if by_app:
        out["spoilers"]["withheld_by_the_app"] = sum(c != "clean" for c in by_app)
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
        out["spoiler"] = {"items": len(rs), "name_leaks": name_leaks(rs),
                          "flagged_for_review": sum(bool(r.get("leak_review")) for r in rs)}
    rs = [r for r in rows if r["category"] == "unanswerable"]
    if rs:
        out["unanswerable"] = {"items": len(rs),
                               "flags_unknown": sum(r["flags_unknown"] for r in rs)}
    rs = [r for r in rows if r["category"] == "recap"]
    if rs:
        found = sum(len(r["covers_found"]) for r in rs)
        terms = found + sum(len(r["covers_missing"]) for r in rs)
        out["recap"] = {"items": len(rs), "routed": sum(r["routed"] for r in rs),
                        "terms_mentioned": f"{found} of {terms}", "name_leaks": name_leaks(rs)}
    return out


def agreement(rows, labels) -> dict:
    """How often code and (if run) the judge match the hand grades. A hand PARTIAL accepts either
    CORRECT or WRONG, since key terms cannot see partial answers. A hand leak is any leak, while
    the code's verdict is its name check, so the hand leaks it misses are plot points."""
    def same(dim, got, hand):
        return got == hand or (dim == "correct" and hand == "PARTIAL" and got in ("CORRECT", "WRONG"))

    report = {}
    for source, dims in (("code", ("correct", "leak")), ("judge", ("correct", "leak", "abstain"))):
        for dim in dims:
            key = "name_leak" if (source, dim) == ("code", "leak") else dim
            pairs = [(r["id"], (r if source == "code" else r.get("judge", {})).get(key),
                      labels[r["id"]][dim]) for r in rows if dim in labels.get(r["id"], {})]
            # nothing found is compared with a hand SAFE; what was not checked is left out
            pairs = [(i, "SAFE" if g == NONE_FOUND else g, h) for i, g, h in pairs
                     if g not in (None, NOT_CHECKED)]
            if pairs:
                report[f"{source} {key}"] = {
                    "agree": sum(same(dim, g, h) for _, g, h in pairs), "of": len(pairs),
                    "disagreements": [f"{i}: {source} {g}, hand {h}"
                                      for i, g, h in pairs if not same(dim, g, h)]}
                if dim == "leak":
                    report[f"{source} {key}"]["hand_leaks_found"] = (
                        f"{sum(g == h == 'LEAK' for _, g, h in pairs)} of "
                        f"{sum(h == 'LEAK' for _, _, h in pairs)}")
    return report


def run_info(args, set_path: Path, models: set) -> dict:
    """What a run measured, so two runs can be compared: the code (of this checkout, which the
    stack should have been deployed from), the questions, the model, and how far the chapter
    summaries that recaps read had got."""
    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    info = {"date": datetime.now().isoformat(timespec="seconds"),
            "revision": git("rev-parse", "--short", "HEAD"),
            "uncommitted_changes": None if status is None else bool(status),
            "set_sha256": hashlib.sha256(set_path.read_bytes()).hexdigest()[:16],
            "models": sorted(models),
            "name_check": "on" if args.source else NOT_CHECKED}
    if args.novel_id and not args.no_retrieval:
        try:
            resp = httpx.get(f"{args.gateway}/novels/{args.novel_id}/summaries", timeout=10)
            resp.raise_for_status()
            info["chapter_summaries"] = resp.json()
        except httpx.HTTPError as e:
            info["chapter_summaries"] = f"unavailable: {e}"
    return info


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
    book = book_index(args)
    if book is None:
        print("No --source: answers are not checked for names the reader has not reached, "
              f"and say {NOT_CHECKED}.", file=sys.stderr)

    stored, mode, info = {}, "no-retrieval" if args.no_retrieval else "app", None
    if args.rejudge:
        old = json.loads(Path(args.rejudge).read_text())
        mode, info = old.get("mode", "app"), old.get("run")
        stored = {r["id"]: r for r in old["results"]}
        items = [i for i in items if i["id"] in stored]

    rows, models = [], set()
    for n, item in enumerate(items, 1):
        started = time.perf_counter()
        if args.rejudge:
            r = stored[item["id"]]
            reply = {"answer": r["answer"], "sources": r.get("sources") or
                     [{"chapter_number": c} for c in r.get("source_chapters", [])],
                     "route": r.get("route"), "recap_range": r.get("recap_range"),
                     "error": r.get("error")}
            seconds = r.get("seconds")
        else:
            reply = ask_model_directly(args, item) if args.no_retrieval else ask_app(args, item)
            seconds = round(time.perf_counter() - started, 2)
            if reply.get("model_used"):
                models.add(reply["model_used"])
        row = score(args, item, reply, reveals, book)
        row["seconds"] = seconds
        if reply.get("spoiler_check"):
            row["spoiler_check"] = reply["spoiler_check"]
        elif args.rejudge and stored[item["id"]].get("spoiler_check"):
            row["spoiler_check"] = stored[item["id"]]["spoiler_check"]
        if reply["sources"] and "text" in reply["sources"][0]:
            row["sources"] = reply["sources"]
        rows.append(row)
        verdict = row.get("correct") or row["name_leak"]
        if item["category"] == "recap":
            verdict = (f"{len(row['covers_found'])} of {len(item['covers'])} terms, "
                       + ("routed" if row["routed"]
                          else f"NOT ROUTED ({row.get('route')}, {row['recap_range']})"))
        if row.get("error"):
            verdict += f" (error: {row['error']})"
        print(f"[{n}/{len(items)}] {item['id']} {item['category']}: {verdict}", file=sys.stderr)

    summary = summarize(rows)
    result = {"set": args.set, "mode": mode, "judge": args.judge_model if args.judge else None,
              "answers_from": args.rejudge,
              "run": info if args.rejudge else run_info(args, set_path, models),
              "summary": summary, "results": rows}
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
    print(f"Spoilers: names {'are checked' if book else 'are NOT CHECKED'}; plot points are "
          f"not. Read at least the {len(summary['spoilers']['to_read'])} answers in to_read "
          "(spoiler questions, flagged answers and a sample of the rest).", file=sys.stderr)


if __name__ == "__main__":
    main()
