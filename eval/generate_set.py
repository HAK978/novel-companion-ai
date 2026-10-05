"""Draft an evaluation set for one book with an LLM, keeping only what can be checked.

Works for any book an ingestion adapter can read. For each chapter the model drafts:
  fact        a fact from the chapter, asked at a later reading position
  recent      a fact from the chapter the reader has just finished
  spoiler     a reveal from the chapter, asked before the reader gets there; the answer
              must NOT contain it
  unanswerable  a question about something the book never mentions

Checks that need no judgment are applied in code: every answer must quote its evidence
verbatim from the chapter, and an unanswerable question must name something absent from
the whole book. A second model pass then drops questions that are not self-contained
(e.g. "what did he see in the sky?", which has many answers in a long book). The output is
a draft: read it before freezing it.

    python eval/generate_set.py --book hound --source-type epub \
        --source /home/me/novel-data/hound-of-the-baskervilles/hound.epub
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "ingestion"))
from adapters import get_adapter  # noqa: E402
from chunking import clean_html  # noqa: E402

LLM_URL = "http://localhost:8004/v1"
MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
MAX_WORDS = 7000  # keeps one chapter inside the model's context

DRAFT_PROMPT = """You are writing test questions for a reading companion app. Below is \
chapter {n} of a book.

Write {k} questions of this kind: {kind}

Rules:
- Each question must make sense on its own in a long book: name the people, places or
  events involved, so it has exactly one correct answer. Never use bare pronouns.
- The answer must be stated in this chapter.
- "evidence" must be an exact, word-for-word quote from the chapter (one or two sentences)
  that supports the answer.

Return only a JSON array of objects with keys "question", "answer", "evidence".

Chapter {n}:
{text}"""

KINDS = {
    "fact": "a concrete fact a reader might later want to look up (who, what, where, why).",
    "spoiler": "a question about a SECRET or SURPRISE that this chapter reveals for the "
               "first time: a hidden identity, a true relationship, who was behind "
               "something, a death or outcome. Before this chapter a reader could not "
               "know the answer. Phrase it as an innocent question a curious reader might "
               "ask earlier. If the chapter reveals no secret, return an empty array.",
}

UNANSWERABLE_PROMPT = """You are writing test questions for a reading companion app. Below \
is chapter {n} of a book.

Write 2 natural questions a reader might ask, each about a person with a made-up name that \
does NOT appear in this book, presented as if they were a character (for example: "Why did \
Edmund Hale leave the village?"). Do not mark the name as made up, do not mention chapter \
numbers, and do not use any real character's name. Put the made-up name in "invented".

Return only a JSON array of objects with keys "question", "invented".

Chapter {n}:
{text}"""

CHECK_PROMPT = """A question will be asked about a long novel, by a reader who has read part \
of it. Is the question self-contained and specific enough that it has a single correct \
answer in the whole book? Questions with bare pronouns ("what did he see?") or generic \
events ("what happened at the meeting?") are not.

Question: {question}

Answer with one word: YES or NO."""


def llm(prompt: str, max_tokens: int = 1500) -> str:
    resp = httpx.post(f"{LLM_URL}/chat/completions", timeout=300, json={
        "model": MODEL, "temperature": 0.3, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_array(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict)]


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def quoted(evidence: str, chapter: str) -> bool:
    ev = normalize(evidence).strip(" .\"'")
    return len(ev) >= 20 and ev in chapter


TITLES = {"mr", "mrs", "miss", "dr", "sir", "lady", "lord", "inspector", "reverend", "the"}


def absent_from(name: str, book: str) -> bool:
    """Every word of the name (titles aside) is missing from the book."""
    words = [w for w in re.findall(r"[a-z']+", normalize(name)) if w not in TITLES]
    return bool(words) and not any(re.search(rf"\b{re.escape(w)}\b", book) for w in words)


def well_formed(question: str) -> bool:
    return not re.search(r"\bchapter\b|invented|made-up|made up", question, re.I)


def self_contained(question: str) -> bool:
    return llm(CHECK_PROMPT.format(question=question), max_tokens=3).strip().upper().startswith("Y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="short name, used in ids and the file name")
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-type", default="epub")
    ap.add_argument("--per-chapter", type=int, default=4, help="fact questions drafted per chapter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    chapters = get_adapter(args.source_type, args.source).fetch_all()
    texts = {c["number"]: clean_html(c["content"]) for c in chapters}
    last = max(texts)
    whole_book = normalize(" ".join(texts.values()))
    items, dropped = [], {"no_quote": 0, "not_self_contained": 0, "name_in_book": 0}

    def keep(category, question, answer, evidence, source, position):
        items.append({
            "id": f"{args.book}-{len(items) + 1:03d}", "category": category,
            "question": question, "current_chapter": position,
            "reference_answer": answer, "evidence": evidence,
            "expected_source_chapters": [source] if category in ("fact", "recent") else [],
            "reveal_chapter": source if category == "spoiler" else None,
        })

    for n, text in texts.items():
        body = " ".join(text.split()[:MAX_WORDS])
        norm = normalize(text)
        print(f"chapter {n}", file=sys.stderr)

        drafts = [("fact", d) for d in parse_array(llm(DRAFT_PROMPT.format(
            n=n, k=args.per_chapter, kind=KINDS["fact"], text=body)))]
        if n >= 3:  # a reveal needs earlier chapters to be asked from
            drafts += [("spoiler", d) for d in parse_array(llm(DRAFT_PROMPT.format(
                n=n, k=2, kind=KINDS["spoiler"], text=body)))]

        for i, (kind, d) in enumerate(drafts):
            q, a, ev = d.get("question", ""), d.get("answer", ""), d.get("evidence", "")
            if kind == "spoiler" and re.search(r"not (yet )?(revealed|stated|mentioned)", a, re.I):
                continue
            if not (q and a and well_formed(q) and quoted(ev, norm)):
                dropped["no_quote"] += 1
                continue
            if not self_contained(q):
                dropped["not_self_contained"] += 1
                continue
            if kind == "spoiler":
                keep("spoiler", q, a, ev, n, rng.randint(1, n - 2))
            elif i == 0:
                keep("recent", q, a, ev, n, n)  # just read
            else:
                keep("fact", q, a, ev, n, rng.randint(n, last))

        for d in parse_array(llm(UNANSWERABLE_PROMPT.format(n=n, text=body))):
            q, name = d.get("question", ""), d.get("invented", "")
            if not (q and name and name.lower() in q.lower() and well_formed(q)):
                continue
            if not absent_from(name, whole_book):
                dropped["name_in_book"] += 1
                continue
            keep("unanswerable", q, "The book does not say.", "", n, rng.randint(n, last))

    out = ROOT / "eval" / "sets" / f"{args.book}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(i) + "\n" for i in items))
    counts = {c: sum(i["category"] == c for i in items) for c in
              ("fact", "recent", "spoiler", "unanswerable")}
    print(f"wrote {len(items)} to {out}: {counts}; dropped {dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
