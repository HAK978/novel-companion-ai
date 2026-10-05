"""Search first, then read: answer a question from the few chapters a keyword search finds,
read in full.

    python eval/experiments/search_then_read.py --source ~/novel-data/shadow-slave \
        --end 1291 --question "How many steps of Shadow Dance has Sunny mastered so far?"

Every chapter up to the reader's is ranked with BM25 and the top few are read, in story order,
in one call. On Shadow Slave at chapter 1291, asked "How many steps of Shadow Dance has Sunny
mastered so far, and how did he reach each one?", it read chapters 218, 359, 539, 653, 1140
and 1291 and answered correctly in 13.6 s. Reading all 300 chapters from 992 took about 5
minutes and produced a list of 11 "steps", mostly invented (recap_methods.py): with every
chapter in front of it, the model reports something "relevant" everywhere. --dry-run shows the
chosen chapters without calling a model.
"""

import argparse
import math
import re
import sys
import time
from collections import Counter

import recap_methods as rm

STOPWORDS = set("""a about after again all also an and any are as at be been before being both
but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its just me more most my no nor not now of off
on once only or other our out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your far many much""".split())


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def bm25_rank(query: str, docs: dict[int, str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    terms = [t for t in tokens(query) if t not in STOPWORDS]
    counts = {n: Counter(tokens(text)) for n, text in docs.items()}
    lengths = {n: sum(c.values()) for n, c in counts.items()}
    average = sum(lengths.values()) / len(lengths)
    df = Counter(t for c in counts.values() for t in set(c) if t in terms)

    def score(n):
        c = counts[n]
        return sum(math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                   * c[t] * (k1 + 1) / (c[t] + k1 * (1 - b + b * lengths[n] / average))
                   for t in terms if c[t])

    return sorted(docs, key=score, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-type", default="local_json")
    ap.add_argument("--end", type=int, required=True, help="the reader's chapter")
    ap.add_argument("--question", required=True)
    ap.add_argument("--top", type=int, default=6, help="chapters to read")
    ap.add_argument("--dry-run", action="store_true", help="only show which chapters are chosen")
    args = ap.parse_args()

    chapters = {c["number"]: rm.clean_html(c["content"])
                for c in rm.get_adapter(args.source_type, args.source).fetch_all()
                if c["number"] <= args.end}
    chapters = {n: t for n, t in chapters.items() if t}
    picked = sorted(bm25_rank(args.question, chapters)[:args.top])
    print(f"chapters chosen: {picked}", file=sys.stderr)
    if args.dry_run:
        return

    # keep what fits in one call, dropping the lowest-ranked chapters first
    while True:
        text = "\n\n".join(f"[Chapter {n}]\n{chapters[n]}" for n in picked)
        if len(picked) == 1 or rm.count_tokens(text) <= rm.BLOCK_TOKENS:
            break
        ranked = bm25_rank(args.question, {n: chapters[n] for n in picked})
        picked = sorted(ranked[:-1])

    started, meter = time.perf_counter(), rm.Meter()
    answer = rm.chat(
        f'A reader asked: "{args.question}"\n\nBelow are the chapters a search found most '
        "relevant, in story order. Answer from them only, citing chapters. If they disagree, "
        f"the later chapter is the current state. At most 250 words.\n\n{text}",
        args.end, meter, 600)
    print(f"read chapters {picked} in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    print(answer)


if __name__ == "__main__":
    main()
