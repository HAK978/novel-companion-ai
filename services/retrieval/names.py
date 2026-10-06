"""Taking names a reader has not reached out of text written for them."""

import re

from words import stem

_SENTENCES = re.compile(r"(?<=[.!?])\s+")
# Capitalized words in a row: one name, perhaps of several words ("Beryl Garcia", "Costa Rica")
_NAME_RUN = re.compile(r"\b[A-Z][a-z'\u2019]*(?:\s+[A-Z][a-z'\u2019]*)*")


def withhold(text: str, unread: set[str], names: set[str]) -> str:
    """`text` without the words in `unread` (reduced, as in words.py). One that is part of a
    longer name is taken out of it ("his wife, Beryl Garcia" becomes "his wife, Beryl"),
    which keeps the answer the model gave. A sentence where nothing of the name is left that
    is itself a name (`names`: words the book only ever capitalizes) is dropped: "The Garcias
    were proud" must not become "The were proud"."""
    def is_unread(word: str) -> bool:
        return stem(word.replace("\u2019", "'")) in unread

    kept = []
    for sentence in _SENTENCES.split(text):
        dropped = False

        def shorten(match):
            nonlocal dropped
            words = match.group().split()
            left = [w for w in words if not is_unread(w)]
            if len(left) == len(words):
                return match.group()
            if not any(stem(w.replace("\u2019", "'")) in names for w in left):
                dropped = True
                return match.group()
            if is_unread(words[-1]) and words[-1].endswith(("'s", "\u2019s")):
                left[-1] += "'s"  # "Beryl Garcia's husband" becomes "Beryl's husband"
            return " ".join(left)

        sentence = _NAME_RUN.sub(shorten, sentence)
        if not dropped:
            kept.append(sentence)
    return " ".join(kept)
