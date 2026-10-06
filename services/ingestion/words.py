"""Reducing a book's words to the form its word record keys on.

Ingestion records where each word of a book first appears (book_words); retrieval checks
text written for a reader against that record for names they have not reached yet. Both
must reduce a word the same way, or a name would never match its record, so identical
copies live in services/ingestion and services/retrieval (tests/test_words.py fails if they
drift).
"""

import re

WORD = re.compile(r"[A-Za-z']+")
# Capitalized words of three letters or more: what could be a name
CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")


def stem(word: str) -> str:
    """Lowercased, with plurals and possessives folded: "Baskervilles" and "Baskerville's"
    both become "baskerville", "Memories" becomes "memory"."""
    word = word.strip("'").lower()
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    for suffix in ("'s", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def word_uses(text: str) -> dict[str, bool]:
    """Each word in `text`, reduced, and whether the text writes it in lowercase anywhere. A
    word a book only ever writes capitalized is a name."""
    uses: dict[str, bool] = {}
    for token in WORD.findall(text):
        token = token.strip("'")
        if token:
            key = stem(token)
            uses[key] = uses.get(key, False) or token[0].islower()
    return uses
