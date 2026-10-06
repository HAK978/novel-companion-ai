"""Reducing words the same way where the word record is written (ingestion) and where text
is checked against it (retrieval): a mismatch would mean a name never matches its record."""

import pytest
from conftest import ROOT, load_module

COPIES = [ROOT / "services" / svc / "words.py" for svc in ("ingestion", "retrieval")]


@pytest.fixture(scope="module")
def words():
    return load_module("ingestion", "words.py", alias="ingestion_words")


def test_both_services_carry_identical_copies():
    assert COPIES[0].read_text() == COPIES[1].read_text()


@pytest.mark.parametrize("word,reduced", [
    ("Garcia", "garcia"),
    ("Baskervilles", "baskerville"),
    ("Baskerville's", "baskerville"),
    ("Memories", "memory"),
    ("'Tis", "tis"),
    ("its", "its"),  # too short to be a plural
])
def test_plurals_and_possessives_fold(words, word, reduced):
    assert words.stem(word) == reduced


def test_a_word_only_ever_capitalized_is_a_name(words):
    uses = words.word_uses("Garcia met the hound. The Hound howled. 'Garcia!'")

    assert uses["garcia"] is False
    assert uses["hound"] is True
