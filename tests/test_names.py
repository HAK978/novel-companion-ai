"""Taking names a reader has not reached out of an answer, keeping what makes sense."""

import pytest
from conftest import load_module


@pytest.fixture(scope="module")
def names():
    return load_module("retrieval", "names.py", alias="retrieval_names")


@pytest.mark.parametrize("text,kept", [
    # part of a longer name: taken out of it, which keeps the model's answer
    ("She is his wife, Beryl Garcia, as Holmes said.", "She is his wife, Beryl, as Holmes said."),
    ("He met Beryl Garcia's husband.", "He met Beryl's husband."),
    # the only name in a sentence: the sentence goes
    ("Beryl waited. Garcia was her name. Night fell.", "Beryl waited. Night fell."),
    ("She came from Costa Rica. Beryl waited.", "Beryl waited."),
    # plurals and possessives are the same name; "The" is no name, so the sentence goes
    ("The Garcias were proud. Beryl waited.", "Beryl waited."),
    ("Nothing here is unread.", "Nothing here is unread."),
])
def test_unread_names_are_withheld(names, text, kept):
    unread = {"garcia", "costa", "rica"}
    book_names = unread | {"beryl", names.stem("Holmes")}  # words the book only capitalizes

    assert names.withhold(text, unread, book_names) == kept


@pytest.mark.parametrize("text,found", [
    ("Sunny is exploring the Underworld.", {"underworld"}),
    # opening a sentence, a word is capitalized anyway: only the name mid-sentence counts
    ("Afterward, Sunny rests. Notably, he wins.", {"sunny"}),
    # nor are the headings a model writes the book's words
    ("**Chapter 863 Summary:** Sunny waits.", set()),
    ("He met Beryl Garcia's husband.", {"beryl", "garcia"}),
])
def test_capitalized_words_opening_no_sentence(names, text, found):
    assert names.capitalized_mid_sentence(text) == found
