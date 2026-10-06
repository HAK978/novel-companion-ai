"""The book's word record against real PostgreSQL: written as chapters are ingested, read to
find names a reader has not reached (in answers) and to flag summaries.

Passages only come from chapters a reader has read, but the model may know the book: at
chapter 14 of the Hound it called Stapleton's wife "Beryl Garcia", a name first used in
chapter 15.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

pytestmark = pytest.mark.db


@pytest.fixture
def book(ingestion_tasks, db_engine, monkeypatch):
    """Novel 1; record(chapter, text) stores a chapter's words as ingestion does."""
    monkeypatch.setattr(ingestion_tasks, "engine", db_engine)
    with db_engine.begin() as conn:
        conn.execute(text("INSERT INTO novels (id, title) VALUES (1, 'Main')"))

    def record(chapter, words):
        with db_engine.connect() as conn:
            ingestion_tasks._record_words(conn, 1, chapter, words)
            conn.commit()

    def stored():
        with db_engine.connect() as conn:
            return {w: (first, lower) for w, first, lower in conn.execute(text(
                "SELECT word, first_chapter, first_lowercase_chapter FROM book_words"))}

    record.stored = stored
    return record


def test_each_word_keeps_the_chapter_it_first_appears_in(book):
    # chapters arrive in any order, and again on re-ingestion
    book(15, "Beryl Garcia of Costa Rica.")
    book(7, "Miss Beryl walked on the moor.")
    book(15, "Beryl Garcia of Costa Rica.")

    stored = book.stored()
    assert stored["beryl"] == (7, None)
    assert stored["garcia"] == (15, None)
    assert stored["moor"] == (7, 7)


def test_a_word_also_written_in_lowercase_is_not_a_name(book):
    book(3, "The Hound howled.")
    book(9, "and the hound came")

    assert book.stored()["hound"] == (3, 9)


@pytest.fixture
def check(retrieval_main, db_engine, book, monkeypatch):
    """The text as a reader at `chapter` gets it back, and how many names were taken out."""
    monkeypatch.setattr(retrieval_main, "engine", db_engine)
    client = TestClient(retrieval_main.app)
    book(1, "she was his wife and she came home at the end")  # ordinary words, early on
    book(7, "Miss Beryl Stapleton lived at Merripit House.")
    book(15, "She was Beryl Garcia, of Costa Rica.")
    book(16, "The Moor was quiet.")
    book(20, "out on the moor")

    def withhold(text, chapter=14, ignore=""):
        body = client.post("/withhold-unread-names", json={
            "novel_id": 1, "current_chapter": chapter, "text": text, "ignore": ignore}).json()
        return body["text"], body["withheld"]
    return withhold


def test_names_first_used_after_the_readers_chapter_are_taken_out(check):
    assert check("She is his wife, Beryl Garcia. She came from Costa Rica. Sad.") == (
        "She is his wife, Beryl. Sad.", 3)


def test_names_the_reader_has_reached_stay(check):
    text = "She is Beryl Garcia, of Costa Rica."
    assert check(text, chapter=15) == (text, 0)


def test_the_readers_own_words_stay(check):
    text = "Garcia is not in what you've read."
    assert check(text, ignore="Who is garcia?") == (text, 0)


def test_words_the_book_writes_in_lowercase_are_not_names(check):
    # first used capitalized in chapter 16 ("The Moor"), but an ordinary word
    assert check("The Moor was dark.", chapter=3) == ("The Moor was dark.", 0)


def test_names_the_book_never_uses_are_not_this_checks_business(check):
    # invented names are a different failure, flagged in summaries
    assert check("Sebastian fled.") == ("Sebastian fled.", 0)


def test_summaries_are_flagged_against_the_book_so_far(ingestion_tasks, book):
    book(1, "Sunny waited by the gate.")
    book(3, "Nephis arrived.")
    first = ingestion_tasks._first_chapters(1, ["Sunny", "Nephis", "Mordret"])

    summary = "In chapter 2, Sunny meets Nephis and Mordret."
    assert ingestion_tasks._unverified_names(summary, first, 2) == ["Mordret", "Nephis"]
    assert ingestion_tasks._unverified_names(summary, first, 3) == ["Mordret"]


def test_the_record_can_be_rebuilt_from_indexed_chapters(ingestion_tasks, book, db_engine,
                                                        monkeypatch):
    # novels ingested before the record existed
    with db_engine.begin() as conn:
        for n in (1, 2):
            conn.execute(text("INSERT INTO chapters (novel_id, chapter_number, title, "
                              "ingestion_status) VALUES (1, :n, :t, 'complete')"),
                         {"n": n, "t": f"Chapter {n}"})
    texts = {1: "Sunny waited.", 2: "Nephis arrived."}
    monkeypatch.setattr(ingestion_tasks, "_chapter_text",
                        lambda novel_id, n: (f"Chapter {n}", texts[n]))

    assert ingestion_tasks.rebuild_word_index(1) == {"novel_id": 1, "chapters": 2}
    stored = book.stored()
    # (the record keys words by their reduced form: "Nephis" is "nephi", like a plural)
    assert stored["sunny"] == (1, None) and stored["nephi"] == (2, None)
    assert stored["chapter"] == (1, None)  # titles count: the reader sees them
