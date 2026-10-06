"""Chapter summaries: scheduling and writing them, against real PostgreSQL.

The model, Redis and the vector store are stubbed; the SQL is real. Novel 1 has chapters 1-6
ingested, a reader at chapter 2, and a lookahead of 2 chapters.
"""

import httpx
import pytest
from celery.exceptions import Retry
from sqlalchemy import text

pytestmark = pytest.mark.db


class FakeRedis:
    def __init__(self):
        self.sets = {}

    def sadd(self, key, *members):
        found = self.sets.setdefault(key, set())
        new = set(members) - found
        found.update(members)
        return len(new)

    def srem(self, key, *members):
        self.sets.get(key, set()).difference_update(members)

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def expire(self, key, seconds):
        pass


@pytest.fixture
def novel(ingestion_tasks, db_engine, monkeypatch):
    monkeypatch.setattr(ingestion_tasks, "engine", db_engine)
    monkeypatch.setattr(ingestion_tasks, "_redis", FakeRedis())
    monkeypatch.setattr(ingestion_tasks, "SUMMARY_LOOKAHEAD", 2)
    monkeypatch.setattr(ingestion_tasks, "_summary_version", lambda: 1)
    with db_engine.begin() as conn:
        conn.execute(text("INSERT INTO novels (id, title) VALUES (1, 'Main')"))
        for n in range(1, 7):
            conn.execute(text(
                "INSERT INTO chapters (novel_id, chapter_number, title, ingestion_status) "
                "VALUES (1, :n, :title, 'complete')"), {"n": n, "title": f"Chapter {n}"})
        conn.execute(text(
            "INSERT INTO reading_progress (novel_id, user_id, current_chapter) "
            "VALUES (1, 'default', 2)"))
    with db_engine.connect() as conn:  # the book's words, as ingestion records them
        for n in range(1, 7):
            ingestion_tasks._record_words(conn, 1, n, f"Chapter {n} Sunny waited by the gate.")
        conn.commit()
    return db_engine


@pytest.fixture
def queue(ingestion_tasks, monkeypatch):
    """Record the batches handed to the summaries queue instead of enqueueing them."""
    batches = []
    fake = type("Task", (), {"delay": staticmethod(lambda novel_id, chapters: batches.append(chapters))})
    monkeypatch.setattr(ingestion_tasks, "summarize_chapters", fake)
    return batches


def summarize(novel, chapter, version):
    with novel.begin() as conn:
        conn.execute(text(
            "INSERT INTO chapter_summaries (novel_id, chapter_number, summary, model, prompt_version) "
            "VALUES (1, :ch, 'old', 'm', :v)"), {"ch": chapter, "v": version})


# --- scheduling ---

def test_summaries_are_queued_up_to_the_reader_plus_the_lookahead(ingestion_tasks, novel, queue):
    result = ingestion_tasks.schedule_summaries(1)

    assert sorted(queue[0]) == [1, 2, 3, 4]
    assert result["up_to"] == 4


def test_a_reader_further_on_extends_the_horizon(ingestion_tasks, novel, queue):
    ingestion_tasks.schedule_summaries(1, reader_chapter=4)

    assert sorted(queue[0]) == [1, 2, 3, 4, 5, 6]


def test_the_chapters_just_read_are_summarized_first(ingestion_tasks, novel, queue):
    # newest first back from the reader, so a recap of the last few chapters works long
    # before chapter 1's summary is needed; the chapters ahead of the reader come last
    ingestion_tasks.schedule_summaries(1, reader_chapter=4)

    assert queue == [[4, 3, 2, 1, 5, 6]]


def test_queued_chapters_are_not_queued_again(ingestion_tasks, novel, queue):
    # every progress update schedules; while the model is off nothing completes
    ingestion_tasks.schedule_summaries(1)
    again = ingestion_tasks.schedule_summaries(1)

    assert queue == [[2, 1, 3, 4]]
    assert again["already_queued"] == 4


def test_summaries_from_an_older_prompt_are_written_again(ingestion_tasks, novel, queue):
    summarize(novel, 1, version=1)
    summarize(novel, 2, version=0)

    ingestion_tasks.schedule_summaries(1)

    assert queue == [[2, 3, 4]]


def test_work_is_split_into_batches(ingestion_tasks, novel, queue, monkeypatch):
    monkeypatch.setattr(ingestion_tasks, "SUMMARY_BATCH", 3)

    ingestion_tasks.schedule_summaries(1, up_to=6)

    assert queue == [[2, 1, 3], [4, 5, 6]]


# --- writing ---

@pytest.fixture
def model(ingestion_tasks, monkeypatch):
    """A served model that summarizes any chapter; records which chapters it was asked for."""
    asked, state = [], {"health": "ok"}

    def get(url, timeout=None):
        body = {"status": state["health"]} if url.endswith("/health") else {"prompt_version": 1}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    def post(url, json=None, timeout=None):
        asked.append(json["chapter_number"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "summary": f"In chapter {json['chapter_number']}, Sunny waits for Garcia.",
            "model": "m", "prompt_version": 1})

    monkeypatch.setattr(ingestion_tasks.httpx, "get", get)
    monkeypatch.setattr(ingestion_tasks.httpx, "post", post)
    monkeypatch.setattr(ingestion_tasks, "_chapter_text",
                        lambda novel_id, n: (f"Chapter {n}", "Sunny waited by the gate."))
    return asked, state


def stored(novel):
    with novel.connect() as conn:
        return conn.execute(text(
            "SELECT chapter_number, summary, prompt_version, unverified_names "
            "FROM chapter_summaries ORDER BY chapter_number")).all()


def test_summaries_are_stored_with_their_flags(ingestion_tasks, novel, model):
    result = ingestion_tasks.summarize_chapters(1, [1, 2])

    assert result["written"] == 2
    # "Garcia" is not in the chapter, so the summary is flagged
    assert stored(novel) == [(1, "In chapter 1, Sunny waits for Garcia.", 1, ["Garcia"]),
                             (2, "In chapter 2, Sunny waits for Garcia.", 1, ["Garcia"])]


def test_a_task_that_runs_twice_asks_the_model_once(ingestion_tasks, novel, model):
    asked, _ = model

    ingestion_tasks.summarize_chapters(1, [1, 2])
    ingestion_tasks.summarize_chapters(1, [1, 2])

    assert sorted(asked) == [1, 2]


def test_summaries_wait_while_the_model_is_not_served(ingestion_tasks, novel, model):
    asked, state = model
    state["health"] = "degraded"

    with pytest.raises(Retry):
        ingestion_tasks.summarize_chapters(1, [1, 2])

    assert asked == []
    assert stored(novel) == []


def test_a_deleted_novel_is_skipped_without_touching_its_collection(ingestion_tasks, novel, model,
                                                                    monkeypatch):
    def refuse(novel_id, n):
        raise AssertionError("opening a deleted novel's collection would recreate it")

    monkeypatch.setattr(ingestion_tasks, "_chapter_text", refuse)
    with novel.begin() as conn:
        for table in ("reading_progress", "chapters", "novels"):
            column = "id" if table == "novels" else "novel_id"
            conn.execute(text(f"DELETE FROM {table} WHERE {column} = 1"))

    assert ingestion_tasks.summarize_chapters(1, [1])["status"] == "skipped"


def test_deleting_a_novel_deletes_its_summaries(novel):
    summarize(novel, 1, version=1)

    with novel.begin() as conn:
        conn.execute(text("DELETE FROM reading_progress WHERE novel_id = 1"))
        conn.execute(text("DELETE FROM chapters WHERE novel_id = 1"))
        conn.execute(text("DELETE FROM novels WHERE id = 1"))

    assert stored(novel) == []
