"""The evaluation's scoring: what it may call a leak, what it leaves to a person, and how it
asks the app about recaps."""

import argparse
import importlib.util

import pytest
from conftest import ROOT

NO_JUDGE = argparse.Namespace(judge=False)
# first chapters of a few words of the Hound: "Garcia" only in the last chapter
BOOK = {"names": {"Stapleton": 6, "Beryl": 7, "Garcia": 15},
        "words": {"stapleton": 6, "beryl": 7, "garcia": 15, "underworld": 250,
                  "afterward": 250, "chapter": 1}}
SPOILER = {"id": "s1", "category": "spoiler", "question": "Who is the villain?",
           "current_chapter": 7}
RECAP = {"id": "r1", "category": "recap", "question": "What happened in the last 3 chapters?",
         "current_chapter": 12, "expected_range": [10, 12], "covers": ["Selden", "hut"]}


@pytest.fixture(scope="module")
def run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply(answer, sources=(), **more):
    return {"answer": answer, "sources": list(sources), **more}


def test_without_the_book_no_answer_passes_as_safe(run_eval):
    row = run_eval.score(NO_JUDGE, SPOILER, reply("His wife is Beryl Garcia."), [], None)
    spoilers = run_eval.summarize([row])["spoilers"]

    assert row["name_leak"] == "NOT CHECKED"
    assert spoilers["answers_naming_unread"] == spoilers["sources_naming_unread"] == "NOT CHECKED"


def test_a_name_from_a_later_chapter_is_a_leak(run_eval):
    row = run_eval.score(NO_JUDGE, {**SPOILER, "current_chapter": 14},
                         reply("She is his wife, Beryl Garcia."), [], BOOK)

    assert row["name_leak"] == "LEAK"
    assert row["unread_names"] == ["Garcia"]


def test_a_plot_point_about_a_known_name_is_left_to_a_person(run_eval):
    # "Stapleton is the villain" names no one unread, yet gives the ending away
    row = run_eval.score(NO_JUDGE, SPOILER, reply("Stapleton is the villain."), [], BOOK)
    spoilers = run_eval.summarize([row])["spoilers"]

    assert row["name_leak"] == "NONE FOUND"
    assert spoilers["answers_naming_unread"] == 0
    assert "not checked" in spoilers["plot_leaks"]
    assert spoilers["to_read"] == ["s1"]


def test_agreement_compares_names_found_with_any_hand_leak(run_eval):
    rows = [run_eval.score(NO_JUDGE, {**SPOILER, "id": i, "current_chapter": 14},
                           reply(answer), [], BOOK)
            for i, answer in (("a", "Beryl Garcia."), ("b", "Stapleton did it."), ("c", "No."))]
    labels = {"a": {"leak": "LEAK"}, "b": {"leak": "LEAK"}, "c": {"leak": "SAFE"}}

    report = run_eval.agreement(rows, labels)["code name_leak"]

    assert (report["agree"], report["of"]) == (2, 3)
    assert report["hand_leaks_found"] == "1 of 2"
    assert report["disagreements"] == ["b: code SAFE, hand LEAK"]


def test_unchecked_answers_are_left_out_of_agreement(run_eval):
    row = run_eval.score(NO_JUDGE, SPOILER, reply("Stapleton."), [], None)

    assert "code name_leak" not in run_eval.agreement([row], {"s1": {"leak": "SAFE"}})


def test_sources_are_checked_as_strictly_as_the_app_checks_summaries(run_eval):
    # a chapter summary is within the reader's chapters by number but can still name a
    # place the book only reaches later; a word opening a sentence is no name
    summary = {"chapter_number": 33, "text": "Afterward, Sunny explores the Underworld."}

    row = run_eval.score(NO_JUDGE, {**SPOILER, "current_chapter": 35}, reply("He rests.", [summary]),
                         [], BOOK)

    assert row["passages_within_progress"]
    assert row["sources_unread"] == ["Underworld"]
    assert run_eval.summarize([row])["spoilers"]["sources_naming_unread"] == 1


def test_a_recap_from_the_ranges_summaries_is_routed(run_eval):
    row = run_eval.score(NO_JUDGE, RECAP, reply("Selden fell and they shut the door.",
                                                route="chapter summaries", recap_range=[10, 12]),
                         [], BOOK)

    assert row["routed"]
    # "shut" is not the hut
    assert (row["covers_found"], row["covers_missing"]) == (["Selden"], ["hut"])
    assert run_eval.summarize([row])["recap"]["terms_mentioned"] == "1 of 2"


@pytest.mark.parametrize("route,recap_range", [
    ("passages", [10, 12]),               # answered from search
    ("chapter summaries", [3, 12]),       # from summaries, of the wrong chapters
    (None, None),                         # a chatbot, asked directly
])
def test_a_recap_from_anything_else_is_not_routed(run_eval, route, recap_range):
    row = run_eval.score(NO_JUDGE, RECAP, reply("Selden died.", route=route,
                                                recap_range=recap_range), [], BOOK)

    assert not row["routed"]


def test_each_recap_goes_to_its_endpoint_written_fresh(run_eval, monkeypatch):
    calls = []

    class Response:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    def post(url, json=None, **kwargs):
        calls.append((url.removeprefix("http://app"), json))
        return Response({"summary": "A recap.", "source": "chapter summaries",
                         "start_chapter": 4, "end_chapter": 6,
                         "summary_range": {"start": 4, "end": 14}})

    monkeypatch.setattr(run_eval.httpx, "post", post)
    args = argparse.Namespace(gateway="http://app", novel_id=4)

    summarized = run_eval.ask_app(args, {"endpoint": "summarize", "start_chapter": 4,
                                         "current_chapter": 6, "question": "Summarize."})
    caught_up = run_eval.ask_app(args, {"endpoint": "catch-me-up", "current_chapter": 14,
                                        "question": "Catch me up."})

    assert [path for path, _ in calls] == ["/summarize", "/progress", "/catch-me-up"]
    assert calls[0][1] == {"novel_id": 4, "start_chapter": 4, "end_chapter": 6,
                           "current_chapter": 6, "use_cache": False}
    assert calls[1][1]["current_chapter"] == 14
    assert calls[2][1]["use_cache"] is False
    assert summarized["recap_range"] == [4, 6]
    assert caught_up["recap_range"] == [4, 14]
    assert caught_up["route"] == "chapter summaries"


def test_a_sample_of_unflagged_answers_is_read_too(run_eval):
    # a plot leak can sit in an answer nothing flagged: three were in fact questions
    def run(word):
        return [run_eval.score(NO_JUDGE, {"id": f"f{i:02}", "category": "fact", "question": "q",
                                          "current_chapter": 7}, reply(f"{word} {i}."), [], BOOK)
                for i in range(20)]

    first = run_eval.summarize(run("Answer"))["spoilers"]["to_read"]

    assert len(first) == run_eval.UNFLAGGED_TO_READ
    assert run_eval.summarize(run("Answer"))["spoilers"]["to_read"] == first  # re-scored
    assert run_eval.summarize(run("Reply"))["spoilers"]["to_read"] != first  # another run

