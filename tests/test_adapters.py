"""Source adapters decide chapter numbers, and the spoiler filter trusts chapter numbers.

Shadow Slave's crawl happened to zero-pad file names and carry a `serial` field, which hid
an ordering bug that any other source would hit.
"""

import json
import logging

import pytest
from conftest import load_module


@pytest.fixture(scope="module")
def adapter_module():
    return load_module("ingestion", "adapters/local_json.py", alias="ingestion_local_json")


def body(n):
    return f"<p>{f'Text of chapter {n}. ' * 10}</p>"


def write_files(root, names, serials=None):
    for i, name in enumerate(names):
        record = {"title": f"Chapter {name}", "content": body(name)}
        if serials is not None:
            record["serial"] = serials[i]
        path = root / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))


def load(adapter_module, path, **kwargs):
    return adapter_module.LocalJsonAdapter(str(path)).fetch_all(**kwargs)


def test_unpadded_file_names_keep_reading_order(adapter_module, tmp_path):
    # Alphabetical order stored chapter 10's text as chapter 2
    write_files(tmp_path, ["1", "2", "3", "10", "11", "20"])

    chapters = load(adapter_module, tmp_path)

    assert [(c["number"], c["title"]) for c in chapters] == [
        (1, "Chapter 1"), (2, "Chapter 2"), (3, "Chapter 3"),
        (4, "Chapter 10"), (5, "Chapter 11"), (6, "Chapter 20"),
    ]


def test_first_n_chapters_means_first_in_reading_order(adapter_module, tmp_path):
    write_files(tmp_path, ["1", "2", "3", "10", "11", "20"])

    titles = [c["title"] for c in load(adapter_module, tmp_path, max_chapters=3)]

    assert titles == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_serial_numbers_win_over_file_names(adapter_module, tmp_path):
    write_files(tmp_path, ["a", "b", "c"], serials=[3, 1, 2])

    assert [c["title"] for c in load(adapter_module, tmp_path)] == ["Chapter b", "Chapter c", "Chapter a"]


def test_lightnovel_crawler_layout(adapter_module, tmp_path):
    # the layout Shadow Slave was crawled into: zero-padded files in numbered folders
    write_files(tmp_path, ["001/00001", "001/00002", "002/00101"], serials=[1, 2, 101])
    (tmp_path / "meta.json").write_text("{}")

    assert [c["number"] for c in load(adapter_module, tmp_path)] == [1, 2, 101]


def test_duplicate_chapter_numbers_are_rejected(adapter_module, tmp_path):
    # e.g. a source that restarts numbering each volume: one chapter would overwrite another
    write_files(tmp_path, ["v1c1", "v2c1"], serials=[1, 1])

    with pytest.raises(ValueError, match="number 1"):
        load(adapter_module, tmp_path)


def test_combined_file_with_restarted_numbering_is_rejected(adapter_module, tmp_path):
    source = tmp_path / "novel.json"
    source.write_text(json.dumps([
        {"number": n, "title": f"Vol {v} Ch {n}", "content": body(n)}
        for v in (1, 2) for n in (1, 2)
    ]))

    with pytest.raises(ValueError, match="unique"):
        load(adapter_module, source)


def test_unreadable_file_is_skipped_loudly(adapter_module, tmp_path, caplog):
    write_files(tmp_path, ["1", "2"])
    (tmp_path / "3.json").write_text("{not json")

    with caplog.at_level(logging.WARNING):
        chapters = load(adapter_module, tmp_path)

    assert len(chapters) == 2
    assert "3.json" in caplog.text
