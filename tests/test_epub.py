"""EPUB chapters must be the book's chapters, in reading order.

Run against a real Project Gutenberg book, the old reader shifted every chapter number by
three (title, subtitle and contents pages counted as chapters), stored the Gutenberg
license as a chapter, and used file names as titles. These tests build small EPUBs that
reproduce each structure.
"""

import re

import pytest
from conftest import load_module
from ebooklib import epub


@pytest.fixture(scope="module")
def reader():
    return load_module("ingestion", "adapters/epub.py", alias="ingestion_epub")


def prose(tag, words=150):
    """Body text whose every word identifies where it came from: tag_0, tag_1, ..."""
    return f"<p>{' '.join(f'{tag}_{i}' for i in range(words))}.</p>"


def make_epub(path, files, toc, spine, add_order=None, nonlinear=()):
    """files: {name: body html}; toc: [(title, href)]; spine: [names] in reading order."""
    book = epub.EpubBook()
    book.set_identifier("test-book")
    book.set_title("Test Book")
    book.set_language("en")
    items = {}
    for name in add_order or files:  # manifest order can differ from the spine
        item = epub.EpubHtml(title=name, file_name=name, lang="en")
        item.content = f"<html><body>{files[name]}</body></html>"
        book.add_item(item)
        items[name] = item
    book.toc = [epub.Link(href, title, f"toc{i}") for i, (title, href) in enumerate(toc)]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + [
        (items[n], "no") if n in nonlinear else items[n] for n in spine
    ]
    epub.write_epub(str(path), book)
    return str(path)


def chapters(reader, path, **kwargs):
    return reader.EpubAdapter(path).fetch_all(**kwargs)


def words_of(chapter):
    """The prose words in a chapter, read from its text, never its markup."""
    text = re.sub(r"<[^>]+>", " ", chapter["content"])
    return {w.strip(".") for w in text.split() if "_" in w}


def test_reading_order_comes_from_the_spine(reader, tmp_path):
    files = {f"c{i}.xhtml": f"<h1 id='c{i}'>Chapter {i}</h1>{prose(f'ch{i}w')}" for i in (1, 2, 3)}
    path = make_epub(tmp_path / "b.epub", files,
                     toc=[(f"Chapter {i}", f"c{i}.xhtml#c{i}") for i in (1, 2, 3)],
                     spine=["c1.xhtml", "c2.xhtml", "c3.xhtml"],
                     add_order=["c3.xhtml", "c1.xhtml", "c2.xhtml"])  # manifest shuffled

    assert [c["title"] for c in chapters(reader, path)] == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_front_and_back_matter_are_not_chapters(reader, tmp_path):
    files = {
        "title.xhtml": f"<h1 id='t'>The Test Book</h1>{prose('titlepage')}",
        "contents.xhtml": f"<h1 id='toc'>Contents</h1>{prose('contents')}",
        "c1.xhtml": f"<h1 id='c1'>Chapter 1. Arrival</h1>{prose('one')}",
        "c2.xhtml": f"<h1 id='c2'>Chapter 2. Departure</h1>{prose('two')}",
        "license.xhtml": f"<h1 id='lic'>The Full Project Gutenberg License</h1>{prose('legal')}",
    }
    path = make_epub(tmp_path / "b.epub", files, toc=[
        ("The Test Book", "title.xhtml#t"), ("Contents", "contents.xhtml#toc"),
        ("Chapter 1. Arrival", "c1.xhtml#c1"), ("Chapter 2. Departure", "c2.xhtml#c2"),
        ("The Full Project Gutenberg License", "license.xhtml#lic"),
    ], spine=list(files))

    result = chapters(reader, path)

    # numbered from the book's first chapter, not shifted by the pages before it
    assert [(c["number"], c["title"]) for c in result] == [
        (1, "Chapter 1. Arrival"), (2, "Chapter 2. Departure")]


def test_several_chapters_in_one_file_are_split_at_their_anchors(reader, tmp_path):
    files = {"all.xhtml": "".join(f"<h2 id='c{i}'>Chapter {i}</h2>{prose(f'ch{i}w')}" for i in (1, 2, 3))}
    path = make_epub(tmp_path / "b.epub", files,
                     toc=[(f"Chapter {i}", f"all.xhtml#c{i}") for i in (1, 2, 3)], spine=["all.xhtml"])

    result = chapters(reader, path)

    assert len(result) == 3
    assert all(w.startswith("ch2w") for w in words_of(result[1]))  # no bleed from 1 or 3


def test_a_chapter_continued_in_the_next_file_stays_whole(reader, tmp_path):
    files = {
        "c1a.xhtml": f"<h1 id='c1'>Chapter 1</h1>{prose('firsthalf')}",
        "c1b.xhtml": prose("secondhalf"),  # no contents entry of its own
        "c2.xhtml": f"<h1 id='c2'>Chapter 2</h1>{prose('next')}",
    }
    path = make_epub(tmp_path / "b.epub", files,
                     toc=[("Chapter 1", "c1a.xhtml#c1"), ("Chapter 2", "c2.xhtml#c2")], spine=list(files))

    first = words_of(chapters(reader, path)[0])

    assert any(w.startswith("firsthalf") for w in first)
    assert any(w.startswith("secondhalf") for w in first)


def test_prologue_is_story_but_about_the_author_is_not(reader, tmp_path):
    files = {
        "pro.xhtml": f"<h1 id='p'>Prologue</h1>{prose('pro')}",
        "c1.xhtml": f"<h1 id='c1'>Chapter 1</h1>{prose('one')}",
        "c2.xhtml": f"<h1 id='c2'>Chapter 2</h1>{prose('two')}",
        "about.xhtml": f"<h1 id='a'>About the Author</h1>{prose('bio')}",
    }
    path = make_epub(tmp_path / "b.epub", files, toc=[
        ("Prologue", "pro.xhtml#p"), ("Chapter 1", "c1.xhtml#c1"),
        ("Chapter 2", "c2.xhtml#c2"), ("About the Author", "about.xhtml#a"),
    ], spine=list(files))

    assert [c["title"] for c in chapters(reader, path)] == ["Prologue", "Chapter 1", "Chapter 2"]


def test_gutenberg_boilerplate_is_clipped_from_the_last_chapter(reader, tmp_path):
    files = {
        "c1.xhtml": f"<h1 id='c1'>Chapter 1</h1>{prose('one')}",
        "c2.xhtml": f"<h1 id='c2'>Chapter 2</h1>{prose('two')}",
        "end.xhtml": "<p>*** END OF THE PROJECT GUTENBERG EBOOK TEST ***</p>"
                     f"{prose('preamble')}<h1 id='lic'>License</h1>{prose('legal')}",
    }
    path = make_epub(tmp_path / "b.epub", files, toc=[
        ("Chapter 1", "c1.xhtml#c1"), ("Chapter 2", "c2.xhtml#c2"), ("License", "end.xhtml#lic"),
    ], spine=list(files))

    result = chapters(reader, path)

    # neither attached to the final chapter nor stored as a chapter of its own
    assert not any("GUTENBERG" in c["content"].upper() for c in result)
    assert not any(w.startswith("preamble") for c in result for w in words_of(c))


def test_non_linear_items_are_skipped(reader, tmp_path):
    files = {
        "c1.xhtml": f"<h1 id='c1'>Chapter 1</h1>{prose('one')}",
        "notes.xhtml": f"<h1 id='n'>Chapter Notes</h1>{prose('note')}",
        "c2.xhtml": f"<h1 id='c2'>Chapter 2</h1>{prose('two')}",
    }
    path = make_epub(tmp_path / "b.epub", files, toc=[
        ("Chapter 1", "c1.xhtml#c1"), ("Chapter 2", "c2.xhtml#c2")],
        spine=list(files), nonlinear={"notes.xhtml"})

    result = chapters(reader, path)

    assert not any(w.startswith("note") for c in result for w in words_of(c))


def test_first_n_chapters(reader, tmp_path):
    files = {f"c{i}.xhtml": f"<h1 id='c{i}'>Chapter {i}</h1>{prose(f'ch{i}w')}" for i in range(1, 6)}
    path = make_epub(tmp_path / "b.epub", files,
                     toc=[(f"Chapter {i}", f"c{i}.xhtml#c{i}") for i in range(1, 6)], spine=list(files))

    assert [c["number"] for c in chapters(reader, path, max_chapters=2)] == [1, 2]
