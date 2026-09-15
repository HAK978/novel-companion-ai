"""Text cleaning and chunking — the code that decides what ever reaches the index."""

import pytest


def test_clean_html_strips_tags(chunking):
    raw = "<p>Sunny walked <em>slowly</em> into the <strong>Dark City</strong>.</p>"
    assert chunking.clean_html(raw) == "Sunny walked slowly into the Dark City."


def test_clean_html_collapses_whitespace(chunking):
    raw = "<div>line one\n\n\n   line two\t\ttabbed</div>"
    assert chunking.clean_html(raw) == "line one line two tabbed"


@pytest.mark.parametrize("empty", ["", None])
def test_clean_html_handles_empty_input(chunking, empty):
    assert chunking.clean_html(empty) == ""


def test_chunk_text_drops_fragments_below_minimum(chunking):
    # Chunks shorter than 50 characters are noise (stray headings, page numbers)
    assert chunking.chunk_text("Too short.") == []


def test_chunk_text_splits_long_text(chunking):
    text = " ".join(f"word{i}." for i in range(1000))
    chunks = chunking.chunk_text(text, chunk_size=200)

    assert len(chunks) > 1
    assert all(len(c) >= 50 for c in chunks)


def test_chunk_text_prefers_sentence_boundaries(chunking):
    # 400 words of full sentences; each chunk should land on a sentence end
    sentence = "The shadow moved across the broken street and vanished. "
    chunks = chunking.chunk_text(sentence * 100, chunk_size=100)

    assert len(chunks) > 1
    assert all(c.rstrip().endswith(".") for c in chunks[:-1])


def test_chunk_text_preserves_content(chunking):
    text = " ".join(f"token{i}" for i in range(500)) + "."
    chunks = chunking.chunk_text(text, chunk_size=100)

    rejoined = " ".join(chunks)
    # Every token survives chunking; nothing is silently dropped mid-document
    assert "token0" in rejoined
    assert "token499" in rejoined
    assert rejoined.count("token250") == 1


def test_chunk_size_is_respected(chunking):
    text = " ".join(f"word{i}." for i in range(600))

    small = chunking.chunk_text(text, chunk_size=100)
    large = chunking.chunk_text(text, chunk_size=400)

    assert len(small) > len(large)
