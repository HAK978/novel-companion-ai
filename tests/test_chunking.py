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


# --- HTML from sources other than the one crawl this was first built against ---


def test_script_and_style_contents_are_dropped(chunking):
    raw = "<style>.ad{color:red}</style><script>var tracker = load();</script><p>Elena drew her sword.</p>"
    assert chunking.clean_html(raw) == "Elena drew her sword."


def test_paragraphs_do_not_fuse(chunking):
    # 'himself.</p><p>After' used to become 'himself.After'
    assert chunking.clean_html("<p>He sighed.</p><p>After that, silence.</p>") == (
        "He sighed. After that, silence."
    )


def test_line_breaks_separate_words(chunking):
    assert chunking.clean_html("first line<br>second line<br/>third") == "first line second line third"


def test_inline_tags_do_not_split_words(chunking):
    assert chunking.clean_html("un<em>believ</em>able <a href='#'>link</a>") == "unbelievable link"


def test_entities_are_decoded(chunking):
    raw = "<p>Tom &amp; Jerry&#8217;s &quot;house&quot;</p>"
    assert chunking.clean_html(raw) == "Tom & Jerry\u2019s \"house\""


def test_escaped_markup_stays_text(chunking):
    assert chunking.clean_html("<p>Type &lt;b&gt; for bold</p>") == "Type <b> for bold"


def test_comments_are_dropped(chunking):
    assert chunking.clean_html("<p>Before<!-- ad slot --> after</p>") == "Before after"
