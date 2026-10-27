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


def test_a_short_chapter_ending_is_kept(chunking):
    # pieces under 50 characters were dropped as noise: Shadow Slave's chapter 331 lost its
    # last line, "…The Spire Messenger was dead."
    text = " ".join(f"Sentence {i} ends here." for i in range(100)) + " He was dead."

    chunks = chunking.chunk_text(text, chunk_size=400)

    assert chunks[-1].endswith("He was dead.")
    assert chunking.chunk_text("Too short.") == ["Too short."]


def test_chunks_never_lose_or_repeat_text(chunking):
    # every length up to two full chunks, so every kind of ending comes up
    words = [f"w{i}" + ("." if i % 7 == 3 else "") for i in range(900)]
    for n in range(1, 901):
        text = " ".join(words[:n])
        assert " ".join(chunking.chunk_text(text, 400)) == text, n


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


@pytest.mark.parametrize("pieces", [2, 3])
def test_windows_rebuild_their_chunk_when_the_last_is_short(chunking, pieces):
    # a tail under 50 characters joins the window before it, without losing or repeating text
    full = " ".join(f"w{i}." for i in range(10 * (pieces - 1)))
    chunk = f"{full} tail."
    windows = chunking.split_windows(chunk, 10)

    assert len(windows) == pieces - 1
    assert " ".join(windows) == chunk


def test_windows_rebuild_their_chunk(chunking):
    chunk = " ".join(f"Sentence {i} ends here." for i in range(120))
    windows = chunking.split_windows(chunk, 170)

    assert len(windows) > 1
    assert " ".join(windows) == chunk
    assert all(len(w.split()) <= 170 for w in windows)


def test_windows_never_lose_or_repeat_text(chunking):
    # every chunk length up to two full chunks, so every kind of tail comes up
    words = [f"w{i}" + ("." if i % 7 == 3 else "") for i in range(800)]
    for n in range(1, 801):
        chunk = " ".join(words[:n])
        assert " ".join(chunking.split_windows(chunk, 170)) == chunk, n


def test_windows_the_model_cannot_read_whole_are_split_again(chunking):
    # 170 words nearly always fit the model's 256 tokens, but names and curly quotes take
    # extra tokens: 47 of Shadow Slave's 28,751 windows ran over
    chunk = " ".join(f"Sentence {i} ends here." for i in range(40))

    def fits(text):  # a model that reads 25 words
        return len(text.split()) <= 25

    windows = chunking.split_windows(chunk, 170, fits)

    assert all(fits(w) for w in windows)
    assert " ".join(windows) == chunk
    assert all(w.endswith(".") for w in windows)  # cut at sentence ends


def test_a_window_without_sentence_ends_is_split_between_words(chunking):
    chunk = " ".join(f"w{i}" for i in range(60))

    windows = chunking.split_windows(chunk, 170, lambda text: len(text.split()) <= 20)

    assert " ".join(windows) == chunk
    assert all(len(w.split()) <= 20 for w in windows)

