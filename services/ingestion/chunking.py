import html
import re

# Elements whose contents are never prose. Dropping only their tags used to leave their
# code behind as "chapter text": '<script>var t = load();</script>' became 'var t = load();'.
_INVISIBLE = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Block boundaries become spaces so paragraphs don't fuse ('end.</p><p>Next' was
# 'end.Next'); inline tags (<em>, <a>, <span>) vanish without one so words stay whole.
_BLOCK = re.compile(
    r"</?(?:p|div|br|hr|h[1-6]|li|ul|ol|tr|td|th|table|blockquote|section|article|"
    r"header|footer|pre)\b[^>]*>",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")


def clean_html(html_content: str) -> str:
    if not html_content:
        return ""
    text = _COMMENT.sub(" ", html_content)
    text = _INVISIBLE.sub(" ", text)
    text = _BLOCK.sub(" ", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)  # after tag removal, so escaped text like &lt;b&gt; stays text
    return re.sub(r"\s+", " ", text).strip()


# A word that ends a sentence, closing quotes or brackets included: 'end.', 'said."', 'why?)'
_ENDS_SENTENCE = re.compile(r"[.!?][\"'\u201d\u2019)\]]*$")


def _split(text: str, size: int) -> list[str]:
    """Pieces of about `size` words, each cut after the last word ending a sentence in its
    final 30% when there is one. Pieces end at whole words, so joined with spaces they give
    the text back. (Cutting at the last "." character used to split 'said."' in two.)"""
    pieces, current = [], []
    for word in text.split():
        current.append(word)
        if len(current) >= size:
            cut = next((i for i in range(len(current), int(len(current) * 0.7), -1)
                        if _ENDS_SENTENCE.search(current[i - 1])), len(current))
            pieces.append(" ".join(current[:cut]))
            current = current[cut:]
    if current:
        pieces.append(" ".join(current))
    return pieces


def _merge_short_tail(pieces: list[str]) -> list[str]:
    """A last piece under 50 characters joins the one before it, so nothing is lost."""
    if len(pieces) > 1 and len(pieces[-1]) < 50:
        tail = pieces.pop()  # popped first: popping inside the assignment shifted its target
        pieces[-1] = f"{pieces[-1]} {tail}"
    return pieces


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into chunks of about `chunk_size` words at sentence ends; joined with spaces
    they give the text back. A short ending joins the chunk before it: it used to be dropped
    as noise, which cut the last line off 13 Shadow Slave chapters ("…The Spire Messenger
    was dead.")."""
    return _merge_short_tail(_split(text, chunk_size))


def _fit(window: str, fits) -> list[str]:
    """`window`, split at the sentence end nearest its middle (or the middle word) until every
    piece fits."""
    words = window.split()
    if fits(window) or len(words) < 2:
        return [window]
    middle = len(words) // 2
    ends = [i for i in range(1, len(words)) if _ENDS_SENTENCE.search(words[i - 1])]
    cut = min(ends, key=lambda i: abs(i - middle)) if ends else middle
    return _fit(" ".join(words[:cut]), fits) + _fit(" ".join(words[cut:]), fits)


def split_windows(chunk: str, size: int, fits=None) -> list[str]:
    """Split a chunk into windows of about `size` words for the embedding model, which reads
    only the start of a long text. `fits(text)` says whether the model reads a text whole; a
    window that does not fit is split again, since a word count cannot promise a token count.
    Nothing is dropped: the windows joined with spaces give the chunk back."""
    windows = _merge_short_tail(_split(chunk, size))
    return [piece for window in windows for piece in _fit(window, fits)] if fits else windows
