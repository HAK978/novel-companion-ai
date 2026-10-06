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


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into chunks, preserving sentence boundaries."""
    return [c for c in _split(text, chunk_size) if len(c.strip()) >= 50]


def split_windows(chunk: str, size: int) -> list[str]:
    """Split a chunk into windows of about `size` words for the embedding model, which reads
    only the start of a long text. Nothing is dropped: a short tail joins the window before
    it, so the windows joined with spaces give the chunk back."""
    windows = _split(chunk, size)
    if len(windows) > 1 and len(windows[-1]) < 50:
        tail = windows.pop()  # popped first: popping inside the assignment shifted its target
        windows[-1] = f"{windows[-1]} {tail}"
    return windows
