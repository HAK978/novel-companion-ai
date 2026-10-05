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


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into chunks, preserving sentence boundaries."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)

        if len(current_chunk) >= chunk_size:
            chunk_text = " ".join(current_chunk)
            # Try to break at a sentence boundary
            last_end = max(
                chunk_text.rfind("."),
                chunk_text.rfind("!"),
                chunk_text.rfind("?"),
            )

            if last_end > len(chunk_text) * 0.7:
                chunks.append(chunk_text[: last_end + 1].strip())
                current_chunk = chunk_text[last_end + 1 :].strip().split()
            else:
                chunks.append(chunk_text)
                current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if len(c.strip()) >= 50]
