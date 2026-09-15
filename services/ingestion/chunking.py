import re


def clean_html(html_content: str) -> str:
    if not html_content:
        return ""
    text = re.sub(r"<[^>]+>", "", html_content)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
