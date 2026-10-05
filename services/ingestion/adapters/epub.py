"""EPUB source adapter.

Reading order comes from the spine, and chapter boundaries from the table of contents.
Iterating the package's document list instead (what this did before) produced, on a real
Project Gutenberg book: every chapter number shifted by three (title page, subtitle page
and contents counted as chapters 1-3), the 2,911-word Gutenberg license stored as a
chapter, and file names for titles. The spoiler filter trusts chapter numbers, so they
have to mean the book's chapters.
"""

import re
from posixpath import normpath

from adapters.base import BaseAdapter, in_reading_order

# Contents-page titles that are never story text
_NOT_STORY = re.compile(
    r"\b(contents|table of contents|title page|cover|copyright|dedication|license|"
    r"project gutenberg|acknowledg(e)?ments?|about the author|also by|colophon|imprint|"
    r"index|bibliography|footnotes|endnotes|transcriber)\b",
    re.IGNORECASE,
)
# Titles that number a chapter: "Chapter 3", "CHAPTER IV", "3.", "IV.", "Part Two"
_NUMBERED = re.compile(
    r"^\s*(chapter|part|book)\s+([0-9]+|[ivxlcdm]+|[a-z]+)\b|^\s*([0-9]+|[ivxlcdm]+)\s*[.:)]",
    re.IGNORECASE,
)
_STORY_BOOKENDS = re.compile(r"^\s*(prologue|epilogue)\b", re.IGNORECASE)
_BODY = re.compile(r"<body[^>]*>(.*)</body>", re.IGNORECASE | re.DOTALL)
# Project Gutenberg brackets every book with these markers, a published convention meant
# for exactly this: without the clip, the license preamble ended up in the last chapter.
_PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
                       re.IGNORECASE)
_PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
# a short entry that survives the filters is a title or half-title page, not a chapter
_MIN_WORDS = 100


def _flatten(toc) -> list[tuple[str, str]]:
    """(title, href) for every contents entry, depth-first in reading order."""
    entries = []
    for entry in toc:
        if isinstance(entry, tuple):  # (section, children)
            section, children = entry
            if getattr(section, "href", None):
                entries.append((section.title, section.href))
            entries.extend(_flatten(children))
        else:
            entries.append((entry.title, entry.href))
    return entries


def _word_count(html: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", html).split())


class EpubAdapter(BaseAdapter):
    """Reads chapters from an EPUB file."""

    def __init__(self, source_path: str):
        self.path = source_path

    def fetch_all(self, max_chapters: int | None = None) -> list[dict]:
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ImportError("Install ebooklib: pip install ebooklib") from None

        book = epub.read_epub(self.path)

        # 1. The text of the book in reading order, remembering where each file starts.
        text, starts = [], {}
        offset = 0
        for idref, linear in book.spine:
            item = book.get_item_with_id(idref)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            if str(linear).lower() == "no" or isinstance(item, epub.EpubNav):
                continue  # auxiliary content and the navigation document itself
            html = item.get_content().decode("utf-8", errors="ignore")
            body = _BODY.search(html)
            html = body.group(1) if body else html
            starts[normpath(item.get_name())] = offset
            text.append(html)
            offset += len(html)
        full = "".join(text)

        # 2. Where each contents entry begins in that text.
        positions = []
        for title, href in _flatten(book.toc):
            path, _, fragment = (href or "").partition("#")
            base = starts.get(normpath(path))
            if base is None:
                continue  # points outside the reading order
            pos = base
            if fragment:
                anchor = re.search(rf"""\bid\s*=\s*["']{re.escape(fragment)}["']""",
                                   full[base:])
                if anchor:
                    pos = full.rfind("<", base, base + anchor.start()) if anchor.start() else base
                    pos = base if pos < 0 else pos
            positions.append((pos, (title or "").strip()))
        positions.sort()

        # 3. Each entry runs until the next one, clipped to the publisher's own story
        #    markers when present; then keep only story.
        start_marker, end_marker = _PG_START.search(full), _PG_END.search(full)
        lo = start_marker.end() if start_marker else 0
        hi = full.rfind("<", 0, end_marker.start()) if end_marker else len(full)
        hi = len(full) if hi < 0 else hi
        spans = []
        for i, (pos, title) in enumerate(positions):
            stop = positions[i + 1][0] if i + 1 < len(positions) else len(full)
            spans.append((title, full[max(pos, lo): min(stop, hi)]))
        story = self._story(spans)

        chapters = [
            {"number": n, "title": title, "content": html, "volume": 1}
            for n, (title, html) in enumerate(story, 1)
        ]
        return in_reading_order(chapters, max_chapters)

    @staticmethod
    def _story(spans: list[tuple[str, str]]) -> list[tuple[str, str]]:
        spans = [(t, h) for t, h in spans if not _NOT_STORY.search(t) and h.strip()]

        numbered = [i for i, (t, _) in enumerate(spans) if _NUMBERED.search(t)]
        if len(numbered) >= 2:
            # The book numbers its chapters: what precedes the first numbered chapter and
            # follows the last is front and back matter, except a prologue or epilogue.
            first, last = numbered[0], numbered[-1]
            if first > 0 and _STORY_BOOKENDS.search(spans[first - 1][0]):
                first -= 1
            if last + 1 < len(spans) and _STORY_BOOKENDS.search(spans[last + 1][0]):
                last += 1
            spans = spans[first: last + 1]

        return [(t, h) for t, h in spans if _word_count(h) >= _MIN_WORDS]
