from typing import List, Optional
from adapters.base import BaseAdapter


class EpubAdapter(BaseAdapter):
    """Parses EPUB ebook files into chapters."""

    def __init__(self, source_path: str):
        self.path = source_path

    def fetch_all(self, max_chapters: Optional[int] = None) -> List[dict]:
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ImportError("Install ebooklib: pip install ebooklib")

        book = epub.read_epub(self.path)
        chapters = []

        for i, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), 1):
            if max_chapters and i > max_chapters:
                break

            content = item.get_content().decode("utf-8", errors="ignore")

            # Skip very short items (likely metadata/TOC pages)
            if len(content) < 200:
                continue

            chapters.append({
                "number": len(chapters) + 1,
                "title": item.get_name() or f"Chapter {len(chapters) + 1}",
                "content": content,
                "volume": 1,
            })

        return chapters
