

class BaseAdapter:
    """Base interface for all source adapters."""

    def fetch_chapter_list(self) -> list[dict]:
        """Return list of available chapters with metadata."""
        raise NotImplementedError

    def fetch_chapter(self, chapter_number: int) -> dict:
        """Return a single chapter: {"number": int, "title": str, "content": str, "volume": int}"""
        raise NotImplementedError

    def fetch_all(self, max_chapters: int | None = None) -> list[dict]:
        """Fetch all chapters up to max_chapters."""
        raise NotImplementedError
