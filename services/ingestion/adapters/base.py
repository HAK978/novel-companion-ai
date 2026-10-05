

class BaseAdapter:
    """Base interface for all source adapters."""

    def fetch_all(self, max_chapters: int | None = None) -> list[dict]:
        """Return chapters in reading order: {"number", "title", "content", "volume"}."""
        raise NotImplementedError


def in_reading_order(chapters: list[dict], max_chapters: int | None) -> list[dict]:
    """Sort by chapter number, reject duplicates, then take the first `max_chapters`.

    Chapter numbers define reading order for the spoiler filter, so two chapters sharing a
    number would let one silently overwrite the other.
    """
    chapters.sort(key=lambda ch: ch["number"])
    seen: dict[int, str] = {}
    for ch in chapters:
        if ch["number"] in seen:
            raise ValueError(
                f"two chapters claim number {ch['number']} ({seen[ch['number']]!r} and "
                f"{ch['title']!r}). Chapter numbers define reading order for the spoiler "
                f"filter, so they must be unique; a source that restarts numbering each "
                f"volume needs renumbering before ingestion."
            )
        seen[ch["number"]] = ch["title"]
    return chapters[:max_chapters] if max_chapters else chapters
