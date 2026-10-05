import json
import logging
import re
from pathlib import Path

from adapters.base import BaseAdapter

log = logging.getLogger(__name__)


def _natural_key(path: Path, root: Path) -> list:
    """Order '2.json' before '10.json' by comparing digit runs as numbers.

    Plain string sorting put chapter 10 second; with no `serial` field to correct it, chapter
    10's text was stored as chapter 2, and the spoiler filter trusts chapter numbers.
    """
    relative = path.relative_to(root).as_posix().lower()
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", relative)]


def _in_reading_order(chapters: list[dict], max_chapters: int | None) -> list[dict]:
    """Sort by chapter number, reject duplicates, then take the first `max_chapters`."""
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


class LocalJsonAdapter(BaseAdapter):
    """Reads chapter JSON files from a local directory or a combined JSON file."""

    def __init__(self, source_path: str):
        self.path = Path(source_path)

    def fetch_all(self, max_chapters: int | None = None) -> list[dict]:
        if self.path.is_file():
            return self._load_combined_json(max_chapters)
        elif self.path.is_dir():
            return self._load_chapter_files(max_chapters)
        else:
            raise FileNotFoundError(f"Source path not found: {self.path}")

    def _load_chapter_files(self, max_chapters: int | None = None) -> list[dict]:
        """Load one-chapter-per-file JSON from a directory tree (lightnovel-crawler layout)."""
        files = sorted(
            (f for f in self.path.rglob("*.json") if f.name != "meta.json"),
            key=lambda f: _natural_key(f, self.path),
        )

        chapters = []
        for position, f in enumerate(files, 1):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("skipping unreadable chapter file %s: %s", f, exc)
                continue
            content = data.get("content") or data.get("body") or ""
            if len(content) < 100:
                continue
            serial = data.get("serial")
            chapters.append({
                # the source's own numbering when it has one, else natural file order
                "number": int(serial) if serial is not None else position,
                "title": data.get("title") or f"Chapter {position}",
                "content": content,
                "volume": data.get("volume", 1),
            })
        return _in_reading_order(chapters, max_chapters)

    def _load_combined_json(self, max_chapters: int | None = None) -> list[dict]:
        """Load from a single JSON file containing all chapters."""
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array format and {"chapters": [...]} format
        if isinstance(data, list):
            raw_chapters = data
        elif isinstance(data, dict) and "chapters" in data:
            raw_chapters = data["chapters"]
        else:
            raise ValueError("JSON must be an array or have a 'chapters' key")

        chapters = []
        for position, ch in enumerate(raw_chapters, 1):
            number = ch.get("number")
            chapters.append({
                "number": int(number) if number is not None else position,
                "title": ch.get("title") or f"Chapter {position}",
                "content": ch.get("body") or ch.get("content") or "",
                "volume": ch.get("volume", 1),
            })
        return _in_reading_order(chapters, max_chapters)
