import json
from pathlib import Path
from typing import List, Optional
from adapters.base import BaseAdapter


class LocalJsonAdapter(BaseAdapter):
    """Reads chapter JSON files from a local directory or a combined JSON file."""

    def __init__(self, source_path: str):
        self.path = Path(source_path)

    def fetch_all(self, max_chapters: Optional[int] = None) -> List[dict]:
        if self.path.is_file():
            return self._load_combined_json(max_chapters)
        elif self.path.is_dir():
            return self._load_chapter_files(max_chapters)
        else:
            raise FileNotFoundError(f"Source path not found: {self.path}")

    def _load_chapter_files(self, max_chapters: Optional[int] = None) -> List[dict]:
        """Load individual chapter JSON files from a directory (including subdirectories)."""
        # Search recursively for JSON files, exclude meta.json
        files = sorted(
            f for f in self.path.rglob("*.json")
            if f.name != "meta.json"
        )
        if max_chapters:
            files = files[:max_chapters]

        chapters = []
        for i, f in enumerate(files, 1):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                content = data.get("content", data.get("body", ""))
                if not content or len(content) < 100:
                    continue
                chapters.append({
                    "number": data.get("serial", i),
                    "title": data.get("title", f"Chapter {i}"),
                    "content": content,
                    "volume": data.get("volume", 1),
                })
            except Exception:
                continue
        return chapters

    def _load_combined_json(self, max_chapters: Optional[int] = None) -> List[dict]:
        """Load from a single JSON file containing all chapters."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array format and {"chapters": [...]} format
        if isinstance(data, list):
            raw_chapters = data
        elif isinstance(data, dict) and "chapters" in data:
            raw_chapters = data["chapters"]
        else:
            raise ValueError("JSON must be an array or have a 'chapters' key")

        if max_chapters:
            raw_chapters = raw_chapters[:max_chapters]

        chapters = []
        for i, ch in enumerate(raw_chapters, 1):
            chapters.append({
                "number": ch.get("number", i),
                "title": ch.get("title", f"Chapter {i}"),
                "content": ch.get("body", ch.get("content", "")),
                "volume": ch.get("volume", 1),
            })
        return chapters
