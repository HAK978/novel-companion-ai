"""
Bulk ingest chapter JSON files into the Novel Companion AI system.

Usage:
    python seed_chapters.py /path/to/json/folder [--max-chapters 100] [--gateway http://localhost:8000]
"""

import json
import sys
import time
import argparse
from pathlib import Path

import httpx


def seed(json_folder: str, gateway_url: str, max_chapters: int = None):
    folder = Path(json_folder)
    json_files = sorted(folder.glob("*.json"))

    if max_chapters:
        json_files = json_files[:max_chapters]

    print(f"Found {len(json_files)} chapter files")
    print(f"Gateway: {gateway_url}")
    print()

    task_ids = []

    for i, file_path in enumerate(json_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            payload = {
                "number": i,
                "title": data.get("title", f"Chapter {i}"),
                "content": data.get("body", ""),
                "volume": data.get("volume", 1),
            }

            resp = httpx.post(f"{gateway_url}/ingest", json=payload, timeout=10)
            result = resp.json()
            task_ids.append(result.get("task_id"))

            if i % 50 == 0:
                print(f"  Submitted {i}/{len(json_files)}")

        except Exception as e:
            print(f"  Error on chapter {i}: {e}")

    print(f"\nSubmitted {len(task_ids)} chapters for ingestion")
    print("Chapters are being processed by Celery workers in the background.")
    print(f"Check status: GET {gateway_url}/ingest/status/<task_id>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed chapters into Novel Companion AI")
    parser.add_argument("json_folder", help="Path to folder containing chapter JSON files")
    parser.add_argument("--max-chapters", type=int, default=None, help="Limit number of chapters")
    parser.add_argument("--gateway", default="http://localhost:8000", help="Gateway URL")
    args = parser.parse_args()

    seed(args.json_folder, args.gateway, args.max_chapters)
