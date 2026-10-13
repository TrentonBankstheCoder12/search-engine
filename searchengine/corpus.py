"""Load documents from JSON Lines files: one {"id", "title", "text"} object per line."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

BUNDLED_CORPUS = Path(__file__).parent / "data" / "corpus.jsonl"


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            doc = json.loads(line)
            if "id" not in doc or "text" not in doc:
                raise ValueError(f"{path}:{line_no}: each document needs 'id' and 'text'")
            yield doc
