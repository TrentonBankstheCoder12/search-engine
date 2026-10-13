"""Positional inverted index with incremental updates and JSON persistence.

Shape of the data:

    postings = {
        "ledger": {"doc-a": [3, 17], "doc-c": [0]},   # term -> doc -> positions
        "payment": {"doc-b": [5]},
    }

The *inverted* part (term -> documents) is what makes search fast: answering
"which documents contain 'ledger'?" is one dict lookup instead of a scan.
The *positional* part (the lists of word offsets) is what makes phrase
queries possible, and len(positions) doubles as the term frequency for BM25.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import Token, tokenize, word_spans

FORMAT_VERSION = 1


def _analyze(title: str, text: str) -> list[Token]:
    """Tokenize title and body into one positional stream.

    Title words are searchable just like body words. Body positions are shifted
    past the title plus a gap of one, so a phrase query can never match by
    straddling the end of the title and the start of the body.
    """
    offset = len(word_spans(title)) + 1 if title else 0
    body = [Token(t.term, t.position + offset, t.start, t.end) for t in tokenize(text)]
    return tokenize(title) + body


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    length: int  # number of indexed tokens (stopwords excluded); BM25's |D|


class InvertedIndex:
    def __init__(self) -> None:
        self.docs: dict[str, Document] = {}
        self.postings: dict[str, dict[str, list[int]]] = {}
        # Running total so the average document length is O(1) to read and
        # stays correct across adds and deletes.
        self._total_length = 0

    # --- updates ------------------------------------------------------------

    def add(self, doc_id: str, text: str, title: str = "") -> None:
        """Index a document. Re-adding an existing id replaces it (upsert)."""
        if doc_id in self.docs:
            self.delete(doc_id)

        tokens = _analyze(title, text)
        for tok in tokens:
            self.postings.setdefault(tok.term, {}).setdefault(doc_id, []).append(tok.position)

        self.docs[doc_id] = Document(doc_id, title, text, len(tokens))
        self._total_length += len(tokens)

    def delete(self, doc_id: str) -> bool:
        """Remove a document and every posting that points at it.

        We don't keep a forward index (doc -> terms); instead we re-tokenize the
        stored text to learn which posting lists to touch. That trades a little
        CPU on delete for not storing every term twice.
        """
        doc = self.docs.pop(doc_id, None)
        if doc is None:
            return False
        for term in {tok.term for tok in _analyze(doc.title, doc.text)}:
            plist = self.postings.get(term)
            if plist is None:
                continue
            plist.pop(doc_id, None)
            if not plist:
                # Drop empty lists so document frequency (len(plist)) stays exact.
                del self.postings[term]
        self._total_length -= doc.length
        return True

    # --- statistics used by ranking --------------------------------------------

    @property
    def doc_count(self) -> int:
        return len(self.docs)

    @property
    def avg_doc_length(self) -> float:
        return self._total_length / len(self.docs) if self.docs else 0.0

    def doc_freq(self, term: str) -> int:
        """Number of documents containing ``term`` (the n in BM25's IDF)."""
        return len(self.postings.get(term, {}))

    def term_freq(self, term: str, doc_id: str) -> int:
        return len(self.postings.get(term, {}).get(doc_id, ()))

    def positions(self, term: str, doc_id: str) -> list[int]:
        return self.postings.get(term, {}).get(doc_id, [])

    def docs_with(self, term: str) -> set[str]:
        return set(self.postings.get(term, {}))

    def all_doc_ids(self) -> set[str]:
        return set(self.docs)

    # --- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "docs": {
                d.doc_id: {"title": d.title, "text": d.text, "length": d.length} for d in self.docs.values()
            },
            "postings": self.postings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InvertedIndex:
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported index format version: {data.get('version')!r}")
        index = cls()
        for doc_id, d in data["docs"].items():
            index.docs[doc_id] = Document(doc_id, d["title"], d["text"], d["length"])
            index._total_length += d["length"]
        index.postings = {term: dict(plist) for term, plist in data["postings"].items()}
        return index

    def save(self, path: str | Path) -> None:
        """Write the index atomically: a crash mid-write never leaves a half file.

        We write to a temp file in the same directory, then ``os.replace`` it
        over the target, which is atomic on POSIX and Windows.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".index-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> InvertedIndex:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
