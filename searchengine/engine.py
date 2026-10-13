"""The search engine: ties index, query parser, ranking and snippets together.

Search runs in two phases, the same split real engines use:

1. **Matching** (boolean): evaluate the query tree with set operations to
   decide *which* documents qualify. AND = intersection, OR = union,
   NOT = complement against all documents, phrases = position checks.
2. **Ranking**: score only the matching documents with BM25 and sort.

Keeping them separate means boolean logic decides membership exactly, while
BM25 only orders the results; a document can never sneak in on score alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .index import InvertedIndex
from .query import And, Node, Not, Or, Phrase, Term, parse, positive_terms
from .ranking import BM25, DEFAULT_B, DEFAULT_K1
from .snippets import make_snippet


@dataclass(frozen=True, slots=True)
class SearchResult:
    doc_id: str
    title: str
    score: float
    snippet: str


class SearchEngine:
    def __init__(
        self, index: InvertedIndex | None = None, k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> None:
        self.index = index or InvertedIndex()
        self.ranker = BM25(self.index, k1=k1, b=b)

    # --- document management (thin pass-throughs, kept for a tidy API) ----

    def add(self, doc_id: str, text: str, title: str = "") -> None:
        self.index.add(doc_id, text, title)

    def delete(self, doc_id: str) -> bool:
        return self.index.delete(doc_id)

    def save(self, path: str | Path) -> None:
        self.index.save(path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> SearchEngine:
        return cls(InvertedIndex.load(path), **kwargs)

    # --- searching ---------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        tree = parse(query)
        if tree is None:
            return []
        matches = self.match(tree)
        terms = positive_terms(tree)

        scored = [(self.ranker.score(doc_id, terms), doc_id) for doc_id in matches]
        # Highest score first; doc_id breaks ties so output is deterministic.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        results = []
        for score, doc_id in scored[:limit]:
            doc = self.index.docs[doc_id]
            results.append(SearchResult(doc_id, doc.title, round(score, 4), make_snippet(doc.text, terms)))
        return results

    def match(self, node: Node) -> set[str]:
        """Return the ids of documents satisfying the boolean query ``node``."""
        match node:
            case Term(term=t):
                return self.index.docs_with(t)
            case Phrase():
                return self._match_phrase(node)
            case And(children=cs):
                # Evaluate positive children first and intersect, smallest set
                # first; then subtract NOT children. This avoids materializing
                # "every document except X" for queries like `fraud NOT card`.
                positives = sorted((self.match(c) for c in cs if not isinstance(c, Not)), key=len)
                result = set.intersection(*positives) if positives else self.index.all_doc_ids()
                for c in cs:
                    if isinstance(c, Not):
                        result -= self.match(c.child)
                return result
            case Or(children=cs):
                return set().union(*(self.match(c) for c in cs))
            case Not(child=c):
                return self.index.all_doc_ids() - self.match(c)
        raise TypeError(f"unknown query node: {node!r}")

    def _match_phrase(self, phrase: Phrase) -> set[str]:
        """Documents where the terms appear at the expected relative positions.

        First intersect the posting lists (cheap) to get candidates containing
        every term, then for each candidate check whether some start position p
        has term i at p + offset_i for every i.
        """
        offsets = phrase.terms
        candidates = set.intersection(*(self.index.docs_with(t) for _, t in offsets))
        found = set()
        for doc_id in candidates:
            position_sets = [(off, set(self.index.positions(t, doc_id))) for off, t in offsets]
            first_off, first_term = offsets[0]
            for p in self.index.positions(first_term, doc_id):
                start = p - first_off
                if all(start + off in pos for off, pos in position_sets):
                    found.add(doc_id)
                    break
        return found
