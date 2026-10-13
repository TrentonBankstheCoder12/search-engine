"""BM25 relevance scoring.

For a query Q and a document D:

    score(D, Q) = sum over terms t in Q of
                  IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))

    IDF(t) = ln(1 + (N - n + 0.5) / (n + 0.5))

where tf is how often t appears in D, |D| is D's length in tokens, avgdl is the
average length over the collection, N is the number of documents and n is the
number of documents that contain t.

The two tuning knobs:

* ``k1`` (term-frequency saturation, typically 1.2-2.0). The tf part rises
  quickly at first and then flattens toward k1 + 1, so the 10th occurrence of
  "ledger" adds far less than the 2nd. k1 = 0 ignores tf entirely (pure
  boolean presence); a very large k1 makes score grow almost linearly with tf,
  which rewards keyword stuffing.

* ``b`` (length normalization, 0-1, typically 0.75). A long document mentions
  everything a few times just by being long. b = 1 fully scales tf down by
  |D| / avgdl; b = 0 turns length normalization off.

The IDF used here is the "plus one" variant (as in Lucene) which can never go
negative. The textbook form, ln((N - n + 0.5) / (n + 0.5)), turns negative for
terms in more than half the documents, which would make matching a common word
*lower* a document's score.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from .index import InvertedIndex

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def idf(doc_count: int, doc_freq: int) -> float:
    return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def term_score(tf: int, doc_len: int, avgdl: float, idf_value: float, k1: float, b: float) -> float:
    if tf == 0:
        return 0.0
    norm = k1 * (1 - b + b * doc_len / avgdl) if avgdl else k1
    return idf_value * tf * (k1 + 1) / (tf + norm)


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.index = index
        self.k1 = k1
        self.b = b

    def score(self, doc_id: str, query_terms: Iterable[str]) -> float:
        index = self.index
        doc_len = index.docs[doc_id].length
        avgdl = index.avg_doc_length
        total = 0.0
        # Each distinct query term counts once; repeating a word in the query
        # should not double its weight.
        for term in set(query_terms):
            tf = index.term_freq(term, doc_id)
            if tf:
                total += term_score(
                    tf, doc_len, avgdl, idf(index.doc_count, index.doc_freq(term)), self.k1, self.b
                )
        return total
