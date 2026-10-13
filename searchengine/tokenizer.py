"""Turn raw text into index terms.

Pipeline: lowercase -> split on non-alphanumerics -> drop stopwords -> stem.

Each token keeps two kinds of location:
  * ``position``: its word index in the original text, *counting stopwords*.
    Phrase queries compare positions, so keeping the gaps means
    "order book" will not falsely match "order of the book".
  * ``start``/``end``: character offsets, used by the snippet generator to
    highlight the exact original words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .stemmer import stem

# Words so common they carry almost no meaning for ranking. Removing them
# shrinks the index noticeably (they appear in nearly every document).
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself s t
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE | re.ASCII)


@dataclass(frozen=True, slots=True)
class Token:
    term: str  # the stemmed form stored in the index
    position: int  # word offset in the source text (stopwords included)
    start: int  # character offset of the original word
    end: int


def tokenize(text: str) -> list[Token]:
    """Analyze ``text`` into stemmed, stopword-free tokens with positions."""
    tokens: list[Token] = []
    # Match on the original text and lowercase each word afterwards: lowercasing
    # the whole string first can change its length for some Unicode characters,
    # which would make the character offsets point at the wrong place.
    for position, match in enumerate(_WORD_RE.finditer(text)):
        word = match.group().lower()
        if word in STOPWORDS:
            continue
        tokens.append(Token(stem(word), position, match.start(), match.end()))
    return tokens


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character span of every word (stopwords included), indexed by position."""
    return [m.span() for m in _WORD_RE.finditer(text)]


def terms(text: str) -> list[str]:
    """Convenience wrapper: just the stemmed terms, in order."""
    return [t.term for t in tokenize(text)]
