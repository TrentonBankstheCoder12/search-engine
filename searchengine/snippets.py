"""Build a short excerpt of a document with the query terms highlighted.

Strategy: slide a fixed-size window of words over the body text and pick the
window containing the most *distinct* query terms (ties go to the one with more
total hits, then the earliest). Distinct beats total so that a query for
"interest rate risk" prefers a window mentioning all three over one that says
"interest" five times.
"""

from __future__ import annotations

import html

from .tokenizer import tokenize, word_spans

DEFAULT_WINDOW = 30


def make_snippet(
    text: str,
    terms: set[str],
    window: int = DEFAULT_WINDOW,
    start_tag: str = "<mark>",
    end_tag: str = "</mark>",
    escape_html: bool = True,
) -> str:
    spans = word_spans(text)
    if not spans:
        return ""

    # position -> stemmed term, only for words that match the query.
    hits = {t.position: t.term for t in tokenize(text) if t.term in terms}

    best_start, best_key = 0, (-1, -1, 0)
    last_start = max(0, len(spans) - window)
    for start in range(last_start + 1):
        in_window = [hits[p] for p in range(start, start + window) if p in hits]
        key = (len(set(in_window)), len(in_window), -start)
        if key > best_key:
            best_start, best_key = start, key

    end = min(len(spans), best_start + window)
    esc = html.escape if escape_html else (lambda s: s)

    # Rebuild the excerpt from the original characters so punctuation and
    # capitalization survive; only the matched words get wrapped in tags.
    out: list[str] = []
    cursor = spans[best_start][0]
    for pos in range(best_start, end):
        s, e = spans[pos]
        out.append(esc(text[cursor:s]))
        word = esc(text[s:e])
        out.append(f"{start_tag}{word}{end_tag}" if pos in hits else word)
        cursor = e
    # Keep trailing punctuation of the final word (e.g. the full stop).
    tail_end = spans[end][0] if end < len(spans) else len(text)
    out.append(esc(text[cursor:tail_end].rstrip()))

    snippet = "".join(out).strip()
    prefix = "... " if best_start > 0 else ""
    suffix = " ..." if end < len(spans) else ""
    return prefix + snippet + suffix
