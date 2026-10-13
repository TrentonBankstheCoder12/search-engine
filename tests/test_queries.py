import pytest

from searchengine.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse
from searchengine.snippets import make_snippet


def ids(results):
    return {r.doc_id for r in results}


# --- parser -----------------------------------------------------------------


def test_parser_precedence_and_implicit_and():
    # NOT binds tightest, then AND (explicit or implied by a space), then OR.
    assert parse("wire ach OR NOT fraud") == Or((And((Term("wire"), Term("ach"))), Not(Term("fraud"))))
    assert parse("(wire OR ach) AND settle") == And((Or((Term("wire"), Term("ach"))), Term("settl")))


def test_parser_phrase_keeps_stopword_gaps():
    assert parse('"cost of trading"') == Phrase(((0, "cost"), (2, "trade")))


def test_parser_drops_stopwords_and_lowercase_operators():
    assert parse("the ledger") == Term("ledger")
    assert parse("the and of") is None
    # Lowercase "or" is just a (stop)word, not an operator.
    assert parse("wire or ach") == And((Term("wire"), Term("ach")))


@pytest.mark.parametrize("bad", ['"unclosed', "(wire OR ach", "wire OR", "AND", ")"])
def test_parser_rejects_malformed_queries(bad):
    with pytest.raises(QuerySyntaxError):
        parse(bad)


# --- boolean logic ----------------------------------------------------------


def test_and_requires_all_terms(engine):
    assert ids(engine.search("order book")) == {"orders", "market"}
    assert ids(engine.search("order AND limit")) == {"orders"}


def test_or_takes_union(engine):
    assert ids(engine.search("wire OR ach")) == {"wire", "ach"}


def test_not_excludes(engine):
    assert ids(engine.search("transfers NOT batches")) == {"wire"}
    assert ids(engine.search("settle AND NOT reverse")) == {"ach"}
    # A bare NOT means "everything except".
    assert ids(engine.search("NOT order")) == {"fraud", "wire", "ach"}


def test_grouping(engine):
    assert ids(engine.search("(wire OR card) NOT fraud")) == {"wire"}


def test_boolean_laws_hold_on_corpus(corpus_engine):
    """Property-style check: set algebra identities across many term pairs."""
    words = ["risk", "payment", "bank", "rate", "card", "market", "order", "account", "price", "data"]
    everything = corpus_engine.index.all_doc_ids()
    m = lambda q: corpus_engine.match(parse(q))  # noqa: E731
    for a in words:
        assert m(f"{a} OR NOT {a}") == everything
        assert m(f"{a} AND NOT {a}") == set()
        for b in words:
            assert m(f"{a} AND {b}") == m(a) & m(b)
            assert m(f"{a} OR {b}") == m(a) | m(b)
            # De Morgan: NOT (a OR b) == NOT a AND NOT b
            assert m(f"NOT ({a} OR {b})") == m(f"NOT {a} AND NOT {b}")


# --- phrases ----------------------------------------------------------------


def test_phrase_requires_adjacency_and_order(engine):
    # Both docs contain "order" and "book", but only one has them adjacent.
    assert ids(engine.search('"order book"')) == {"orders"}
    assert ids(engine.search('"book order"')) == set()


def test_phrase_respects_stopword_gaps(engine):
    assert ids(engine.search('"rests in the order book"')) == {"orders"}
    # Same content words, but the stopword count in between is different.
    assert ids(engine.search('"rests order book"')) == set()


def test_phrase_combines_with_boolean(engine):
    assert ids(engine.search('"wire transfers" OR "insufficient funds"')) == {"wire", "ach"}
    assert ids(engine.search('transfers NOT "same day"')) == {"ach"}


def test_phrase_does_not_straddle_title_and_body(engine):
    # Title "Wire transfers" + body "Wire transfers settle ...": "transfers wire"
    # would only exist across the boundary, so it must not match.
    assert ids(engine.search('"transfers wire"')) == set()


def test_phrase_on_corpus(corpus_engine):
    hits = ids(corpus_engine.search('"bid ask spread"', limit=50))
    assert "bid-ask-spread" in hits
    assert ids(corpus_engine.search('"settle one business day"')) == {"settlement"}


# --- snippets ---------------------------------------------------------------


def test_snippet_highlights_matches_and_escapes_html():
    text = "Fees <b>matter</b>. " + "filler " * 40 + "Low fees compound over time."
    snip = make_snippet(text, {"fee", "compound"}, window=8)
    assert "<mark>fees</mark>" in snip and "<mark>compound</mark>" in snip
    assert snip.startswith("... ")  # window moved past the beginning
    assert "<b>" not in make_snippet(text, {"fee"}, window=8)
    assert "&lt;b&gt;" in make_snippet(text, {"fee"}, window=8)


def test_search_results_include_snippets(engine):
    [r] = engine.search("insufficient")
    assert "<mark>insufficient</mark>" in r.snippet
    # Negated terms are never highlighted.
    for r in engine.search("transfers NOT batches"):
        assert "batches</mark>" not in r.snippet
