import math

import pytest

from searchengine import SearchEngine
from searchengine.ranking import BM25, idf, term_score


@pytest.fixture()
def tiny():
    # Lengths 2, 4, 3 -> avgdl = 3. N = 3. "ledger" is in 2 of the 3 docs.
    e = SearchEngine(k1=1.5, b=0.75)
    e.add("d1", "ledger ledger")
    e.add("d2", "ledger audit risk fraud")
    e.add("d3", "audit audit audit")
    return e


def test_bm25_matches_hand_computed_example(tiny):
    """Worked by hand (see WALKTHROUGH.md section 4):

    IDF(ledger) = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.470004
    d1: tf=2, |D|=2 -> norm = 1.5 * (0.25 + 0.75 * 2/3) = 1.125
        score = 0.470004 * 2 * 2.5 / (2 + 1.125) = 0.752006
    d2: tf=1, |D|=4 -> norm = 1.5 * (0.25 + 0.75 * 4/3) = 1.875
        score = 0.470004 * 1 * 2.5 / (1 + 1.875) = 0.408699
    """
    assert tiny.index.avg_doc_length == 3
    assert idf(3, 2) == pytest.approx(math.log(1.6))
    ranker = BM25(tiny.index, k1=1.5, b=0.75)
    assert ranker.score("d1", ["ledger"]) == pytest.approx(0.752006, abs=1e-6)
    assert ranker.score("d2", ["ledger"]) == pytest.approx(0.408699, abs=1e-6)
    assert ranker.score("d3", ["ledger"]) == 0

    results = tiny.search("ledger")
    assert [r.doc_id for r in results] == ["d1", "d2"]


def test_multi_term_score_is_sum_of_term_scores(tiny):
    ranker = BM25(tiny.index)
    both = ranker.score("d2", ["ledger", "audit"])
    assert both == pytest.approx(ranker.score("d2", ["ledger"]) + ranker.score("d2", ["audit"]))
    # Repeating a query word must not double its weight.
    assert ranker.score("d2", ["ledger", "ledger"]) == ranker.score("d2", ["ledger"])


def test_k1_controls_term_frequency_saturation():
    # With k1 = 0 the tf factor is constant: 10 occurrences score like 1.
    assert term_score(10, 5, 5, 1.0, k1=0.0, b=0.75) == term_score(1, 5, 5, 1.0, k1=0.0, b=0.75)
    # With k1 > 0 more occurrences help, but never beyond the (k1 + 1) * idf ceiling.
    scores = [term_score(tf, 5, 5, 1.0, k1=1.5, b=0.75) for tf in (1, 2, 5, 50, 5000)]
    assert scores == sorted(scores)
    assert scores[-1] < 2.5


def test_b_controls_length_normalization():
    short, long_ = 5, 50
    # b = 0: document length is ignored.
    assert term_score(2, short, 20, 1.0, 1.5, b=0.0) == term_score(2, long_, 20, 1.0, 1.5, b=0.0)
    # b = 0.75: the same tf counts for more in a short document.
    assert term_score(2, short, 20, 1.0, 1.5, b=0.75) > term_score(2, long_, 20, 1.0, 1.5, b=0.75)


def test_idf_is_never_negative_and_favours_rare_terms():
    n_docs = 100
    values = [idf(n_docs, df) for df in range(1, n_docs + 1)]
    assert all(v > 0 for v in values)
    assert values == sorted(values, reverse=True)


def test_scores_are_sorted_descending(corpus_engine):
    results = corpus_engine.search("risk OR interest OR payment", limit=50)
    assert len(results) > 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_relevant_document_ranks_first_on_corpus(corpus_engine):
    assert corpus_engine.search("chargeback dispute")[0].doc_id == "chargebacks"
    assert corpus_engine.search("salted password hash")[0].doc_id == "password-hashing"
