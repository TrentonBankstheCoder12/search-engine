import json
import random

from searchengine import InvertedIndex, SearchEngine


def test_postings_are_positional():
    idx = InvertedIndex()
    idx.add("d", "risk and liquidity risk", title="")
    assert idx.positions("risk", "d") == [0, 3]
    assert idx.term_freq("risk", "d") == 2
    assert idx.docs["d"].length == 3  # "and" is a stopword


def test_delete_removes_document_and_postings(engine):
    assert engine.delete("wire") is True
    assert "wire" not in engine.index.docs
    assert all("wire" not in plist for plist in engine.index.postings.values())
    # Terms that only that document used are gone entirely (no empty lists).
    assert "revers" not in engine.index.postings
    assert {r.doc_id for r in engine.search("transfers")} == {"ach"}
    assert engine.delete("wire") is False


def test_delete_updates_statistics(engine):
    before = engine.index.doc_freq("settl")
    engine.delete("ach")
    assert engine.index.doc_freq("settl") == before - 1
    expected = sum(d.length for d in engine.index.docs.values()) / 4
    assert engine.index.avg_doc_length == expected


def test_re_adding_an_id_replaces_it(engine):
    engine.add("wire", "Completely new text about stablecoins.", "Replaced")
    assert engine.search("reverse") == []
    assert [r.doc_id for r in engine.search("stablecoins")] == ["wire"]
    assert engine.index.doc_count == 5


def test_persistence_round_trip(engine, tmp_path):
    path = tmp_path / "nested" / "index.json"
    engine.save(path)
    loaded = SearchEngine.load(path)

    assert loaded.index.to_dict() == engine.index.to_dict()
    for q in ["order", '"order book"', "transfers NOT batches", "card OR wire"]:
        assert loaded.search(q) == engine.search(q)
    # Save leaves no temp files behind.
    assert [p.name for p in path.parent.iterdir()] == ["index.json"]

    # A loaded index is still fully mutable.
    loaded.delete("orders")
    loaded.add("new", "brand new order flow")
    assert {r.doc_id for r in loaded.search("order")} == {"market", "new"}


def test_load_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 999, "docs": {}, "postings": {}}))
    try:
        InvertedIndex.load(path)
    except ValueError as exc:
        assert "version" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_incremental_updates_match_a_fresh_build(corpus_engine):
    """Invariant: any sequence of adds/deletes leaves the same index as building
    from scratch with only the surviving documents."""
    docs = {d.doc_id: d for d in corpus_engine.index.docs.values()}
    rng = random.Random(42)
    live = InvertedIndex()
    present: set[str] = set()
    for _ in range(300):
        doc_id = rng.choice(list(docs))
        if doc_id in present and rng.random() < 0.5:
            live.delete(doc_id)
            present.discard(doc_id)
        else:
            live.add(doc_id, docs[doc_id].text, docs[doc_id].title)
            present.add(doc_id)

    fresh = InvertedIndex()
    for doc_id in sorted(present):
        fresh.add(doc_id, docs[doc_id].text, docs[doc_id].title)

    assert live.docs.keys() == fresh.docs.keys()
    assert live.postings == fresh.postings
    assert live.avg_doc_length == fresh.avg_doc_length
