import pytest

from searchengine import SearchEngine
from searchengine.corpus import BUNDLED_CORPUS, read_jsonl


@pytest.fixture()
def engine():
    """A tiny hand-made collection, small enough to reason about in your head."""
    e = SearchEngine()
    e.add("orders", "A limit order rests in the order book until the price is reached.", "Limit orders")
    e.add("market", "A market order trades now; the book may be thin so the order can slip.", "Market orders")
    e.add("fraud", "Card fraud scoring flags risky card payments for review.", "Fraud scoring")
    e.add("wire", "Wire transfers settle the same day and are hard to reverse.", "Wire transfers")
    e.add("ach", "ACH transfers settle in batches and can be returned for insufficient funds.", "ACH")
    return e


@pytest.fixture(scope="session")
def corpus_engine():
    e = SearchEngine()
    for doc in read_jsonl(BUNDLED_CORPUS):
        e.add(doc["id"], doc["text"], doc["title"])
    return e
