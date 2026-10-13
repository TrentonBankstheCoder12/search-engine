import pytest

from searchengine.stemmer import measure, stem
from searchengine.tokenizer import STOPWORDS, terms, tokenize

# --- tokenizer --------------------------------------------------------------


def test_lowercases_and_splits_on_punctuation():
    assert terms("Ledger, AUDIT; risk!") == ["ledger", "audit", "risk"]


def test_removes_stopwords_but_keeps_their_positions():
    tokens = tokenize("the cost of the trade")
    assert [t.term for t in tokens] == ["cost", "trade"]
    # "the"=0, "cost"=1, "of"=2, "the"=3, "trade"=4: the gap is preserved so a
    # phrase query for "cost trade" will not match this text.
    assert [t.position for t in tokens] == [1, 4]


def test_character_offsets_point_at_original_words():
    text = "Wire Transfers settle fast."
    for tok in tokenize(text):
        assert text[tok.start : tok.end].lower().startswith(tok.term[:3])


def test_only_stopwords_yields_nothing():
    assert tokenize("to be or not to be") == []
    assert {"the", "and", "of"} <= STOPWORDS


def test_numbers_are_kept():
    assert terms("settles in 1 day, rate 5") == ["settl", "1", "dai", "rate", "5"]


# --- stemmer ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # Classic examples from Porter's paper
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("cats", "cat"),
        ("agreed", "agre"),
        ("hopping", "hop"),
        ("falling", "fall"),
        ("filing", "file"),
        ("happy", "happi"),
        ("relational", "relat"),
        ("conditional", "condit"),
        ("generalization", "gener"),
        ("adoption", "adopt"),
        ("controlling", "control"),
        ("sing", "sing"),
        # Domain words: different forms must collapse to the same stem
        ("payments", "payment"),
        ("settled", "settl"),
        ("settlement", "settlement"),
    ],
)
def test_porter_examples(word, expected):
    assert stem(word) == expected


def test_related_forms_share_a_stem():
    groups = [
        ["connect", "connected", "connecting", "connection", "connections"],
        ["trade", "trades", "trading", "traded"],
        ["compound", "compounding", "compounds"],
    ]
    for group in groups:
        assert len({stem(w) for w in group}) == 1, group


def test_measure():
    assert [measure(w) for w in ("tr", "tree", "trouble", "oats", "troubles", "private")] == [
        0, 0, 1, 1, 2, 2,
    ]  # fmt: skip


def test_short_words_untouched():
    assert stem("as") == "as"
    assert stem("is") == "is"
