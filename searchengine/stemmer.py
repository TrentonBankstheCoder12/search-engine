"""A from-scratch implementation of the Porter stemmer (Martin Porter, 1980).

Stemming maps related word forms onto one index term so that a search for
"payments" also finds "payment" and "paying". The output is not always a real
word ("relational" -> "relat"); that's fine, because the same function runs on
both documents and queries, so they still meet in the middle.

Vocabulary used throughout:
  * A word is viewed as [C](VC)^m[V], where C is a run of consonants and V a
    run of vowels. ``m`` is the "measure", roughly the number of syllables.
    Most rules only fire when the remaining stem has m > 0 or m > 1, which
    stops us from chopping short words down to nothing ("sing" keeps its "ing").
  * "y" counts as a vowel when it follows a consonant ("happy"), and as a
    consonant otherwise ("yes", "toy").
"""

from __future__ import annotations

VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in VOWELS:
        return False
    if ch == "y":
        # y is a consonant at the start of a word or after a vowel.
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def measure(stem: str) -> int:
    """Count VC sequences in ``stem``: tree=0, trouble=1, troubles=2."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if prev_vowel and not vowel:
            m += 1
        prev_vowel = vowel
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """consonant-vowel-consonant where the last C is not w, x or y ("hop", not "snow").

    Used to decide when to restore a final "e": "filing" -> "fil" -> "file".
    """
    if len(word) < 3:
        return False
    return (
        _is_consonant(word, len(word) - 3)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


def _replace_suffix(word: str, rules: list[tuple[str, str]], min_measure: int) -> str:
    """Apply the first rule whose suffix matches, if the leftover stem is long enough.

    Only the *longest* matching suffix is considered (rules are listed so that
    happens), and if its measure condition fails we stop rather than trying a
    shorter suffix, exactly as the original algorithm specifies.
    """
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if measure(stem) > min_measure:
                return stem + replacement
            return word
    return word


# --- Step 1: plurals and -ed / -ing -------------------------------------------


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]  # caresses -> caress
    if word.endswith("ies"):
        return word[:-2]  # ponies -> poni
    if word.endswith("ss"):
        return word  # caress -> caress
    if word.endswith("s"):
        return word[:-1]  # cats -> cat
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        stem = word[:-3]
        return stem + "ee" if measure(stem) > 0 else word  # agreed -> agree, feed stays

    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if not _has_vowel(stem):
                return word  # "bled", "sing": no vowel left, so not really a suffix
            return _step1b_cleanup(stem)
    return word


def _step1b_cleanup(stem: str) -> str:
    # After removing -ed/-ing, tidy the stem so it lines up with the base word.
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"  # conflated -> conflate, troubled -> trouble
    if _ends_double_consonant(stem) and stem[-1] not in "lsz":
        return stem[:-1]  # hopping -> hop (but falling -> fall)
    if measure(stem) == 1 and _ends_cvc(stem):
        return stem + "e"  # filing -> file
    return stem


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"  # happy -> happi, so it matches "happiness" later
    return word


# --- Steps 2-4: derivational suffixes -------------------------------------------

_STEP2 = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
]

_STEP3 = [
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
]

_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]  # fmt: skip


def _longest_first(rules: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(rules, key=lambda r: len(r[0]), reverse=True)


_STEP2_SORTED = _longest_first(_STEP2)
_STEP3_SORTED = _longest_first(_STEP3)
_STEP4_SORTED = sorted(_STEP4, key=len, reverse=True)


def _step4(word: str) -> str:
    for suffix in _STEP4_SORTED:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if measure(stem) <= 1:
                return word
            # "-ion" is only removed after s or t: adoption -> adopt, but not "lion".
            if suffix == "ion" and not stem.endswith(("s", "t")):
                return word
            return stem
    return word


# --- Step 5: final tidy-up ----------------------------------------------------


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem  # probate -> probat, but rate stays rate
    if measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]  # controll -> control
    return word


def stem(word: str) -> str:
    """Return the Porter stem of a lowercase word."""
    if len(word) <= 2:
        return word  # too short to have a meaningful suffix
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _replace_suffix(word, _STEP2_SORTED, 0)
    word = _replace_suffix(word, _STEP3_SORTED, 0)
    word = _step4(word)
    return _step5(word)
