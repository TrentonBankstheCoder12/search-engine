# Walkthrough: Search Engine

A guide to how this project works and why it was built this way. Read it top to bottom, then try to explain each section out loud without looking. That is roughly what an interviewer will ask you to do.

## 1. Text analysis: from a sentence to index terms

Before anything is stored, text runs through one pipeline (`searchengine/tokenizer.py`):

1. **Split** on anything that isn't a letter or digit: `"bid-ask spread!"` becomes `bid`, `ask`, `spread`.
2. **Lowercase**, so `Payment` and `payment` are the same term.
3. **Drop stopwords** (`the`, `of`, `and`, ...). They appear in nearly every document, so they cost space and tell BM25 almost nothing.
4. **Stem** each word (section 2).

Each token remembers its **position** (word number) and its **character offsets**. Positions are counted *before* stopwords are removed, so `"cost of trading"` stores `cost` at 0 and `trade` at 2. That gap matters for phrase queries (section 5).

The most important rule: **queries go through exactly the same pipeline as documents.** If documents were stemmed and queries weren't, `payments` in a query would never match the indexed `payment`.

## 2. The Porter stemmer

Stemming collapses word forms: `trade`, `trades`, `trading`, `traded` all become `trade`. The implementation in `searchengine/stemmer.py` is Porter's 1980 algorithm, written out step by step:

- **Step 1** strips plurals and `-ed`/`-ing` (`ponies` -> `poni`, `hopping` -> `hop`, `filing` -> `file`).
- **Steps 2-4** strip derivational suffixes (`-ational`, `-ness`, `-ment`, ...).
- **Step 5** tidies a trailing `e` or double `l`.

The guard that keeps it sensible is the **measure** `m`: roughly the number of vowel-consonant sequences (syllables) left in the stem. Most rules only fire if the remaining stem has `m > 0` or `m > 1`, which is why `sing` keeps its `ing` but `singing` doesn't.

Stems aren't always real words (`relational` -> `relat`). That's fine: they are internal keys, and the snippet generator highlights the *original* words using character offsets.

**Trade-off:** stemming improves recall (more matches) but can hurt precision. `university` and `universe` both stem to `univers`. Production engines often index both stemmed and exact forms.

## 3. The positional inverted index

`searchengine/index.py` stores:

```
postings["ledger"] = {"doc-a": [3, 17], "doc-c": [0]}
```

- **Inverted:** you look up a *term* and get the *documents*. "Which docs contain ledger?" is one dict lookup instead of reading every document.
- **Positional:** each document maps to the list of positions where the term occurs. `len(positions)` is the term frequency BM25 needs, and the positions themselves power phrase queries.

The index also keeps each document's token count and a running total, so the average document length (needed by BM25) is O(1) and stays correct after deletes.

Titles are indexed too. The body's positions are shifted past the title with a one-word gap, so a phrase can't match across the title/body boundary.

**Incremental updates:** `add` with an existing id deletes the old version first (upsert). `delete` re-tokenizes the stored text to find which posting lists to clean and removes lists that become empty, so document frequency (`len(postings[term])`) is always exact. The alternative is a forward index (doc -> terms), which makes deletes faster at the cost of storing every term twice.

## 4. BM25 ranking

`searchengine/ranking.py`. For each query term `t` present in document `D`:

```
IDF(t)  = ln(1 + (N - n + 0.5) / (n + 0.5))
score  += IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))
```

`N` = number of documents, `n` = documents containing `t`, `tf` = occurrences of `t` in `D`, `|D|` = document length, `avgdl` = average length.

Three ideas in one formula:

- **IDF (rarity):** a term in every document can't tell documents apart; a rare term is a strong signal. The `1 +` inside the log keeps IDF positive (the textbook version goes negative for terms in over half the documents).
- **`k1` (saturation, default 1.5):** the tf part grows quickly, then flattens toward `k1 + 1`. The 2nd mention of "chargeback" matters a lot, the 20th barely at all, so keyword stuffing doesn't win. `k1 = 0` ignores tf completely.
- **`b` (length normalization, default 0.75):** a long document mentions many words just by being long. `b = 1` fully divides tf by relative length; `b = 0` ignores length.

**Hand-computed example** (this is `test_bm25_matches_hand_computed_example`). Three docs: `d1 = "ledger ledger"` (length 2), `d2 = "ledger audit risk fraud"` (4), `d3 = "audit audit audit"` (3). So `N = 3`, `avgdl = 3`, and `ledger` is in `n = 2` docs. With `k1 = 1.5`, `b = 0.75`:

```
IDF          = ln(1 + 1.5/2.5) = ln(1.6)                 = 0.470004
d1: norm     = 1.5 * (0.25 + 0.75 * 2/3) = 1.125
    score    = 0.470004 * 2 * 2.5 / (2 + 1.125)          = 0.752006
d2: norm     = 1.5 * (0.25 + 0.75 * 4/3) = 1.875
    score    = 0.470004 * 1 * 2.5 / (1 + 1.875)          = 0.408699
```

d1 wins: it mentions `ledger` twice *and* is shorter than average.

## 5. Query parsing, boolean matching and phrases

`searchengine/query.py` is a small **recursive-descent parser**, one function per grammar rule. Precedence, from tightest to loosest: `NOT`, then `AND` (a space also means AND), then `OR`. So `wire ach OR NOT fraud` parses as `(wire AND ach) OR (NOT fraud)`. The result is a tree of `Term`, `Phrase`, `And`, `Or`, `Not` nodes.

`searchengine/engine.py` then searches in **two phases**:

1. **Match** (which documents qualify), using set algebra on posting lists: AND = intersection (smallest set first), OR = union, NOT = subtract. Inside an AND, `NOT` children are subtracted from the positive result so we never build "all documents except X" unless the query is a bare `NOT`.
2. **Rank** only the matches with BM25. Terms under a `NOT` don't contribute to score or highlighting.

**Phrases** use positions. For `"order book"` (offsets `order`=0, `book`=1): intersect the two posting lists to get candidates, then for each start position `p` of `order`, check that `book` appears at `p + 1`. Because stopword gaps were kept, `"rests in the order book"` matches the source text but `"rests order book"` does not.

## 6. Snippets

`searchengine/snippets.py` slides a 30-word window across the body and keeps the window with the most *distinct* query terms (then most total hits, then earliest). Distinct beats total so a query for `interest rate risk` prefers a window with all three words over one that says "interest" five times. The excerpt is rebuilt from the original characters, so capitalization and punctuation survive, and the text is HTML-escaped before `<mark>` tags are added so a document can't inject markup into a web page.

## 7. Persistence

The index saves as compact JSON with a `version` field (`InvertedIndex.save`). Writing is **atomic**: write to a temp file in the same directory, then `os.replace` it over the old file. A crash mid-save leaves the previous index intact instead of a truncated file. JSON was chosen over a binary format because it is debuggable by eye; section 8 explains what that costs.

## 8. Known limitation (be ready to discuss this)

Every save rewrites the **whole** index, and every search loads it **fully into memory**. That's fine for thousands of documents and wrong for millions. Real engines like Lucene write immutable **segments**: new documents go into a small new segment, deletes are recorded as a bitmap of "tombstoned" doc ids, and background merges combine segments and physically drop deleted docs. Posting lists are also stored sorted and compressed (delta encoding plus variable-length integers) rather than as JSON.

A second limitation: ranking scores *every* matching document. For a broad query on a large index, engines use top-k algorithms like WAND that skip documents which can't make the top 10.

## Likely interview questions

1. **Why an inverted index instead of scanning documents?** Lookup cost becomes proportional to the matching postings, not the collection size. Scanning 10 million documents for one word is far slower than one hash lookup plus reading a short list.
2. **Explain BM25's `k1` and `b`.** `k1` controls how fast repeated occurrences stop adding score (saturation); `b` controls how much long documents are penalized (section 4).
3. **How do phrase queries work?** Store positions in postings, intersect the terms' documents, then check positions line up at the right offsets (section 5).
4. **Why must queries use the same analyzer as documents?** Otherwise the query term (`payments`) won't equal the indexed term (`payment`) and nothing matches.
5. **How would you scale this to millions of documents?** Segments plus merges, compressed sorted postings, top-k (WAND), and sharding the index by document across machines with a coordinator merging each shard's top results.
6. **How do you delete from an inverted index efficiently?** Here: remove the doc from each of its posting lists. At scale: mark it deleted in a bitmap (tombstone), filter it out at query time, and reclaim space during segment merges.
7. **Why is the save atomic?** A crash halfway through writing would otherwise corrupt the only copy. Write to a temp file then rename, because rename is atomic.

## Try extending it

- Weight title matches higher than body matches (BM25F)
- Add prefix queries (`settl*`) by keeping a sorted term list and binary-searching it
- Add fuzzy matching with Levenshtein distance for typos
- Replace JSON with delta-encoded, varint-compressed posting lists and measure the size difference
- Implement a skip-list or galloping intersection for AND over very long posting lists
