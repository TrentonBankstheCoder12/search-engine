# Search Engine

[![CI](https://github.com/TrentonBankstheCoder12/search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/TrentonBankstheCoder12/search-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

A **full-text search engine built from scratch**, with no search libraries: a tokenizer and Porter stemmer, a positional inverted index, BM25 ranking, boolean and exact-phrase queries, highlighted snippets, and an index that persists to disk and supports incremental adds and deletes. Search is everywhere in financial software: compliance teams searching communications, support agents searching transaction notes, analysts searching filings and research. Building one by hand shows how relevance ranking actually works and how a data structure choice (positions in the postings) unlocks features like phrase search.

## Features

- **Text analysis:** lowercasing, stopword removal and a from-scratch Porter stemmer (`payments`, `paying` and `payment` all match)
- **Positional inverted index:** term -> document -> word positions, so lookups are a dict access instead of a scan
- **BM25 ranking** with tunable `k1` (term-frequency saturation) and `b` (length normalization)
- **Query language:** `AND`, `OR`, `NOT`, parentheses, and `"exact phrases"`; a space between words means AND
- **Snippets** that pick the most relevant window of text and wrap matches in `<mark>` (HTML-escaped)
- **Persistence:** compact JSON, written atomically (temp file + rename), with a format version check
- **Incremental updates:** add, replace or delete single documents without rebuilding
- **Interfaces:** a CLI (`index`, `search`, `delete`) and a FastAPI endpoint `GET /search?q=`
- **Bundled corpus:** 44 short original paragraphs on finance and engineering concepts

## Architecture

```
  "fraud NOT card"           JSONL documents
        │                          │
        ▼                          ▼
 ┌──────────────┐          ┌──────────────┐
 │  query.py    │          │ tokenizer.py │  lowercase → split → stopwords → stemmer.py
 │ parse → tree │          └──────┬───────┘
 └──────┬───────┘                 ▼
        │                  ┌──────────────┐       save/load (atomic JSON)
        │                  │   index.py   │ ◀──────────────▶ index.json
        │                  │ term→doc→pos │
        ▼                  └──────┬───────┘
 ┌──────────────────────────────────────────┐
 │ engine.py                                │
 │  1. match: set algebra on posting lists  │
 │     AND ∩, OR ∪, NOT −, phrase=positions │
 │  2. rank:  ranking.py (BM25)             │
 │  3. snippets.py: best window + <mark>    │
 └──────┬──────────────────────────┬────────┘
        ▼                          ▼
     cli.py                     api.py (FastAPI)
```

## Quick start

```bash
pip install -r requirements-dev.txt

# Build an index from the bundled corpus, then query it
python -m searchengine index
python -m searchengine search '"order book" OR spread'

# Or run the API (builds from the bundled corpus if no index.json exists)
uvicorn searchengine.api:app --reload
# http://localhost:8000/search?q=fraud%20NOT%20card   (docs at /docs)
```

## Usage

```text
$ python -m searchengine search 'fraud NOT card' -n 2
1. Fraud scoring  (fraud-scoring, score 4.832)
   [Fraud] scoring assigns each transaction a risk score based on signals such as the amount, ...

2. Chargebacks  (chargebacks, score 2.987)
   ... A high chargeback rate can get a merchant dropped by its payment processor, so [fraud] ...
```

| Query | Meaning |
|---|---|
| `compound interest` | both terms (implicit AND) |
| `ach OR wire` | either term |
| `transfers NOT wire` | first term, excluding documents with the second |
| `"order book"` | the words adjacent and in this order |
| `("limit order" OR spread) NOT crypto` | grouping |

Operators must be uppercase; lowercase `and`/`or`/`not` are ordinary (stop)words.

Add your own documents as JSON Lines (`{"id": ..., "title": ..., "text": ...}` per line):

```bash
python -m searchengine index my_docs.jsonl     # merges into the existing index
python -m searchengine index --rebuild a.jsonl # start fresh
python -m searchengine delete wire-transfers
```

API response:

```bash
curl 'localhost:8000/search?q=%22bid%20ask%20spread%22&limit=1'
# {"query": "\"bid ask spread\"", "total": 1,
#  "results": [{"id": "bid-ask-spread", "title": "The bid-ask spread", "score": 11.887,
#               "snippet": "The <mark>bid</mark>-<mark>ask</mark> <mark>spread</mark> is ..."}]}
```

## Testing

```bash
ruff check .
pytest -q
```

64 tests cover tokenization and stopword position gaps, the stemmer against examples from Porter's paper, BM25 against a hand-computed example (and the effect of `k1` and `b`), phrase matching, boolean logic (including a property-style check of De Morgan's laws and set identities across the corpus), snippets, persistence round-trips, deletions, the CLI and the API. An invariant test applies 300 random adds and deletes and checks the index is identical to one built from scratch.

## What I'd add next

- Segment-based storage (immutable segments + merges, like Lucene) so saving doesn't rewrite the whole index
- Top-k retrieval with WAND/MaxScore so ranking doesn't score every matching document
- Prefix and fuzzy (edit-distance) queries for typos in names and tickers
- Field weighting (title matches worth more than body matches)

See [WALKTHROUGH.md](WALKTHROUGH.md) for the design decisions behind this project.

## License

MIT
