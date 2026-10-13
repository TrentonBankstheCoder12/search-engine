"""Command-line interface.

    python -m searchengine index [FILE.jsonl ...]   # build or update the index
    python -m searchengine search "query"           # query it
    python -m searchengine delete DOC_ID

With no files, ``index`` loads the bundled corpus. The index file defaults to
``index.json`` (override with --index or the SEARCH_INDEX env var).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .corpus import BUNDLED_CORPUS, read_jsonl
from .engine import SearchEngine
from .query import QuerySyntaxError

DEFAULT_INDEX = "index.json"


def _open_engine(path: Path) -> SearchEngine:
    return SearchEngine.load(path) if path.exists() else SearchEngine()


def cmd_index(args: argparse.Namespace) -> int:
    path = Path(args.index)
    # --rebuild starts empty; otherwise new files are merged into the existing index.
    engine = SearchEngine() if args.rebuild else _open_engine(path)
    files = args.files or [BUNDLED_CORPUS]
    count = 0
    for file in files:
        for doc in read_jsonl(file):
            engine.add(str(doc["id"]), doc["text"], doc.get("title", ""))
            count += 1
    engine.save(path)
    print(
        f"indexed {count} documents ({engine.index.doc_count} total, "
        f"{len(engine.index.postings)} terms) -> {path}"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    path = Path(args.index)
    if not path.exists():
        print(f"no index at {path}; run `python -m searchengine index` first", file=sys.stderr)
        return 1
    engine = SearchEngine.load(path, k1=args.k1, b=args.b)
    try:
        results = engine.search(args.query, limit=args.limit)
    except QuerySyntaxError as exc:
        print(f"query error: {exc}", file=sys.stderr)
        return 2
    if not results:
        print("no results")
        return 0
    # ANSI bold for highlights in a terminal; plain brackets when piped.
    tty = sys.stdout.isatty()
    for rank, r in enumerate(results, start=1):
        snippet = r.snippet.replace("<mark>", "\033[1m" if tty else "[").replace(
            "</mark>", "\033[0m" if tty else "]"
        )
        print(f"{rank}. {r.title or r.doc_id}  ({r.doc_id}, score {r.score:.3f})")
        print(f"   {snippet}\n")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    path = Path(args.index)
    engine = _open_engine(path)
    if not engine.delete(args.doc_id):
        print(f"no document with id {args.doc_id!r}", file=sys.stderr)
        return 1
    engine.save(path)
    print(f"deleted {args.doc_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="searchengine", description=__doc__.splitlines()[0])
    parser.add_argument("--index", default=os.environ.get("SEARCH_INDEX", DEFAULT_INDEX))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="add JSONL files to the index (default: bundled corpus)")
    p.add_argument("files", nargs="*", type=Path)
    p.add_argument("--rebuild", action="store_true", help="start from an empty index")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="run a query")
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=5)
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("delete", help="remove a document by id")
    p.add_argument("doc_id")
    p.set_defaults(func=cmd_delete)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
