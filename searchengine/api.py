"""HTTP API: GET /search?q=...

Run with ``uvicorn searchengine.api:app``. On startup the app loads the index
file named by SEARCH_INDEX (default ``index.json``) if it exists, otherwise it
builds an in-memory index from the bundled corpus so the API works out of the box.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .corpus import BUNDLED_CORPUS, read_jsonl
from .engine import SearchEngine
from .query import QuerySyntaxError


def build_default_engine() -> SearchEngine:
    path = Path(os.environ.get("SEARCH_INDEX", "index.json"))
    if path.exists():
        return SearchEngine.load(path)
    engine = SearchEngine()
    for doc in read_jsonl(BUNDLED_CORPUS):
        engine.add(doc["id"], doc["text"], doc.get("title", ""))
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests may install their own engine before startup; don't overwrite it.
    if getattr(app.state, "engine", None) is None:
        app.state.engine = build_default_engine()
    yield


app = FastAPI(
    title="Search Engine",
    description="BM25 full-text search with boolean and phrase queries.",
    version="1.0.0",
    lifespan=lifespan,
)


class Hit(BaseModel):
    id: str
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[Hit]


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=500, description="Query string"),
    limit: int = Query(10, ge=1, le=100),
) -> SearchResponse:
    engine: SearchEngine = app.state.engine
    try:
        results = engine.search(q, limit=limit)
    except QuerySyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"invalid query: {exc}") from exc
    hits = [Hit(id=r.doc_id, title=r.title, score=r.score, snippet=r.snippet) for r in results]
    return SearchResponse(query=q, total=len(hits), results=hits)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": app.state.engine.index.doc_count}
