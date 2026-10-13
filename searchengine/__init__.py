"""A small full-text search engine built from scratch."""

from .engine import SearchEngine, SearchResult
from .index import InvertedIndex

__all__ = ["InvertedIndex", "SearchEngine", "SearchResult"]
