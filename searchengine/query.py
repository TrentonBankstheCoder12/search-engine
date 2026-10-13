"""Query language: parse a query string into a small syntax tree.

Grammar (lowest to highest precedence):

    or_expr  := and_expr ("OR" and_expr)*
    and_expr := unary (["AND"] unary)*        # juxtaposition means AND
    unary    := "NOT" unary | primary
    primary  := WORD | "quoted phrase" | "(" or_expr ")"

Examples:
    compound interest            -> AND(compound, interest)
    ach OR wire                  -> OR(ach, wire)
    fraud NOT card               -> AND(fraud, NOT(card))
    "order book" AND (bid OR ask)

Operators must be UPPERCASE so ordinary words like "or" and "not" in natural
language stay plain (they are stopwords and simply disappear).

Words run through the same analyzer as documents, so "Payments" in a query
becomes the stem "payment", matching what is in the index. A word that
analyzes to nothing (a stopword) is dropped from the tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tokenizer import tokenize


@dataclass(frozen=True)
class Term:
    term: str


@dataclass(frozen=True)
class Phrase:
    # (relative position, term) pairs. Positions keep stopword gaps, so
    # "cost of trading" is stored as [(0, "cost"), (2, "trade")].
    terms: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class And:
    children: tuple[Node, ...]


@dataclass(frozen=True)
class Or:
    children: tuple[Node, ...]


@dataclass(frozen=True)
class Not:
    child: Node


Node = Term | Phrase | And | Or | Not


class QuerySyntaxError(ValueError):
    pass


# Lexer: quoted phrases, parentheses, or runs of anything else.
_LEX_RE = re.compile(r'"([^"]*)"|(\()|(\))|([^\s()"]+)')
_OPERATORS = {"AND", "OR", "NOT"}


def _lex(query: str) -> list[tuple[str, str]]:
    if query.count('"') % 2:
        raise QuerySyntaxError("unbalanced quote")
    out: list[tuple[str, str]] = []
    for m in _LEX_RE.finditer(query):
        phrase, lparen, rparen, word = m.groups()
        if phrase is not None:
            out.append(("PHRASE", phrase))
        elif lparen:
            out.append(("(", lparen))
        elif rparen:
            out.append((")", rparen))
        elif word in _OPERATORS:
            out.append((word, word))
        else:
            out.append(("WORD", word))
    return out


def _analyze_text(text: str) -> Node | None:
    """Turn a word or phrase into a Term, a Phrase, or None if only stopwords.

    A single lexer word can still yield several terms ("bid-ask" -> bid, ask);
    we treat that as a phrase, which is what the user almost certainly meant.
    """
    tokens = tokenize(text)
    if not tokens:
        return None
    if len(tokens) == 1:
        return Term(tokens[0].term)
    base = tokens[0].position
    return Phrase(tuple((t.position - base, t.term) for t in tokens))


class _Parser:
    """Recursive-descent parser: one method per grammar rule."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i][0] if self.i < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node | None:
        node = self.or_expr()
        if self.peek() is not None:
            raise QuerySyntaxError(f"unexpected {self.tokens[self.i][1]!r}")
        return node

    def or_expr(self) -> Node | None:
        children = [self.and_expr()]
        while self.peek() == "OR":
            self.take()
            children.append(self.and_expr())
        return _combine(Or, children)

    def and_expr(self) -> Node | None:
        children = [self.unary()]
        while self.peek() in ("AND", "NOT", "WORD", "PHRASE", "("):
            if self.peek() == "AND":
                self.take()
            children.append(self.unary())
        return _combine(And, children)

    def unary(self) -> Node | None:
        if self.peek() == "NOT":
            self.take()
            child = self.unary()
            return Not(child) if child is not None else None
        return self.primary()

    def primary(self) -> Node | None:
        kind = self.peek()
        if kind is None:
            raise QuerySyntaxError("unexpected end of query")
        if kind in ("WORD", "PHRASE"):
            return _analyze_text(self.take()[1])
        if kind == "(":
            self.take()
            node = self.or_expr()
            if self.peek() != ")":
                raise QuerySyntaxError("missing closing parenthesis")
            self.take()
            return node
        raise QuerySyntaxError(f"unexpected {self.take()[1]!r}")


def _combine(cls: type[And] | type[Or], children: list[Node | None]) -> Node | None:
    kept = [c for c in children if c is not None]  # drop stopword-only pieces
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return cls(tuple(kept))


def parse(query: str) -> Node | None:
    """Parse ``query``; returns None if it contains only stopwords or is empty."""
    tokens = _lex(query)
    if not tokens:
        return None
    return _Parser(tokens).parse()


def positive_terms(node: Node | None) -> set[str]:
    """Terms that should contribute to ranking and highlighting.

    Terms under a NOT are excluded: a document only matches if it *lacks*
    them, so they cannot add to its score and there is nothing to highlight.
    """
    match node:
        case None | Not():
            return set()
        case Term(term=t):
            return {t}
        case Phrase(terms=ts):
            return {t for _, t in ts}
        case And(children=cs) | Or(children=cs):
            return set().union(*(positive_terms(c) for c in cs))
    raise TypeError(node)
