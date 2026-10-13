import json

import pytest
from fastapi.testclient import TestClient

from searchengine.api import app
from searchengine.cli import main


@pytest.fixture()
def client(engine):
    app.state.engine = engine
    with TestClient(app) as c:
        yield c
    app.state.engine = None


def test_api_search(client):
    r = client.get("/search", params={"q": '"order book"'})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    hit = body["results"][0]
    assert hit["id"] == "orders" and hit["title"] == "Limit orders"
    assert "<mark>order</mark> <mark>book</mark>" in hit["snippet"]


def test_api_validation_and_syntax_errors(client):
    assert client.get("/search").status_code == 422
    assert client.get("/search", params={"q": ""}).status_code == 422
    r = client.get("/search", params={"q": '"unclosed'})
    assert r.status_code == 400 and "invalid query" in r.json()["detail"]


def test_api_limit(client):
    r = client.get("/search", params={"q": "NOT zzz", "limit": 2})
    assert len(r.json()["results"]) == 2


def test_cli_index_search_delete(tmp_path, capsys):
    index = str(tmp_path / "idx.json")
    corpus = tmp_path / "docs.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(d)
            for d in [
                {"id": "a", "title": "Ledgers", "text": "A ledger records every entry."},
                {"id": "b", "title": "Cards", "text": "Card networks route payments."},
            ]
        )
    )
    assert main(["--index", index, "index", str(corpus)]) == 0
    assert "2 total" in capsys.readouterr().out

    assert main(["--index", index, "search", "ledger"]) == 0
    out = capsys.readouterr().out
    assert "Ledgers  (a," in out and "[ledger]" in out

    assert main(["--index", index, "delete", "a"]) == 0
    capsys.readouterr()
    main(["--index", index, "search", "ledger"])
    assert "no results" in capsys.readouterr().out
    assert main(["--index", index, "search", '"oops']) == 2


def test_cli_indexes_bundled_corpus_by_default(tmp_path, capsys):
    index = str(tmp_path / "idx.json")
    assert main(["--index", index, "index"]) == 0
    assert "44 total" in capsys.readouterr().out
