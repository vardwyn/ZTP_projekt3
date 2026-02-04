import json
from pathlib import Path

import pandas as pd
import pytest

from literature import pubmed_fetch


def test_load_config_missing_email(tmp_path):
    cfg = {
        "years": [2024],
        "pubmed": {
            "retmax": 10,
            "queries": [{"id": "q1", "term": "pm25"}],
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(AssertionError, match="pubmed.email"):
        pubmed_fetch.load_config(path)


def test_load_config_invalid_query_schema(tmp_path):
    cfg = {
        "years": [2024],
        "pubmed": {
            "email": "x@y.z",
            "retmax": 10,
            "queries": [{"id": "q1"}],
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(AssertionError, match="term"):
        pubmed_fetch.load_config(path)


def test_write_outputs_deterministic_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"query_id": "b", "query_term": "t", "pmid": "2", "title": "t2", "year": 2024, "journal": "J2", "authors": "A"},
        {"query_id": "a", "query_term": "t", "pmid": "3", "title": "t3", "year": 2024, "journal": "J1", "authors": "A"},
        {"query_id": "a", "query_term": "t", "pmid": "1", "title": "t1", "year": 2024, "journal": "J1", "authors": "A"},
    ]
    pubmed_fetch.write_outputs(rows, 2024)
    df = pd.read_csv("results/literature/2024/pubmed_papers.csv")

    assert df["query_id"].tolist() == ["a", "a", "b"]
    assert df["pmid"].astype(str).tolist() == ["1", "3", "2"]


def test_write_outputs_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pubmed_fetch.write_outputs([], 2024)

    papers = Path("results/literature/2024/pubmed_papers.csv")
    top = Path("results/literature/2024/top_journals.csv")
    summary = Path("results/literature/2024/summary_by_year.csv")

    assert papers.exists()
    assert top.exists()
    assert summary.exists()

    df = pd.read_csv(papers)
    assert df.empty


def test_run_queries_metadata_parsing(monkeypatch):
    class DummyHandle:
        def __init__(self, kind):
            self.kind = kind
        def close(self):
            return None

    def fake_esearch(db, term, retmax):
        return DummyHandle("esearch")

    def fake_esummary(db, id, retmode):
        return DummyHandle("esummary")

    def fake_read(handle):
        if handle.kind == "esearch":
            return {"IdList": ["1", "2"]}
        return [
            {
                "Id": "1",
                "Title": "T1",
                "PubDate": "2024 Dec",
                "FullJournalName": "J1",
                "AuthorList": ["A", "B"],
            },
            {
                "Id": "2",
                "Title": "T2",
                "PubDate": "2024 Nov",
                "FullJournalName": "J2",
                "AuthorList": ["C"],
            },
        ]

    monkeypatch.setattr(pubmed_fetch.Entrez, "esearch", fake_esearch)
    monkeypatch.setattr(pubmed_fetch.Entrez, "esummary", fake_esummary)
    monkeypatch.setattr(pubmed_fetch.Entrez, "read", fake_read)
    monkeypatch.setattr(pubmed_fetch.time, "sleep", lambda _: None)

    cfg = {
        "retmax": 10,
        "api_key": "",
        "queries": [{"id": "q1", "term": "pm25"}],
    }
    rows = pubmed_fetch.run_queries(cfg, 2024)

    assert len(rows) == 2
    assert rows[0]["authors"] == "A; B"
    assert rows[0]["year"] == 2024
    assert rows[0]["journal"] == "J1"


def test_aggregations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"query_id": "q1", "query_term": "t", "pmid": "1", "title": "t1", "year": 2024, "journal": "J1", "authors": "A"},
        {"query_id": "q1", "query_term": "t", "pmid": "2", "title": "t2", "year": 2024, "journal": "J1", "authors": "A"},
        {"query_id": "q2", "query_term": "t", "pmid": "3", "title": "t3", "year": 2024, "journal": "J2", "authors": "A"},
    ]
    pubmed_fetch.write_outputs(rows, 2024)

    top = pd.read_csv("results/literature/2024/top_journals.csv")
    summary = pd.read_csv("results/literature/2024/summary_by_year.csv")

    assert top.iloc[0]["journal"] == "J1"
    assert top.iloc[0]["count"] == 2
    assert set(summary["query_id"]) == {"q1", "q2"}
    assert summary.loc[summary["query_id"] == "q1", "count"].item() == 2
