import argparse
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import yaml
from Bio import Entrez


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pobieranie metadanych z PubMed dla wskazanego roku."
    )
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def _load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Brak pliku config: {path}")
    except yaml.YAMLError as e:
        raise AssertionError(f"Niepoprawny YAML w {path}: {e}")
    return cfg


def load_config(config_path: Path) -> dict:
    cfg = _load_yaml(config_path)

    assert isinstance(cfg, dict), "Config musi być słownikiem."
    assert "years" in cfg and isinstance(cfg["years"], list), "Brak listy years."
    assert "pubmed" in cfg and isinstance(cfg["pubmed"], dict), "Brak sekcji pubmed."

    pubmed = cfg["pubmed"]
    assert pubmed.get("email"), "pubmed.email jest wymagane przez NCBI."
    assert isinstance(pubmed.get("retmax", 0), int), "pubmed.retmax musi być int."
    queries = pubmed.get("queries")
    queries_file = pubmed.get("queries_file")
    if queries_file:
        queries_path = (Path.cwd() / queries_file).resolve()
        queries_cfg = _load_yaml(queries_path)
        assert isinstance(queries_cfg, dict), "Plik zapytań musi być słownikiem."
        queries = queries_cfg.get("queries")

    assert isinstance(queries, list), "pubmed.queries musi być listą."
    assert queries, "pubmed.queries nie może być puste."

    for q in queries:
        assert isinstance(q, dict), "Każde zapytanie musi być słownikiem."
        assert q.get("id"), "Każde zapytanie musi mieć id."
        assert q.get("term"), "Zapytanie musi mieć pole term."

    pubmed["queries"] = queries

    return cfg


def build_year_query(term: str, year: int) -> str:
    return f"({term}) AND (\"{year}\"[dp] : \"{year}\"[dp])"


def extract_year(pubdate: str) -> int | None:
    if not pubdate:
        return None
    match = re.search(r"(19|20)\d{2}", str(pubdate))
    if not match:
        return None
    return int(match.group(0))


def run_queries(pubmed_cfg: dict, year: int) -> list[dict]:
    retmax = pubmed_cfg.get("retmax", 200)
    sleep_s = 0.1 if pubmed_cfg.get("api_key") else 0.34

    results = []
    for q in pubmed_cfg["queries"]:
        term = build_year_query(q["term"], year)
        handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
        search = Entrez.read(handle)
        handle.close()

        pmids = search.get("IdList", [])
        if pmids:
            # NCBI: bez API key max 3 req/s, z API key do 10 req/s
            time.sleep(sleep_s)
            summary_handle = Entrez.esummary(
                db="pubmed",
                id=",".join(pmids),
                retmode="xml",
            )
            summary = Entrez.read(summary_handle)
            summary_handle.close()

            for doc in summary:
                authors = doc.get("AuthorList", [])
                pub_year = extract_year(doc.get("PubDate"))
                if pub_year != year:
                    warnings.warn(
                        "Pomijam rekord o roku różnym od CLI "
                        f"(PMID={doc.get('Id')}, PubDate={doc.get('PubDate')})"
                    )
                    continue
                results.append(
                    {
                        "pmid": str(doc.get("Id")),
                        "title": doc.get("Title"),
                        "year": pub_year,
                        "journal": doc.get("FullJournalName"),
                        "authors": "; ".join(authors),
                        "query_id": q["id"],
                        "query_term": term,
                    }
                )

        # przerwa między zapytaniami (esearch/esummary)
        time.sleep(sleep_s)

    return results


def write_outputs(rows: list[dict], year: int) -> None:
    output_dir = Path("results") / "literature" / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    papers_path = output_dir / "pubmed_papers.csv"
    top_journals_path = output_dir / "top_journals.csv"
    summary_path = output_dir / "summary_by_year.csv"

    columns = [
        "query_id",
        "query_term",
        "pmid",
        "title",
        "year",
        "journal",
        "authors",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["pmid"] = df["pmid"].astype(str)
        df = df.sort_values(["query_id", "pmid"]).reset_index(drop=True)
    df.to_csv(papers_path, index=False)

    if df.empty:
        pd.DataFrame(columns=["journal", "count"]).to_csv(
            top_journals_path, index=False
        )
        pd.DataFrame(columns=["query_id", "year", "count"]).to_csv(
            summary_path, index=False
        )
        return

    top_journals = (
        df["journal"]
        .fillna("UNKNOWN")
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_journals.columns = ["journal", "count"]
    top_journals.to_csv(top_journals_path, index=False)

    summary = (
        df.groupby("query_id")
        .size()
        .reset_index(name="count")
    )
    summary["year"] = year
    summary = summary[["query_id", "year", "count"]]
    summary.to_csv(summary_path, index=False)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    pubmed = cfg["pubmed"]
    Entrez.email = pubmed["email"]
    if pubmed.get("tool"):
        Entrez.tool = pubmed["tool"]
    if pubmed.get("api_key"):
        Entrez.api_key = pubmed["api_key"]

    results = run_queries(pubmed, args.year)

    write_outputs(results, args.year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
