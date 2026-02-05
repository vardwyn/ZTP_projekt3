import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generuje raport Markdown dla task4."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config musi być słownikiem.")
    return cfg


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Brak danych._"
    lines = []
    headers = [str(c) for c in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def load_exceedance_stats(years: list[int]) -> tuple[str, str]:
    rows = []
    for year in years:
        path = Path("results") / "pm25" / str(year) / "exceedance_days.csv"
        df = pd.read_csv(path, index_col=0)
        col = None
        if str(year) in df.columns:
            col = str(year)
        elif year in df.columns:
            col = year
        else:
            col = df.columns[0]
        s = df[col]
        rows.append(
            {
                "rok": year,
                "liczba_stacji": int(s.shape[0]),
                "średnia_przekroczeń": round(float(s.mean()), 2),
                "mediana_przekroczeń": round(float(s.median()), 2),
                "min_przekroczeń": int(s.min()),
                "max_przekroczeń": int(s.max()),
            }
        )
    stats_df = pd.DataFrame(rows).sort_values("rok")
    summary_text = (
        "Poniższa tabela prezentuje podstawowe statystyki liczby dni z "
        "przekroczeniem normy dobowej PM2.5 na stację w każdym roku."
    )
    return summary_text, df_to_markdown(stats_df)


def list_pm25_figures(years: list[int]) -> str:
    blocks = []
    skip_names = {
        "monthly_avg_station.png",
        "monthly_avg_station_comparison.png",
        "city_monthly_heatmaps.png",
    }
    for year in years:
        fig_dir = Path("results") / "pm25" / str(year) / "figures"
        if not fig_dir.exists():
            continue
        blocks.append(f"**Rok {year}**")
        for path in sorted(fig_dir.glob("*.png")):
            if path.name in skip_names:
                continue
            rel = Path("pm25") / str(year) / "figures" / path.name
            blocks.append(f"![{path.name}]({rel.as_posix()})")
        blocks.append("")
    return "\n".join(blocks).strip() if blocks else "_Brak wykresów._"


def load_literature_tables(years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = Path("results") / "literature" / str(year) / "summary_by_year.csv"
        df = pd.read_csv(path)
        frames.append(df)
    all_summary = pd.concat(frames, ignore_index=True)
    if all_summary.empty:
        cols = ["query_id"] + [str(y) for y in years]
        return pd.DataFrame(columns=cols)
    all_summary["year"] = all_summary["year"].astype(int)
    table = (
        all_summary.pivot_table(
            index="query_id",
            columns="year",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    table = table.reindex(columns=["query_id"] + years)
    table.columns = [str(c) for c in table.columns]
    return table


def plot_literature_trend(summary_df: pd.DataFrame, years: list[int], out_path: Path):
    if summary_df.empty:
        totals = pd.Series([0] * len(years), index=[str(y) for y in years])
    else:
        totals = (
            summary_df.set_index("query_id")
            .sum(numeric_only=True)
            .reindex([str(y) for y in years], fill_value=0)
        )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(years, totals.values, marker="o")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Liczba publikacji")
    ax.set_title("Trend liczby publikacji")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return totals


def load_top_journals(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for year in years:
        path = Path("results") / "literature" / str(year) / "pubmed_papers.csv"
        df = pd.read_csv(path)
        df["year"] = int(year)
        frames.append(df)
    all_papers = pd.concat(frames, ignore_index=True)
    if all_papers.empty:
        empty = pd.DataFrame(columns=["journal", "total"])
        return empty, empty

    totals = (
        all_papers["journal"]
        .fillna("UNKNOWN")
        .value_counts()
        .head(10)
        .reset_index()
    )
    totals.columns = ["journal", "total"]

    counts = (
        all_papers.assign(journal=all_papers["journal"].fillna("UNKNOWN"))
        .groupby(["journal", "year"])
        .size()
        .reset_index(name="count")
    )
    top_journals = totals["journal"].tolist()
    pivot = (
        counts[counts["journal"].isin(top_journals)]
        .pivot_table(
            index="journal",
            columns="year",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]
    table = totals.merge(pivot, on="journal", how="left")
    return table, counts[counts["journal"].isin(top_journals)]


def plot_top_journals_trend(counts_df: pd.DataFrame, years: list[int], out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    if counts_df.empty:
        ax.text(0.5, 0.5, "Brak danych", ha="center", va="center")
        ax.axis("off")
    else:
        short_names = {}
        used = set()
        for name in counts_df["journal"].unique():
            short = name if len(name) <= 30 else name[:27] + "…"
            base = short
            i = 2
            while short in used:
                suffix = f" ({i})"
                short = base[: max(0, 30 - len(suffix))] + suffix
                i += 1
            used.add(short)
            short_names[name] = short

        for journal, group in counts_df.groupby("journal"):
            series = (
                group.set_index("year")["count"]
                .reindex(years, fill_value=0)
            )
            ax.plot(
                years,
                series.values,
                marker="o",
                label=short_names.get(journal, journal),
            )
        ax.set_xlabel("Rok")
        ax.set_ylabel("Liczba publikacji")
        ax.set_title("Top czasopisma w czasie")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(
            fontsize="small",
            ncol=1,
            frameon=True,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sample_titles(years: list[int], max_n: int = 5) -> str:
    frames = []
    for year in years:
        path = Path("results") / "literature" / str(year) / "pubmed_papers.csv"
        df = pd.read_csv(path)
        frames.append(df)
    all_papers = pd.concat(frames, ignore_index=True)
    if all_papers.empty or "title" not in all_papers.columns:
        return "_Brak tytułów._"

    pmid_numeric = pd.to_numeric(all_papers["pmid"], errors="coerce")
    all_papers = all_papers.assign(_pmid_num=pmid_numeric)
    all_papers = all_papers.sort_values(
        ["_pmid_num", "pmid"], na_position="last"
    )
    titles = all_papers["title"].dropna().astype(str).head(max_n).tolist()
    if not titles:
        return "_Brak tytułów._"
    return "\n".join(f"- {t}" for t in titles)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    years = [int(y) for y in cfg.get("years", [])]
    years = sorted(set(years))

    template = args.template.read_text(encoding="utf-8")

    pm25_summary_text, pm25_table = load_exceedance_stats(years)
    pm25_summary = pm25_summary_text + "\n\n" + pm25_table
    pm25_figures = list_pm25_figures(years)

    literature_table_df = load_literature_tables(years)
    literature_table = df_to_markdown(literature_table_df)

    report_dir = args.output.parent
    figures_dir = report_dir / "report_task4" / "figures"

    trend_plot_path = figures_dir / "literature_counts_by_year.png"
    totals = plot_literature_trend(literature_table_df, years, trend_plot_path)
    trend_desc = (
        f"Łączna liczba publikacji w pierwszym roku: {int(totals.iloc[0])}, "
        f"a w ostatnim: {int(totals.iloc[-1])}."
        if len(totals) > 1
        else f"Łączna liczba publikacji w roku {years[0]}: {int(totals.iloc[0])}."
    )

    top_journals_table_df, top_journals_counts = load_top_journals(years)
    top_journals_table = df_to_markdown(top_journals_table_df)
    top_journals_plot_path = figures_dir / "top_journals_trend.png"
    plot_top_journals_trend(top_journals_counts, years, top_journals_plot_path)

    titles_block = sample_titles(years, max_n=5)

    filled = template.format(
        pm25_summary=pm25_summary,
        pm25_figures=pm25_figures,
        literature_table=literature_table,
        literature_trend_description=trend_desc,
        literature_trend_plot=(Path("report_task4") / "figures" / trend_plot_path.name).as_posix(),
        top_journals_table=top_journals_table,
        top_journals_trend_plot=(Path("report_task4") / "figures" / top_journals_plot_path.name).as_posix(),
        sample_titles=titles_block,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(filled, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
