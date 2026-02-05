import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pm25.data_preparation import (
    download_gios_archive,
    download_updated_metadata,
    clean_raw_year,
    combine_metadata_frames,
    combine_data_frames,
)
from pm25.statistics_calculation import (
    monthly_avg_with_nan_threshold,
    average_by_city,
    count_days_over_threshold,
)
from pm25.visualizations import (
    plot_monthly_avg_station_per_year,
    plot_monthly_avg_station_mean_std_per_year,
    plot_monthly_avg_station_comparison,
    plot_city_monthly_averages,
    plot_city_monthly_heatmaps,
    plot_extreme_stations_days_over,
    plot_pm25_days_over_by_voivodeship_years,
)

matplotlib.use("Agg")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generuje wyniki PM2.5 dla pojedynczego roku."
    )
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    assert isinstance(cfg, dict), "Config musi być słownikiem."
    pm25_cfg_file = cfg.get("pm25_config_file")
    if pm25_cfg_file:
        pm25_path = (Path.cwd() / pm25_cfg_file).resolve()
        with pm25_path.open("r", encoding="utf-8") as f:
            cfg["pm25"] = yaml.safe_load(f) or {}
    return cfg


def save_figure(path: Path) -> None:
    fig = plt.gcf()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ensure_cleaned_data(year: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm25_cfg = cfg.get("pm25", {})
    raw_dir = Path(pm25_cfg.get("raw_data_dir", "data/raw_data"))
    cleaned_dir = Path(pm25_cfg.get("cleaned_data_dir", "data/cleaned_data"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    cleaned_data_path = cleaned_dir / f"data_{year}.csv.xz"
    cleaned_meta_path = cleaned_dir / f"metadata_{year}.csv.xz"

    if not (cleaned_data_path.exists() and cleaned_meta_path.exists()):
        raw_path = raw_dir / f"raw_data_{year}.csv.xz"
        if not raw_path.exists():
            archives = pm25_cfg.get("archives", {})
            archive = archives.get(year) or archives.get(str(year))
            assert archive, (
                f"Brak archiwum dla roku {year} w config.pm25.archives "
                "i brak lokalnego raw_data."
            )
            raw_df = download_gios_archive(
                archive_url=archive["url"],
                filename=archive["filename"],
                sha256=archive.get("sha256"),
            )
            raw_df.to_csv(raw_path)

        raw_df = pd.read_csv(raw_path, header=0, index_col=0)

        updated_meta_path = pm25_cfg.get("updated_metadata_path")
        if not updated_meta_path:
            raise ValueError("Brak updated_metadata_path w config.pm25.")
        updated_meta_path = Path(updated_meta_path)
        if not updated_meta_path.exists():
            updated_meta_path.parent.mkdir(parents=True, exist_ok=True)
            updated_metadata_df = download_updated_metadata(
                pm25_cfg["updated_metadata_url"],
                sha256=pm25_cfg.get("updated_metadata_sha256"),
            )
            updated_metadata_df.to_csv(updated_meta_path, index=False)

        updated_metadata_df = pd.read_csv(updated_meta_path)
        extra_cols = pm25_cfg.get(
            "extra_metadata_cols", ["Miejscowość", "Województwo"]
        )
        rename_geo = pm25_cfg.get("rename_geo", {})
        data_updated, meta_extended = clean_raw_year(
            raw_df=raw_df,
            updated_metadata_df=updated_metadata_df,
            year=year,
            extra_cols=extra_cols,
            rename_geo=rename_geo,
        )
        meta_extended.to_csv(cleaned_meta_path, index=False)
        data_updated.to_csv(cleaned_data_path)

    data_df = pd.read_csv(
        cleaned_data_path, index_col=0, parse_dates=True, low_memory=False
    )
    data_df.index = pd.to_datetime(data_df.index, format="mixed")
    meta_df = pd.read_csv(cleaned_meta_path)
    return data_df, meta_df


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    data_df, meta_df = ensure_cleaned_data(args.year, cfg)
    pm25_cfg = cfg.get("pm25", {})
    threshold = pm25_cfg.get("daily_threshold", 15)
    max_nan_per_month = pm25_cfg.get("max_nan_per_month", 240)
    exclude_stations = pm25_cfg.get("exclude_stations", [])

    results_dir = Path("results") / "pm25" / str(args.year)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_station = figures_dir / "monthly_avg_station.png"
    fig_mean_std = figures_dir / "monthly_avg_station_mean_std.png"
    fig_comparison = figures_dir / "monthly_avg_station_comparison.png"
    fig_city_avg = figures_dir / "city_monthly_averages.png"
    fig_city_heat = figures_dir / "city_monthly_heatmaps.png"
    fig_extreme = figures_dir / "extreme_stations_days_over.png"
    fig_voiv = figures_dir / "voivodeship_days_over.png"

    # 1) Łączenie ramek
    metadata_combined, _ = combine_metadata_frames([meta_df])
    data_combined, _ = combine_data_frames([data_df])
    data_combined.index = pd.to_datetime(data_combined.index, format="mixed")

    # 2) Przekroczenia normy dobowej
    if exclude_stations:
        data_for_days = data_combined.drop(columns=exclude_stations, errors="ignore")
    else:
        data_for_days = data_combined

    days_over = count_days_over_threshold(
        data_for_days, threshold=threshold, years=(args.year,)
    )
    days_over.to_csv(results_dir / "exceedance_days.csv")

    # 3) Średnie miesięczne stacji
    monthly_avg_station = monthly_avg_with_nan_threshold(
        data_combined, max_nan_per_month=max_nan_per_month
    )
    plot_monthly_avg_station_per_year(
        monthly_avg_station,
        years_order=[args.year],
    )
    save_figure(fig_station)

    plot_monthly_avg_station_mean_std_per_year(
        monthly_avg_station,
        years_order=[args.year],
    )
    save_figure(fig_mean_std)

    plot_monthly_avg_station_comparison(
        monthly_avg_station,
        years_order=[args.year],
    )
    save_figure(fig_comparison)

    # 4) Średnie miesięczne po miastach
    if not {"Kod stacji", "Miejscowość"}.issubset(metadata_combined.columns):
        raise ValueError("Brak kolumn wymaganych do agregacji po miastach.")

    city_data_combined = average_by_city(data_combined, metadata_combined)
    monthly_avg_city = monthly_avg_with_nan_threshold(
        city_data_combined, max_nan_per_month=max_nan_per_month
    )

    available_cities = city_data_combined.columns.to_list()
    preferred_cities = pm25_cfg.get("city_compare_cities", ["Warszawa", "Katowice"])
    cities_for_line = [c for c in preferred_cities if c in available_cities]
    if not cities_for_line:
        cities_for_line = available_cities[:2]

    years_available = sorted(set(monthly_avg_city.index.year))
    preferred_years = pm25_cfg.get("city_compare_years", [args.year])
    years_for_line = [y for y in preferred_years if y in years_available]
    if not years_for_line:
        years_for_line = [args.year]

    if not cities_for_line:
        raise ValueError("Brak miast do wykresu city_monthly_averages.")

    plot_city_monthly_averages(
        monthly_avg_city,
        cities=cities_for_line,
        years=years_for_line,
    )
    save_figure(fig_city_avg)

    plot_city_monthly_heatmaps(
        monthly_avg_city,
        cities=available_cities,
        years=years_available,
        ncols=pm25_cfg.get("city_heatmap_ncols", 2),
    )
    save_figure(fig_city_heat)

    # 5) Skrajne stacje i przekroczenia po województwach
    if not {"Kod stacji", "Miejscowość"}.issubset(metadata_combined.columns):
        raise ValueError("Brak kolumn wymaganych do wykresu skrajnych stacji.")

    plot_extreme_stations_days_over(
        days_over,
        year_ref=args.year,
        years=(args.year,),
        station_metadata=metadata_combined,
    )
    save_figure(fig_extreme)

    if "Województwo" not in metadata_combined.columns:
        raise ValueError("Brak kolumny 'Województwo' do wykresu województw.")

    plot_pm25_days_over_by_voivodeship_years(
        meas_df=data_combined,
        metadata_df=metadata_combined,
        years=[args.year],
        threshold=threshold,
    )
    save_figure(fig_voiv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
