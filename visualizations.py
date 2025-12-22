import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


def plot_monthly_avg_station_per_year(monthly_avg):
    for year, df_y in monthly_avg.groupby(monthly_avg.index.year):

        if not df_y.notna().any().any():
            continue

        plt.figure(figsize=(12, 6))
        months = df_y.index.month

        for station in df_y.columns:
            plt.plot(months, df_y[station], marker="o", label=str(station))

        plt.xticks(range(1, 13))
        plt.xlabel("Miesiąc")
        plt.ylabel("PM2.5 [mcg/m^3]")
        plt.title(f"Średnia miesięczna pomiarów dla stacji – rok {year}")
        plt.legend(title="Stacja", fontsize="small", ncol=2)
        plt.tight_layout()
        plt.show()


def plot_monthly_avg_station_mean_std_per_year(monthly_avg):
    for year, df_y in monthly_avg.groupby(monthly_avg.index.year):

        mean_all = df_y.mean(axis=1, skipna=True)
        std_all  = df_y.std(axis=1, skipna=True)


        if not mean_all.notna().any():
            continue

        months = mean_all.index.month

        plt.figure(figsize=(12, 6))
        
        plt.plot(months, mean_all, marker="o", label="Średnia stacji")

        plt.fill_between(
            months,
            mean_all - std_all,
            mean_all + std_all,
            alpha=0.3,
            label="±1 std dev"
        )

        plt.xticks(range(1, 13))
        plt.xlabel("Miesiąc")
        plt.ylabel("PM2.5 [mcg/m^3]")
        plt.title(f"Miesięczne średnie (połączone stacje) – rok {year}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_monthly_avg_station_comparison(monthly_avg):
    plt.figure(figsize=(12, 6))

    yearly_means = {}

    for year, df_y in monthly_avg.groupby(monthly_avg.index.year):
        mean_all = df_y.mean(axis=1, skipna=True)

        if not mean_all.notna().any():
            continue

        monthly_series = mean_all.groupby(mean_all.index.month).mean()
        yearly_means[year] = monthly_series

    for year, s in sorted(yearly_means.items()):
        months = s.index
        plt.plot(months, s.values, marker="o", label=str(year))

    plt.xticks(range(1, 13))
    plt.xlabel("Miesiąc")
    plt.ylabel("PM2.5 (średnia stacji) [mcg/m^3]")
    plt.title("Miesięczne średnie (połączone stacje) – porównanie lat")
    plt.legend(title="Rok")
    plt.tight_layout()
    plt.show()


def plot_monthly_avg_station_std_comparison(monthly_avg):
    plt.figure(figsize=(12, 6))

    yearly_std = {}

    for year, df_y in monthly_avg.groupby(monthly_avg.index.year):
        std_all = df_y.std(axis=1, skipna=True)

        if not std_all.notna().any():
            continue

        monthly_std_series = std_all.groupby(std_all.index.month).mean()
        yearly_std[year] = monthly_std_series

    for year, s in sorted(yearly_std.items()):
        months = s.index  # 1..12
        plt.plot(months, s.values, marker="o", label=str(year))

    plt.xticks(range(1, 13))
    plt.xlabel("Miesiąc")
    plt.ylabel("Std dev średnich miesięcznych")
    plt.title("Std dev średnich miesięcznych połączonych stacji – porównanie lat")
    plt.legend(title="Rok")
    plt.tight_layout()
    plt.show()


def plot_daily_sma_per_year(daily_sma):
    for year, df_y in daily_sma.groupby(daily_sma.index.year):

        if not df_y.notna().any().any():
            continue

        x = df_y.index.dayofyear

        plt.figure(figsize=(12, 6))
        for station in df_y.columns:
            plt.plot(x, df_y[station].values, label=str(station))

        plt.xlabel("Dzień roku")
        plt.ylabel("PM2.5 (30-dniowa SMA) [mcg/m^3]")
        plt.title(f"30-dniowa SMA – rok {year}")
        plt.legend(title="Stacja", fontsize="small", ncol=2)
        plt.tight_layout()
        plt.show()


def plot_daily_sma_mean_std_per_year(daily_sma):
    for year, df_y in daily_sma.groupby(daily_sma.index.year):
        mean_all = df_y.mean(axis=1, skipna=True)
        std_all  = df_y.std(axis=1, skipna=True)

        if not mean_all.notna().any():
            continue

        x = mean_all.index.dayofyear

        plt.figure()
        plt.plot(x, mean_all.values, label="Średnia stacji")
        plt.fill_between(
            x,
            (mean_all - std_all).values,
            (mean_all + std_all).values,
            alpha=0.3,
            label="±1 std dev",
        )

        plt.xlabel("Dzień roku")
        plt.ylabel("PM2.5 [mcg/m^3]")
        plt.title(f"30-dniowe SMA (połączone stacje) – rok {year}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_daily_sma_comparison_years(daily_sma):
    plt.figure()

    for year, df_y in daily_sma.groupby(daily_sma.index.year):
        mean_all = df_y.mean(axis=1, skipna=True)
        if not mean_all.notna().any():
            continue

        x = mean_all.index.dayofyear
        plt.plot(x, mean_all.values, marker=".", label=str(year))

    plt.xlabel("Dzień roku")
    plt.ylabel("PM2.5 [mcg/m^3]")
    plt.title("30-dniowa SMA – porównanie lat (średnia stacji)")
    plt.legend(title="Rok")
    plt.tight_layout()
    plt.show()


def plot_city_monthly_averages(monthly_avg_city, cities, years):

    df = monthly_avg_city.copy()
    df.index = pd.to_datetime(df.index)

    available_cities = set(df.columns.astype(str))
    missing = [c for c in cities if c not in available_cities]
    if missing:
        raise AssertionError(
            "Some requested cities are not in monthly_avg_city columns: "
            + ", ".join(missing)
        )

    plt.figure(figsize=(12, 6))

    for city in cities:
        series_city = df[city]

        for year in years:
            s_y = series_city[series_city.index.year == year]
            if not s_y.notna().any():
                continue

            months = s_y.index.month
            label = f"{city} {year}"
            plt.plot(months, s_y.values, marker="o", label=label)

    plt.xticks(range(1, 13))
    plt.xlabel("Miesiąc")
    plt.ylabel("PM2.5 [mcg/m^3]")
    plt.title("Średnie miesięczne dla miasta i roku")
    plt.legend(title="Miasto-Rok", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()


def plot_city_monthly_heatmaps(monthly_avg_city, cities, years):

    df = monthly_avg_city.copy()
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)

    years = sorted(set(years))


    available_cities = set(df.columns)
    missing_cities = [c for c in cities if c not in available_cities]
    if missing_cities:
        raise AssertionError(
            "Some requested cities are not in monthly_avg_city columns: "
            + ", ".join(missing_cities)
        )


    df = df[cities]
    df = df[df.index.year.isin(years)]


    monthly = df.copy()
    monthly.index = pd.MultiIndex.from_arrays(
        [monthly.index.year, monthly.index.month],
        names=["year", "month"],
    )

    if np.all(np.isnan(monthly.values)):
        raise ValueError("All monthly means are NaN; nothing to plot.")


    vmin = np.nanmin(monthly.values)
    vmax = np.nanmax(monthly.values)


    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad(color="lightgrey")  #NaNs

    for city in cities:
        s = monthly[city]

        heat = s.unstack(level="month")
        heat = heat.reindex(index=years, columns=range(1, 13))

        data = np.ma.masked_invalid(heat.values)
        nan_mask = np.isnan(heat.values)

        plt.figure()
        im = plt.imshow(
            data,
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        cbar = plt.colorbar(im, label="PM2.5 [mcg/m^3]")


        if nan_mask.any():
            ny, nx = np.where(nan_mask)
            plt.scatter(
                nx, ny,
                marker="x",
                color="black",
                s=40,
                linewidths=0.8,
            )

        plt.xticks(
            ticks=np.arange(0, 12),
            labels=range(1, 13),
        )
        plt.yticks(
            ticks=np.arange(0, len(years)),
            labels=heat.index,
        )

        plt.xlabel("Miesiąc")
        plt.ylabel("Rok")
        plt.title(f"Średnie miesięczne – {city}")

        nan_patch = Patch(
            facecolor="lightgrey",
            edgecolor="black",
            label="Brak danych (NaN)"
        )
        plt.legend(
            handles=[nan_patch],
            loc="center left",
            bbox_to_anchor=(1., -0.1),
            fontsize="small",
            frameon=True,
        )

        plt.tight_layout()
        plt.show()


def plot_extreme_stations_days_over(days_over_df, year_ref=2024,
                                    years=(2014, 2019, 2024), n=3):

    df = days_over_df.copy()

    if year_ref not in df.columns:
        raise ValueError(f"Reference year {year_ref} not in days_over_df.columns")

    for y in years:
        if y not in df.columns:
            raise ValueError(f"Year {y} not in days_over_df.columns")

    # n najniższycz i n najwyższych stacji w year_ref
    s_ref = df[year_ref]
    lowest = s_ref.nsmallest(n)
    highest = s_ref.nlargest(n)

    stations_sel = pd.Index(lowest.index.tolist() + highest.index.tolist()).unique()
    df_sel = df.loc[stations_sel, years]

    # grouped bar plot
    x = np.arange(len(stations_sel)) 
    n_years = len(years)
    width = 0.8 / n_years

    plt.figure(figsize=(12,6))
    for i, y in enumerate(years):
        x_shift = x + (i - (n_years - 1) / 2) * width
        plt.bar(x_shift, df_sel[y].values, width=width, label=str(y))

    plt.xticks(x, stations_sel, rotation=45, ha="right")
    plt.xlabel("Stacja")
    plt.ylabel("Liczba dni powyżej normy dobowej")
    plt.title(f"Liczba dni powyżej normy – najlepsze / najgorsze stacje z roku {year_ref}")
    plt.legend(title="Rok")
    plt.tight_layout()
    plt.show()
