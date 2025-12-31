import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
from matplotlib import gridspec
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


def plot_monthly_avg_station_per_year(
    monthly_avg,
    main_title=None,
    main_title_fontsize=None,
    main_title_y=0.98,
    main_title_top=None,
    years_order=None,
    ncols=1,
    panel_size=(10, 4),
    hspace=0.3,
    wspace=0.0,
    marker_size=5,
    label_fontsize=10,
    title_fontsize=12,
    legend_fontsize="large",
    legend_title="Stacja",
    legend_title_fontsize=None,
    legend_frame=True,
    legend_mode="right",
    legend_ncol=2,
    right_margin=0.99,
    sharey=True,
    sharex=False,
):
    """
    Rysuje miesięczne średnie stacji w postaci jednego wykresu z wieloma panelami.

    Parametry:
    - main_title: tytuł całego wykresu (None = brak tytułu)
    - main_title_fontsize: rozmiar tytułu głównego (domyślnie jak title_fontsize)
    - main_title_y: pozycja tytułu głównego na osi Y (0..1)
    - main_title_top: opcjonalny górny margines figury (0..1) przy tytule
    - years_order: kolejność lat (lista/iterowalne), domyślnie rosnąco
    - ncols: liczba kolumn w układzie paneli (>= 1)
    - panel_size: rozmiar pojedynczego panelu (width, height)
    - hspace / wspace: odstępy między panelami
    - marker_size: rozmiar markerów punktów
    - label_fontsize: rozmiar etykiet osi
    - title_fontsize: rozmiar tytułów paneli (oraz main_title)
    - legend_fontsize: rozmiar legendy
    - legend_title: tytuł legendy
    - legend_title_fontsize: rozmiar tytułu legendy (domyślnie jak legend_fontsize)
    - legend_frame: czy rysować ramkę wokół legendy
    - legend_mode: tryb legendy:
        * "last"   – legenda tylko na ostatnim panelu,
        * "axis"   – legenda na każdym panelu,
        * "figure" – jedna legenda u góry całej figury,
        * "right"  – jedna legenda po prawej stronie figury,
        * "none"   – brak legendy.
    - legend_ncol: liczba kolumn w legendzie
    - right_margin: prawy margines
    - sharex / sharey: współdzielenie osi między panelami
    """

    if monthly_avg.empty:
        return

    # Upewnij się, że indeks jest typu datetime
    df = monthly_avg.copy()
    df.index = pd.to_datetime(df.index)

    # Grupowanie po latach i ustalenie kolejności (tylko lata z danymi)
    groups = {year: df_y for year, df_y in df.groupby(df.index.year)}
    candidate_years = years_order or sorted(groups.keys())
    years = [
        year for year in candidate_years
        if year in groups and groups[year].notna().any().any()
    ]
    if not years:
        return

    # Układ paneli
    ncols = max(1, int(ncols))
    nrows = int((len(years) + ncols - 1) / ncols)
    fig_w = panel_size[0] * ncols
    fig_h = panel_size[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex,
        sharey=sharey,
    )
    axes = (axes if isinstance(axes, (list, np.ndarray)) else [axes])
    axes = list(np.array(axes).ravel())
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # Tytuł całego wykresu (opcjonalnie)
    if main_title:
        fig.suptitle(
            main_title,
            fontsize=main_title_fontsize or title_fontsize,
            x=0.5,
            y=main_title_y,
            ha="center",
        )
        if main_title_top is not None:
            fig.subplots_adjust(top=main_title_top)

    # Gromadzimy etykiety do legendy zbiorczej (jeśli potrzebna)
    legend_handles = {}

    for ax, year in zip(axes, years):
        df_y = groups[year]

        # Oś X = miesiące (1..12), oś Y = wartości stacji
        months = df_y.index.month
        for station in df_y.columns:
            line = ax.plot(
                months,
                df_y[station],
                marker="o",
                markersize=marker_size,
                label=str(station),
            )[0]
            legend_handles[str(station)] = line

        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Miesiąc", fontsize=label_fontsize)
        ax.set_ylabel("PM2.5 [mcg/m^3]", fontsize=label_fontsize)
        ax.set_title(f"Rok {year}", fontsize=title_fontsize)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if legend_mode == "axis":
            leg = ax.legend(
                title=legend_title,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                frameon=legend_frame,
            )
            if legend_title_fontsize:
                leg.get_title().set_fontsize(legend_title_fontsize)

    # Ukryj niewykorzystane panele (gdy ncols > 1 i liczba lat nie wypełnia siatki)
    for ax in axes[len(years):]:
        ax.set_visible(False)

    if legend_mode == "last" and legend_handles and years:
        leg = axes[len(years) - 1].legend(
            title=legend_title,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            frameon=legend_frame,
        )
        if legend_title_fontsize:
            leg.get_title().set_fontsize(legend_title_fontsize)
    elif legend_mode == "right" and legend_handles:
        # Legenda w prawym marginesie figury (wewnątrz obszaru 0..1)
        leg = fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            title=legend_title,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            loc="center left",
            bbox_to_anchor=(right_margin + 0.01, 0.5),
            frameon=legend_frame,
        )
        if legend_title_fontsize:
            leg.get_title().set_fontsize(legend_title_fontsize)
        fig.subplots_adjust(right=right_margin)
    elif legend_mode == "figure" and legend_handles:
        leg = fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            title=legend_title,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=legend_frame,
        )
        if legend_title_fontsize:
            leg.get_title().set_fontsize(legend_title_fontsize)
    # legend_mode == "none" -> brak legendy
    plt.show()


def plot_monthly_avg_station_mean_std_per_year(
    monthly_avg,
    main_title=None,
    main_title_fontsize=None,
    main_title_y=0.98,
    main_title_top=None,
    years_order=None,
    ncols=1,
    panel_size=(10, 4),
    hspace=0.3,
    wspace=0.0,
    marker_size=5,
    label_fontsize=10,
    title_fontsize=12,
    legend_fontsize="large",
    legend_title="Legenda",
    legend_title_fontsize=None,
    legend_frame=True,
    legend_mode="last",
    legend_ncol=1,
    right_margin=0.99,
    sharey=True,
    sharex=False,
):
    """
    Rysuje miesięczne średnie i odchylenia standardowe (po stacjach) dla każdego roku
    jako jeden wykres z wieloma panelami.

    Parametry:
    - main_title: tytuł całego wykresu (None = brak tytułu)
    - main_title_fontsize: rozmiar tytułu głównego (domyślnie jak title_fontsize)
    - main_title_y: pozycja tytułu głównego na osi Y (0..1)
    - main_title_top: opcjonalny górny margines figury (0..1) przy tytule
    - years_order: kolejność lat (lista/iterowalne), domyślnie rosnąco
    - ncols: liczba kolumn w układzie paneli (>= 1)
    - panel_size: rozmiar pojedynczego panelu (width, height)
    - hspace / wspace: odstępy między panelami
    - marker_size: rozmiar markerów punktów
    - label_fontsize: rozmiar etykiet osi
    - title_fontsize: rozmiar tytułów paneli (oraz main_title)
    - legend_fontsize: rozmiar legendy
    - legend_title: tytuł legendy
    - legend_title_fontsize: rozmiar tytułu legendy (domyślnie jak legend_fontsize)
    - legend_frame: czy rysować ramkę wokół legendy
    - legend_mode: tryb legendy:
        * "last"   – legenda tylko na ostatnim panelu,
        * "axis"   – legenda na każdym panelu,
        * "figure" – jedna legenda u góry całej figury,
        * "right"  – jedna legenda po prawej stronie figury,
        * "none"   – brak legendy.
    - legend_ncol: liczba kolumn w legendzie
    - right_margin: prawy margines
    - sharex / sharey: współdzielenie osi między panelami
    """

    if monthly_avg.empty:
        return

    df = monthly_avg.copy()
    df.index = pd.to_datetime(df.index)

    groups = {year: df_y for year, df_y in df.groupby(df.index.year)}
    candidate_years = years_order or sorted(groups.keys())
    years = [
        year for year in candidate_years
        if year in groups and groups[year].notna().any().any()
    ]
    if not years:
        return

    ncols = max(1, int(ncols))
    nrows = int((len(years) + ncols - 1) / ncols)
    fig_w = panel_size[0] * ncols
    fig_h = panel_size[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex,
        sharey=sharey,
    )
    axes = (axes if isinstance(axes, (list, np.ndarray)) else [axes])
    axes = list(np.array(axes).ravel())
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if main_title:
        fig.suptitle(
            main_title,
            fontsize=main_title_fontsize or title_fontsize,
            x=0.5,
            y=main_title_y,
            ha="center",
        )
        if main_title_top is not None:
            fig.subplots_adjust(top=main_title_top)

    legend_handles = {}
    mean_label = "Średnia stacji"
    std_label = "±1 std dev"

    for ax, year in zip(axes, years):
        df_y = groups[year]
        mean_all = df_y.mean(axis=1, skipna=True)
        std_all = df_y.std(axis=1, skipna=True)

        if not mean_all.notna().any():
            ax.set_visible(False)
            continue

        months = mean_all.index.month

        line = ax.plot(
            months,
            mean_all,
            marker="o",
            markersize=marker_size,
            label=mean_label,
        )[0]
        fill = ax.fill_between(
            months,
            (mean_all - std_all),
            (mean_all + std_all),
            alpha=0.3,
            label=std_label,
        )

        legend_handles[mean_label] = line
        legend_handles[std_label] = Patch(facecolor=fill.get_facecolor()[0], alpha=0.3, label=std_label)

        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Miesiąc", fontsize=label_fontsize)
        ax.set_ylabel("PM2.5 [mcg/m^3]", fontsize=label_fontsize)
        ax.set_title(f"Rok {year}", fontsize=title_fontsize)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if legend_mode == "axis":
            leg = ax.legend(
                title=legend_title,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                frameon=legend_frame,
            )
            if legend_title_fontsize:
                leg.get_title().set_fontsize(legend_title_fontsize)

    for ax in axes[len(years):]:
        ax.set_visible(False)

    if legend_mode == "last" and legend_handles and years:
        leg = axes[len(years) - 1].legend(
            title=legend_title,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            frameon=legend_frame,
        )
        if legend_title_fontsize:
            leg.get_title().set_fontsize(legend_title_fontsize)
    elif legend_mode == "right" and legend_handles:
        leg = fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            title=legend_title,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            loc="center left",
            bbox_to_anchor=(right_margin + 0.01, 0.5),
            frameon=legend_frame,
        )
        if legend_title_fontsize:
            leg.get_title().set_fontsize(legend_title_fontsize)
        fig.subplots_adjust(right=right_margin)
    elif legend_mode == "figure" and legend_handles:
        leg = fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            title=legend_title,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=legend_frame,
        )
        if legend_title_fontsize:
            leg.get_title().set_fontsize(legend_title_fontsize)
    # legend_mode == "none" -> brak legendy
    plt.show()


def plot_monthly_avg_station_comparison(
    monthly_avg,
    figsize=(12, 6),
    years_order=None,
    marker_size=5,
    label_fontsize=10,
    title_fontsize=12,
    legend_fontsize="large",
    legend_title="Rok",
    legend_title_fontsize=None,
    legend_frame=True,
):
    """
    Porównuje miesięczne średnie (po stacjach) dla wielu lat na jednym wykresie.

    Dodatkowo pokazuje pasmo +-1 std (odchylenie standardowe między stacjami)
    jako wypełnienie wokół krzywej średniej dla każdego roku.
    """

    if monthly_avg.empty:
        return

    df = monthly_avg.copy()
    df.index = pd.to_datetime(df.index)

    groups = {year: df_y for year, df_y in df.groupby(df.index.year)}
    candidate_years = years_order or sorted(groups.keys())
    years = [
        year for year in candidate_years
        if year in groups and groups[year].notna().any().any()
    ]

    if not years:
        return

    plt.figure(figsize=figsize)

    # Customowa colormapa, wyraźne, nasycone kolory dla 1-8 serii
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
    ]
    if len(years) > len(colors):
        raise ValueError(
            "plot_monthly_avg_station_comparison obsługuje max 1-8 serii / lat. "
            f"Podano {len(years)}."
        )

    for idx, year in enumerate(years):
        df_y = groups.get(year)
        if df_y is None:
            continue

        mean_all = df_y.mean(axis=1, skipna=True)
        std_all = df_y.std(axis=1, skipna=True)

        if not mean_all.notna().any():
            continue

        monthly_mean = mean_all.groupby(mean_all.index.month).mean()
        monthly_std = std_all.groupby(std_all.index.month).mean()

        months = monthly_mean.index
        color = colors[idx]

        plt.plot(
            months,
            monthly_mean.values,
            marker="o",
            markersize=marker_size,
            label=str(year),
            color=color,
        )
        plt.fill_between(
            months,
            (monthly_mean - monthly_std).values,
            (monthly_mean + monthly_std).values,
            alpha=0.2,
            color=color,
        )

    plt.xticks(range(1, 13))
    plt.xlabel("Miesiąc", fontsize=label_fontsize)
    plt.ylabel("PM2.5 (średnia stacji) [mcg/m^3]", fontsize=label_fontsize)
    plt.title("Miesięczne średnie (połączone stacje) – porównanie lat",
              fontsize=title_fontsize)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    leg = plt.legend(title=legend_title, fontsize=legend_fontsize, frameon=legend_frame)
    if legend_title_fontsize:
        leg.get_title().set_fontsize(legend_title_fontsize)
    plt.tight_layout()
    plt.show()


def plot_city_monthly_averages(
    monthly_avg_city,
    cities,
    years,
    figsize=(12, 6),
    marker_size=5,
    line_width=2.0,
    label_fontsize=10,
    title_fontsize=12,
    legend_fontsize="large",
    legend_title="Miasto–Rok",
    legend_title_fontsize=None,
    legend_frame=True,
    city_colors=None,
    year_alphas=None,
):
    """
    Rysuje średnie miesięczne dla wybranych miast i lat na jednym wykresie.

    - Kolor rozróżnia miasto (bazowe: czerwony, zielony, niebieski, żółty, fioletowy).
    - Rok rozróżniany jest przez wariant koloru: bazowy → pastelowy → neonowy.
    - Opcjonalnie można też sterować przezroczystością (alpha).
    - Dozwolone: 1–4 miasta i 1–3 lata (dla czytelności).
    """

    df = monthly_avg_city.copy()
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)

    if not (1 <= len(cities) <= 4):
        raise ValueError("Dozwolone są 1–4 miasta.")
    if not (1 <= len(years) <= 3):
        raise ValueError("Dozwolone są 1–3 lata.")

    available_cities = set(df.columns)
    missing = [c for c in cities if c not in available_cities]
    if missing:
        raise AssertionError(
            "Brakujące miasta w monthly_avg_city columns: "
            + ", ".join(missing)
        )

    # Kolory miast (bazowe)
    default_city_colors = ["#e41a1c", "#2ca02c", "#1f77b4", "#ffd200", "#6a3d9a"]
    city_colors = city_colors or default_city_colors
    if len(city_colors) < len(cities):
        raise ValueError("Za mało kolorów dla podanych miast.")

    # Alpha dla lat (opcjonalnie)
    if year_alphas is None:
        year_alphas = [1.0] * len(years)
    if len(year_alphas) < len(years):
        raise ValueError("Za mało wartości alpha dla podanych lat.")

    plt.figure(figsize=figsize)

    def _pastelize(rgb, mix=0.65):
        # Mieszamy z bielą, by uzyskać pastel
        return tuple((1 - mix) * c + mix * 1.0 for c in rgb)

    def _neonize(rgb):
        # Wzmacniamy nasycenie i lekko podnosimy jasność
        h, l, s = colorsys.rgb_to_hls(*rgb)
        s = min(1.0, s * 1.9 + 0.2)
        l = min(1.0, max(0.0, l * 0.9 + 0.15))
        return colorsys.hls_to_rgb(h, l, s)

    # Iteracja: miasto -> rok (kolor = miasto, wariant = rok)
    for city_idx, city in enumerate(cities):
        series_city = df[city]
        color = city_colors[city_idx]
        base_rgb = to_rgb(color)

        for year_idx, year in enumerate(years):
            s_y = series_city[series_city.index.year == year]
            if not s_y.notna().any():
                continue

            months = s_y.index.month
            label = f"{city} {year}"

            if year_idx == 0:
                year_color = base_rgb
            elif year_idx == 1:
                year_color = _pastelize(base_rgb)
            else:
                year_color = _neonize(base_rgb)

            plt.plot(
                months,
                s_y.values,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=year_color,
                alpha=year_alphas[year_idx],
                label=label,
            )

    plt.xticks(range(1, 13))
    plt.xlabel("Miesiąc", fontsize=label_fontsize)
    plt.ylabel("PM2.5 [mcg/m^3]", fontsize=label_fontsize)
    plt.title("Średnie miesięczne dla miasta i roku", fontsize=title_fontsize)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    leg = plt.legend(
        title=legend_title,
        fontsize=legend_fontsize,
        ncol=2,
        frameon=legend_frame,
    )
    if legend_title_fontsize:
        leg.get_title().set_fontsize(legend_title_fontsize)
    plt.tight_layout()
    plt.show()


def plot_city_monthly_heatmaps(
    monthly_avg_city,
    cities,
    years,
    ncols=1,
    figsize=None,
    side_width=0.03,
    legend_height=0.04,
):
    """
    Rysuje mapy cieplne średnich miesięcznych dla miast w układzie podwykresów.

    - Po prawej stronie każdego wiersza znajduje się osobna skala kolorów,
      a pod nią legenda oznaczająca braki danych (NaN).
    - Układ może mieć 1 lub 2 kolumny paneli (parametr ncols).

    Parametry:
    - ncols: liczba kolumn paneli (1 lub 2)
    - figsize: rozmiar całej figury (domyślnie wyliczany)
    - side_width: względna szerokość kolumny po prawej (względem mapy)
    - legend_height: wysokość legendy w jednostkach figury (0..1)
    """

    df = monthly_avg_city.copy()
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)

    years = sorted(set(years))

    if ncols not in (1, 2):
        raise ValueError("ncols może być równe 1 albo 2.")

    # Walidacja listy miast
    available_cities = set(df.columns)
    missing_cities = [c for c in cities if c not in available_cities]
    if missing_cities:
        raise AssertionError(
            "Brakujące miasta w monthly_avg_city columns: "
            + ", ".join(missing_cities)
        )

    df = df[cities]
    df = df[df.index.year.isin(years)]

    # Indeks wielopoziomowy: (rok, miesiąc)
    monthly = df.copy()
    monthly.index = pd.MultiIndex.from_arrays(
        [monthly.index.year, monthly.index.month],
        names=["year", "month"],
    )

    if np.all(np.isnan(monthly.values)):
        raise ValueError("Wszytkie średnie są NaN!")

    # Wspólna skala kolorów dla wszystkich miast
    vmin = np.nanmin(monthly.values)
    vmax = np.nanmax(monthly.values)

    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad(color="lightgrey")  # NaN

    nrows = int(np.ceil(len(cities) / ncols))
    if figsize is None:
        heatmap_width = 5.2
        side_width_abs = heatmap_width * side_width
        fig_width = (heatmap_width + side_width_abs) * ncols
        fig_height = max(4.2, 4.2 * nrows)
        figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)
    main_gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols + 1,
        figure=fig,
        width_ratios=[1.0] * ncols + [side_width],
        wspace=0.25,
        hspace=0.35,
    )

    nan_patch = Patch(
        facecolor="lightgrey",
        edgecolor="black",
        label="Brak danych (NaN)",
    )

    row_axes = [[] for _ in range(nrows)]
    row_cax = []
    for row in range(nrows):
        row_cax.append(fig.add_subplot(main_gs[row, ncols]))

    for idx, city in enumerate(cities):
        row = idx // ncols
        col = idx % ncols

        ax = fig.add_subplot(main_gs[row, col])
        row_axes[row].append(ax)
        s = monthly[city]

        heat = s.unstack(level="month")
        heat = heat.reindex(index=years, columns=range(1, 13))

        data = np.ma.masked_invalid(heat.values)
        nan_mask = np.isnan(heat.values)

        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        # Zaznacz miejsca z NaN krzyżykiem
        if nan_mask.any():
            ny, nx = np.where(nan_mask)
            ax.scatter(
                nx, ny,
                marker="x",
                color="black",
                s=40,
                linewidths=0.8,
            )

        ax.set_xticks(np.arange(0, 12))
        ax.set_xticklabels(range(1, 13))
        ax.set_yticks(np.arange(0, len(years)))
        ax.set_yticklabels(heat.index)
        ax.set_xlabel("Miesiąc")
        ax.set_ylabel("Rok")
        ax.set_title(f"Średnie miesięczne – {city}")

    # Skale kolorów: po jednej na każdy wiersz
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for cax in row_cax:
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("PM2.5 [mcg/m^3]")

    fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.10)

    # Legendy poniżej kolorbarów na poziomie etykiet osi X
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi

    for row, axes in enumerate(row_axes):
        if not axes:
            continue

        # Pozycja osi X (ticki i etykieta) w pikselach
        y0_candidates = []
        label_boxes = []
        for ax in axes:
            for tick in ax.xaxis.get_ticklabels():
                if tick.get_text():
                    box = tick.get_window_extent(renderer)
                    y0_candidates.append(box.y0)
                    label_boxes.append(box)
            if ax.xaxis.label.get_text():
                box = ax.xaxis.label.get_window_extent(renderer)
                y0_candidates.append(box.y0)
                label_boxes.append(box)

        if y0_candidates:
            row_y0_px = min(y0_candidates)
        else:
            row_y0_px = min(ax.get_window_extent(renderer).y0 for ax in axes)

        # Wysokość legendy w jednostkach figury
        if y0_candidates:
            label_h_px = max(box.height for box in label_boxes)
            legend_h = max(legend_height, (label_h_px * 1.6) / fig_h_px)
        else:
            legend_h = legend_height

        _, row_y0 = fig.transFigure.inverted().transform((0, row_y0_px))
        if y0_candidates:
            label_center = row_y0 + (label_h_px / 2) / fig_h_px
            legend_y0 = label_center - legend_h / 2
        else:
            legend_y0 = row_y0 - legend_h * 0.1

        cax_pos = row_cax[row].get_position()
        lax = fig.add_axes([cax_pos.x0, legend_y0, cax_pos.width, legend_h])
        lax.axis("off")
        lax.legend(
            handles=[nan_patch],
            loc="center",
            fontsize="small",
            frameon=True,
        )

    plt.show()


def plot_extreme_stations_days_over(
    days_over_df,
    year_ref=2024,
    years=(2014, 2019, 2024),
    n=3,
    station_city_map=None,
):
    """
    Rysuje wykres słupkowy dla n najlepszych i n najgorszych stacji (dni > próg).

    Parametry:
    - days_over_df: ramka danych w układzie stacja x rok
    - year_ref: rok referencyjny do wyboru stacji skrajnych
    - years: lata do pokazania na wykresie (kolory słupków)
    - n: liczba najlepszych i najgorszych stacji z roku referencyjnego
    - station_city_map: mapowanie {kod stacji -> miasto} lub Series/DataFrame
      z kolumnami "Kod stacji" i "Miejscowość"
    """

    df = days_over_df.copy()

    # Walidacja lat
    if year_ref not in df.columns:
        raise ValueError(f"Rok bazowy {year_ref} nieobecny w days_over_df.columns")
    for y in years:
        if y not in df.columns:
            raise ValueError(f"Rok {y} nieobecny w days_over_df.columns")

    # Przygotuj mapowanie stacja -> miasto
    if station_city_map is None:
        raise ValueError("Wymagane jest station_city_map (kod stacji -> miasto).")
    if isinstance(station_city_map, pd.DataFrame):
        if not {"Kod stacji", "Miejscowość"}.issubset(station_city_map.columns):
            raise ValueError(
                "station_city_map jako DataFrame musi mieć kolumny "
                "'Kod stacji' i 'Miejscowość'."
            )
        mapping = (
            station_city_map.assign(
                **{
                    "Kod stacji": station_city_map["Kod stacji"].astype(str).str.strip(),
                    "Miejscowość": station_city_map["Miejscowość"].astype(str),
                }
            )
            .drop_duplicates("Kod stacji")
            .set_index("Kod stacji")["Miejscowość"]
            .to_dict()
        )
    elif isinstance(station_city_map, pd.Series):
        ser = station_city_map.copy()
        ser.index = ser.index.astype(str).str.strip()
        mapping = ser.to_dict()
    else:
        mapping = {str(k).strip(): v for k, v in dict(station_city_map).items()}

    # n najlepszych i n najgorszych stacji w year_ref
    s_ref = df[year_ref].dropna()
    lowest = s_ref.nsmallest(n)
    highest = s_ref.nlargest(n)

    stations_sel = pd.Index(lowest.index.tolist() + highest.index.tolist()).unique()
    df_sel = df.loc[stations_sel, years]

    # Etykiety stacji z miastem
    missing_cities = []
    for s in stations_sel:
        if s not in mapping:
            missing_cities.append(s)
            continue
        city_val = mapping[s]
        if pd.isna(city_val) or str(city_val).strip() == "" or str(city_val).strip().lower() == "nan":
            missing_cities.append(s)
    if missing_cities:
        raise ValueError(
            "Brak miasta dla stacji: " + ", ".join(missing_cities)
        )
    station_labels = [f"{s}\n({str(mapping[s]).strip()})" for s in stations_sel]

    # Wykres słupkowy grupowany
    x = np.arange(len(stations_sel))
    n_years = len(years)
    width = 0.8 / n_years

    plt.figure(figsize=(12, 6))
    for i, y in enumerate(years):
        x_shift = x + (i - (n_years - 1) / 2) * width
        plt.bar(x_shift, df_sel[y].values, width=width, label=str(y))

    # Etykiety poziome z nazwą stacji i miastem
    plt.xticks(x, station_labels, rotation=0, ha="center")
    plt.xlabel("Stacja (miasto)")
    plt.ylabel("Liczba dni powyżej normy dobowej")
    plt.title(
        f"Liczba dni powyżej normy – najlepsze / najgorsze stacje z roku {year_ref}"
    )
    plt.legend(title="Rok")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
