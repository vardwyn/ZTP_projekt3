import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Patch


# ============================================================
# Funkcje pomocnicze – wspólne dla wielu wykresów
# ============================================================

def _group_by_year_with_data(df: pd.DataFrame, years_order=None):
    """
    Grupuje dane po roku i zwraca przygotowane grupy.

    - słownik {rok -> ramka danych}
    - listę lat z co najmniej jedną nie-NaN wartością
    """
    groups = {year: df_y for year, df_y in df.groupby(df.index.year)}
    candidate_years = years_order or sorted(groups.keys())
    years = [
        year for year in candidate_years
        if year in groups and groups[year].notna().any().any()
    ]
    return groups, years


def _build_panel_axes(
    n_panels,
    ncols,
    panel_size,
    sharex,
    sharey,
    hspace,
    wspace,
):
    """
    Tworzy figurę i spłaszczoną listę osi dla paneli.

    Zwraca: (fig, axes_list, nrows, ncols).
    """
    ncols = max(1, int(ncols))
    nrows = int((n_panels + ncols - 1) / ncols)

    fig_w = panel_size[0] * ncols
    fig_h = panel_size[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex,
        sharey=sharey,
    )
    axes_list = np.atleast_1d(axes).ravel().tolist()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return fig, axes_list, nrows, ncols


def plot_monthly_avg_station_per_year(
    monthly_avg,
    main_title=None,
    main_title_fontsize=None,
    main_title_y=0.98,
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
    legend_enabled=True,
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
    - legend_enabled: czy pokazać legendę (po prawej) czy nie
    - legend_ncol: liczba kolumn w legendzie
    - right_margin: prawy margines (miejsce na legendę po prawej)
    - sharex / sharey: współdzielenie osi między panelami
    """

    # ------------------------------------------------------------------
    # 1) Walidacja i przygotowanie danych
    # ------------------------------------------------------------------
    if monthly_avg.empty:
        return

    assert isinstance(monthly_avg.index, pd.DatetimeIndex), (
        "Indeks musi być typu DatetimeIndex."
    )
    df = monthly_avg
    groups, years = _group_by_year_with_data(df, years_order)
    if not years:
        return

    # ------------------------------------------------------------------
    # 2) Budowa układu paneli i ustawienia figury
    # ------------------------------------------------------------------
    fig, axes, _, _ = _build_panel_axes(
        n_panels=len(years),
        ncols=ncols,
        panel_size=panel_size,
        sharex=sharex,
        sharey=sharey,
        hspace=hspace,
        wspace=wspace,
    )
    if main_title:
        fig.suptitle(
            main_title,
            fontsize=main_title_fontsize or title_fontsize,
            x=0.5,
            y=main_title_y,
            ha="center",
        )

    # ------------------------------------------------------------------
    # 3) Rysowanie linii oraz zbieranie uchwytów do legendy
    # ------------------------------------------------------------------
    legend_handles = {}

    for ax, year in zip(axes, years):
        df_y = groups[year]
        months = df_y.index.month

        # Linie dla każdej stacji w danym roku
        for station, series in df_y.items():
            line = ax.plot(
                months,
                series,
                marker="o",
                markersize=marker_size,
                label=str(station),
            )[0]
            # Kolejność legendy opiera się o pierwsze wystąpienie stacji;
            legend_handles[str(station)] = line

        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Miesiąc", fontsize=label_fontsize)
        ax.set_ylabel("PM2.5 [mcg/m^3]", fontsize=label_fontsize)
        ax.set_title(f"Rok {year}", fontsize=title_fontsize)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ------------------------------------------------------------------
    # 4) Ukrycie pustych paneli + legenda zbiorcza (jeśli wybrana)
    # ------------------------------------------------------------------
    for ax in axes[len(years):]:
        ax.set_visible(False)

    if legend_enabled and legend_handles:
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

    plt.show()


def plot_monthly_avg_station_mean_std_per_year(
    monthly_avg,
    main_title=None,
    main_title_fontsize=None,
    main_title_y=0.98,
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
    legend_enabled=True,
    legend_ncol=1,
    right_margin=0.99,
    sharey=True,
    sharex=False,
):
    """
    Rysuje miesięczne średnie i odchylenia standardowe (po stacjach) dla każdego roku jako wykres z wieloma panelami.

    Parametry:
    - main_title: tytuł całego wykresu (None = brak tytułu)
    - main_title_fontsize: rozmiar tytułu głównego (domyślnie jak title_fontsize)
    - main_title_y: pozycja tytułu głównego na osi Y (0..1)
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
    - legend_enabled: czy pokazać legendę (po prawej) czy nie
    - legend_ncol: liczba kolumn w legendzie
    - right_margin: prawy margines (miejsce na legendę po prawej)
    - sharex / sharey: współdzielenie osi między panelami
    """

    # ------------------------------------------------------------------
    # 1) Walidacja i przygotowanie danych
    # ------------------------------------------------------------------
    if monthly_avg.empty:
        return

    assert isinstance(monthly_avg.index, pd.DatetimeIndex), (
        "Indeks musi być typu DatetimeIndex."
    )
    df = monthly_avg
    groups, years = _group_by_year_with_data(df, years_order)
    if not years:
        return

    # ------------------------------------------------------------------
    # 2) Budowa układu paneli i ustawienia figury
    # ------------------------------------------------------------------
    fig, axes, _, _ = _build_panel_axes(
        n_panels=len(years),
        ncols=ncols,
        panel_size=panel_size,
        sharex=sharex,
        sharey=sharey,
        hspace=hspace,
        wspace=wspace,
    )
    if main_title:
        fig.suptitle(
            main_title,
            fontsize=main_title_fontsize or title_fontsize,
            x=0.5,
            y=main_title_y,
            ha="center",
        )

    # ------------------------------------------------------------------
    # 3) Rysowanie średniej i odchylenia standardowego
    # ------------------------------------------------------------------
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

        # Uchwyt do legendy – kolory i alfa jak na wykresie
        legend_handles[mean_label] = line
        legend_handles[std_label] = Patch(
            facecolor=fill.get_facecolor()[0],
            alpha=0.3,
            label=std_label,
        )

        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Miesiąc", fontsize=label_fontsize)
        ax.set_ylabel("PM2.5 [mcg/m^3]", fontsize=label_fontsize)
        ax.set_title(f"Rok {year}", fontsize=title_fontsize)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ------------------------------------------------------------------
    # 4) Ukrycie pustych paneli + legenda zbiorcza (jeśli wybrana)
    # ------------------------------------------------------------------
    for ax in axes[len(years):]:
        ax.set_visible(False)

    if legend_enabled and legend_handles:
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

    # ------------------------------------------------------------------
    # 1) Walidacja i przygotowanie danych
    # ------------------------------------------------------------------
    if monthly_avg.empty:
        return

    assert isinstance(monthly_avg.index, pd.DatetimeIndex), (
        "Indeks musi być typu DatetimeIndex."
    )
    df = monthly_avg
    groups, years = _group_by_year_with_data(df, years_order)
    if not years:
        return

    # ------------------------------------------------------------------
    # 2) Rysowanie porównań na wspólnej osi
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    color_cycle = plt.get_cmap("Set1")(np.linspace(0, 1, len(years)))

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
        color = color_cycle[idx]

        ax.plot(
            months,
            monthly_mean.values,
            marker="o",
            markersize=marker_size,
            label=str(year),
            color=color,
        )
        ax.fill_between(
            months,
            (monthly_mean - monthly_std).values,
            (monthly_mean + monthly_std).values,
            alpha=0.2,
            color=color,
        )

    # ------------------------------------------------------------------
    # 3) Opis osi, siatka i legenda
    # ------------------------------------------------------------------
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Miesiąc", fontsize=label_fontsize)
    ax.set_ylabel("PM2.5 (średnia stacji) [mcg/m^3]", fontsize=label_fontsize)
    ax.set_title(
        "Miesięczne średnie (połączone stacje) – porównanie lat",
        fontsize=title_fontsize,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    leg = ax.legend(title=legend_title, fontsize=legend_fontsize, frameon=legend_frame)
    if legend_title_fontsize:
        leg.get_title().set_fontsize(legend_title_fontsize)

    fig.tight_layout()
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
    legend_frame=True,
):
    """
    Rysuje średnie miesięczne dla wybranych miast i lat na jednym wykresie.

    - Kolor rozróżnia miasto (hue).
    - Rok rozróżniany jest przez styl linii/marker (domyślny styl seaborn).
    - Dozwolone: 1–4 miasta i 1–3 lata (dla czytelności).

    Parametry legendy:
    - legend_fontsize / legend_frame: styl legendy
    """

    # ------------------------------------------------------------------
    # 1) Walidacja i przygotowanie danych
    # ------------------------------------------------------------------
    assert isinstance(monthly_avg_city.index, pd.DatetimeIndex), (
        "Indeks musi być typu DatetimeIndex."
    )
    df = monthly_avg_city

    if not (1 <= len(cities) <= 4):
        raise ValueError("Dozwolone są 1–4 miasta.")
    if not (1 <= len(years) <= 3):
        raise ValueError("Dozwolone są 1–3 lata.")

    assert set(cities).issubset(set(df.columns)), (
        "Brakujące miasta w monthly_avg_city columns."
    )

    # ------------------------------------------------------------------
    # 2) Budowa tabeli pod seaborn
    # ------------------------------------------------------------------
    df_long = (
        df[cities]
        .assign(rok=df.index.year, miesiąc=df.index.month)
        .melt(id_vars=["rok", "miesiąc"], var_name="miasto", value_name="pm25")
    )
    df_long = df_long[df_long["rok"].isin(years)]

    if df_long["pm25"].notna().sum() == 0:
        return

    # ------------------------------------------------------------------
    # 3) Wykres: hue=miasto, style=rok (seaborn)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df_long,
        x="miesiąc",
        y="pm25",
        hue="miasto",
        style="rok",
        hue_order=cities,
        style_order=years,
        legend="full",
        linewidth=line_width,
        markersize=marker_size,
        ax=ax,
    )

    # ------------------------------------------------------------------
    # 4) Opis osi, siatka i legenda
    # ------------------------------------------------------------------
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Miesiąc", fontsize=label_fontsize)
    ax.set_ylabel("PM2.5 [mcg/m^3]", fontsize=label_fontsize)
    ax.set_title("Średnie miesięczne dla miasta i roku", fontsize=title_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legenda (2 kolumny)
    handles, labels = ax.get_legend_handles_labels()
    default_legend = ax.get_legend()
    if default_legend:
        default_legend.remove()

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=legend_fontsize,
        frameon=legend_frame,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

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
    Rysuje heatmapy średnich miesięcznych dla miast w układzie paneli.

    - Skala kolorów per wiersz
    - Układ może mieć 1 lub 2 kolumny paneli (parametr ncols).

    Parametry:
    - ncols: liczba kolumn paneli (1 lub 2)
    - figsize: rozmiar całej figury (domyślnie wyliczany)
    - side_width: względna szerokość kolumny po prawej (względem mapy)
    - legend_height: wysokość legendy w jednostkach figury (0..1)
    """

    # ------------------------------------------------------------------
    # 1) Walidacja wejścia i przygotowanie danych
    # ------------------------------------------------------------------
    if ncols not in (1, 2):
        raise ValueError("ncols może być równe 1 albo 2.")

    assert isinstance(monthly_avg_city.index, pd.DatetimeIndex), (
        "Indeks musi być typu DatetimeIndex."
    )
    assert monthly_avg_city.columns.is_unique, "Kolumny muszą być unikalne."
    assert set(cities).issubset(set(monthly_avg_city.columns)), (
        "Wszystkie miasta muszą być obecne w kolumnach."
    )
    years = sorted(set(years))

    df = monthly_avg_city[cities]
    df = df[df.index.year.isin(years)]

    if df.isna().all().all():
        raise ValueError("Wszytkie średnie są NaN!")

    # ------------------------------------------------------------------
    # 2) Skala kolorów wspólna dla wszystkich paneli
    # ------------------------------------------------------------------
    vmin = np.nanmin(df.values)
    vmax = np.nanmax(df.values)

    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad(color="lightgrey")  # NaN

    # ------------------------------------------------------------------
    # 3) Układ figury (siatka paneli + kolumna na colorbary)
    # ------------------------------------------------------------------
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

    # Element legendy dla braków danych
    nan_patch = Patch(
        facecolor="lightgrey",
        edgecolor="black",
        label="Brak danych (NaN)",
    )

    # Osie dla map cieplnych oraz colorbary (po jednym na wiersz)
    row_axes = [[] for _ in range(nrows)]
    row_cax = [fig.add_subplot(main_gs[row, ncols]) for row in range(nrows)]

    # ------------------------------------------------------------------
    # 4) Rysowanie map cieplnych
    # ------------------------------------------------------------------
    for idx, city in enumerate(cities):
        row = idx // ncols
        col = idx % ncols

        ax = fig.add_subplot(main_gs[row, col])
        row_axes[row].append(ax)

        s = df[city]
        heat = s.groupby([s.index.year, s.index.month]).mean().unstack()
        heat = heat.reindex(index=years, columns=range(1, 13))
        data = np.ma.masked_invalid(heat.values)
        nan_mask = np.isnan(heat.values)

        ax.imshow(
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

    # ------------------------------------------------------------------
    # 5) Colorbary – po jednym na każdy wiersz (wspólna skala)
    # ------------------------------------------------------------------
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for cax in row_cax:
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("PM2.5 [mcg/m^3]")

    fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.10)

    # ------------------------------------------------------------------
    # 6) Legendy pod colorbarami
    # ------------------------------------------------------------------
    for row, axes in enumerate(row_axes):
        if not axes:
            continue
        row_bottom = min(ax.get_position().y0 for ax in axes)
        legend_y0 = max(0.0, row_bottom - legend_height + 0.015)
        cax_pos = row_cax[row].get_position()
        lax = fig.add_axes([cax_pos.x0, legend_y0, cax_pos.width, legend_height])
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
    station_metadata=None,
):
    """
    Rysuje wykres słupkowy dla n najlepszych i n najgorszych stacji (dni > próg).

    Parametry:
    - days_over_df: ramka danych w układzie stacja x rok
    - year_ref: rok referencyjny do wyboru stacji skrajnych
    - years: lata do pokazania na wykresie (kolory słupków)
    - n: liczba najlepszych i najgorszych stacji z roku referencyjnego
    - station_metadata: DataFrame z kolumnami "Kod stacji" i "Miejscowość"
      (np. metadata_combined) lub mapowanie/Series kod->miasto
    """

    # ------------------------------------------------------------------
    # 1) Walidacja wejścia
    # ------------------------------------------------------------------
    df = days_over_df

    if year_ref not in df.columns:
        raise ValueError(f"Rok bazowy {year_ref} nieobecny w days_over_df.columns")
    for y in years:
        if y not in df.columns:
            raise ValueError(f"Rok {y} nieobecny w days_over_df.columns")

    # ------------------------------------------------------------------
    # 2) Mapowanie stacji na miasta
    # ------------------------------------------------------------------
    assert isinstance(station_metadata, pd.DataFrame), (
        "station_metadata musi być DataFrame z kolumnami 'Kod stacji' i 'Miejscowość'."
    )
    assert {"Kod stacji", "Miejscowość"}.issubset(station_metadata.columns), (
        "station_metadata musi zawierać kolumny 'Kod stacji' i 'Miejscowość'."
    )
    mapping = (
        station_metadata.assign(
            **{
                "Kod stacji": station_metadata["Kod stacji"].astype(str).str.strip(),
                "Miejscowość": station_metadata["Miejscowość"].astype(str),
            }
        )
        .drop_duplicates("Kod stacji")
        .set_index("Kod stacji")["Miejscowość"]
        .to_dict()
    )

    # ------------------------------------------------------------------
    # 3) Wybór stacji skrajnych (n najlepszych i n najgorszych)
    # ------------------------------------------------------------------
    s_ref = df[year_ref].dropna()
    lowest = s_ref.nsmallest(n)
    highest = s_ref.nlargest(n)
    stations_sel = pd.Index(lowest.index.tolist() + highest.index.tolist()).unique()
    df_sel = df.loc[stations_sel, years]

    # Sprawdzenie, czy dla każdej stacji mamy miasto
    missing_cities = [
        s for s in stations_sel
        if (
            s not in mapping
            or pd.isna(mapping[s])
            or str(mapping[s]).strip() == ""
            or str(mapping[s]).strip().lower() == "nan"
        )
    ]
    if missing_cities:
        raise ValueError("Brak miasta dla stacji: " + ", ".join(missing_cities))

    # Stacja \n (miasto)
    station_labels = [
        f"{s}\n({str(mapping[s]).strip()})"
        for s in stations_sel
    ]

    # ------------------------------------------------------------------
    # 4) Rysowanie wykresu słupkowego grupowanego
    # ------------------------------------------------------------------
    x = np.arange(len(stations_sel))
    n_years = len(years)
    width = 0.8 / n_years

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, y in enumerate(years):
        x_shift = x + (i - (n_years - 1) / 2) * width
        ax.bar(x_shift, df_sel[y].values, width=width, label=str(y))

    ax.set_xticks(x)
    ax.set_xticklabels(station_labels, rotation=0, ha="center")
    ax.set_xlabel("Stacja (miasto)")
    ax.set_ylabel("Liczba dni powyżej normy dobowej")
    ax.set_title(
        f"Liczba dni powyżej normy – najlepsze / najgorsze stacje z roku {year_ref}"
    )
    ax.legend(title="Rok")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()

#### MODIFIED

def plot_pm25_days_over_by_voivodeship_years(
    meas_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    years: list[int],
    threshold: float = 15.0,
):
    """
    Liczba dni z przekroczeniem normy PM2.5
    """

    # ------------------------------------------------------------
    # 1. Przygotowanie dobowych średnich
    # ------------------------------------------------------------
    df = meas_df.copy()
    df.index = df.index - pd.Timedelta(hours=1)
    df.index = df.index.normalize()

    daily_mean = df.groupby(df.index).mean()
    daily_over = daily_mean.gt(threshold)

    # ------------------------------------------------------------
    # 2. Mapowanie stacja → województwo
    # ------------------------------------------------------------
    meta = metadata_df.copy()
    meta["Kod stacji"] = meta["Kod stacji"].astype(str).str.strip()

    station_to_voiv = (
        meta.drop_duplicates("Kod stacji")
            .set_index("Kod stacji")["Województwo"]
    )

    missing = sorted(set(daily_over.columns) - set(station_to_voiv.index))
    if missing:
        raise AssertionError(
            "Brak województwa dla stacji: " + ", ".join(missing)
        )

    daily_over.columns = daily_over.columns.map(station_to_voiv)

    # ------------------------------------------------------------
    # 3. AGREGACJA : województwo × dzień (any)
    # ------------------------------------------------------------
    voiv_daily_over = daily_over.groupby(axis=1, level=0).any()

    # ------------------------------------------------------------
    # 4. Zliczanie dni w latach
    # ------------------------------------------------------------
    result = {}
    for year in years:
        mask = voiv_daily_over.index.year == year
        result[year] = voiv_daily_over.loc[mask].sum()

    days_over_voiv = pd.DataFrame(result).sort_index()

    # ------------------------------------------------------------
    # 5. Wykres
    # ------------------------------------------------------------
    x = np.arange(len(days_over_voiv.index))
    width = 0.8 / len(years)

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, year in enumerate(years):
        shift = x + (i - (len(years) - 1) / 2) * width
        ax.bar(shift, days_over_voiv[year], width=width, label=str(year))

    ax.set_xticks(x)
    ax.set_xticklabels(days_over_voiv.index, rotation=45, ha="right")
    ax.set_xlabel("Województwo")
    ax.set_ylabel("Liczba dni z przekroczeniem normy PM2.5")
    ax.set_title("Liczba dni z przekroczeniem normy PM2.5 – województwa")

    ax.legend(title="Rok")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()

    return days_over_voiv