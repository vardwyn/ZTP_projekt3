import pandas as pd


def analyze_raw_df(df, meta_keys=None):
    """
    Analizuje surowy DataFrame z danymi GIOŚ.

    Zwraca słownik z podstawowymi statystykami:
    - kształt danych,
    - liczba wierszy metadanych i pomiarów,
    - unikalność wybranych pól metadanych,
    - unikalne wartości pola "Czas uśredniania" i "Wskaźnik",
    - top 10 stacji z największym udziałem braków danych.
    """

    meta_keys = meta_keys or []
    labels = df.iloc[:, 0]

    # Pomocniczo pobieramy pojedynczy wiersz metadanych (jeśli istnieje)
    def get_meta_row(name):
        rows = df.loc[labels == name]
        if rows.empty:
            return None
        return rows.iloc[0, 1:]

    meta_rows = {
        "Nr": get_meta_row("Nr"),
        "Kod stacji": get_meta_row("Kod stacji"),
        "Kod stanowiska": get_meta_row("Kod stanowiska"),
        "Czas uśredniania": get_meta_row("Czas uśredniania"),
        "Wskaźnik": get_meta_row("Wskaźnik"),
    }

    # Informacja o unikalności wybranych metadanych
    def uniq_info(s):
        if s is None:
            return {"all_unique": None, "duplicates": None}
        dup = s[s.duplicated(keep=False)]
        return {
            "all_unique": bool(s.is_unique),
            "duplicates": dup if not dup.empty else None,
        }

    uniqueness = {
        "Nr": uniq_info(meta_rows["Nr"]),
        "Kod stacji": uniq_info(meta_rows["Kod stacji"]),
        "Kod stanowiska": uniq_info(meta_rows["Kod stanowiska"]),
    }

    # Unikalne wartości w polach opisowych (jeśli istnieją)
    def uniques_list(s):
        if s is None:
            return None
        return pd.Series(s.dropna().unique()).tolist()

    unique_values = {
        "Czas uśredniania": uniques_list(meta_rows["Czas uśredniania"]),
        "Wskaźnik": uniques_list(meta_rows["Wskaźnik"]),
    }

    # Wydzielamy tylko wiersze z pomiarami czasowymi
    timestamp_mask = ~labels.isin(meta_keys)
    meas = df.loc[timestamp_mask, 1:]

    # Procent braków danych per kolumna-stacja
    nan_per_station = meas.isna().mean() * 100
    top10_nan = nan_per_station.sort_values(ascending=False).head(10)

    # Mapujemy numer kolumny na kod stacji (jeśli jest dostępny)
    if meta_rows["Kod stacji"] is not None:
        station_codes = meta_rows["Kod stacji"].reindex(meas.columns)
    else:
        station_codes = pd.Series(index=meas.columns, dtype=object)

    top10_nan_stations = pd.DataFrame(
        {
            "station_column": top10_nan.index,
            "station_code": station_codes.reindex(top10_nan.index).values,
            "nan_percent": top10_nan.values,
        }
    ).reset_index(drop=True)

    num_timestamps = int(timestamp_mask.sum())
    num_metadata = int((~timestamp_mask).sum())
    num_stations = df.shape[1] - 1

    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "timestamp_counts": {
            "timestamp_rows": num_timestamps,
            "metadata_rows": num_metadata,
            "stations": num_stations,
        },
        "uniqueness": uniqueness,
        "unique_values": unique_values,
        "nan_by_station_top10": top10_nan_stations,
    }


def check_timestamps(df, meta_keys=None, expected_freq=None):
    """
    Sprawdza poprawność i ciągłość znaczników czasu w surowych danych.

    Zwraca m.in.:
    - liczbę poprawnych i niepoprawnych timestampów,
    - duplikaty,
    - dominującą różnicę czasową,
    - listę brakujących i nadmiarowych timestampów.
    """

    meta_keys = meta_keys or []
    labels = df.iloc[:, 0]

    # Wiersze niebędące metadanymi traktujemy jako znaczniki czasu
    ts_mask = ~labels.isin(meta_keys)
    ts_raw = labels[ts_mask]

    ts_parsed = pd.to_datetime(ts_raw, errors="coerce")
    invalid_ts = ts_raw[ts_parsed.isna()]

    ts = ts_parsed.dropna().sort_values()
    dup = ts[ts.duplicated(keep=False)]

    deltas = ts.diff().dropna()

    # Główna różnica czasu (dominująca lub oczekiwana)
    if expected_freq is None:
        main_delta = deltas.value_counts().idxmax() if not deltas.empty else None
    else:
        main_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(expected_freq))

    # Nieregularne odstępy czasu
    irregular_intervals = None
    if main_delta is not None and not deltas.empty:
        bad = deltas[deltas != main_delta]
        if not bad.empty:
            prev_ts = ts.shift(1).loc[bad.index]
            irregular_intervals = pd.DataFrame(
                {
                    "prev_timestamp": prev_ts.values,
                    "current_timestamp": ts.loc[bad.index].values,
                    "actual_delta": bad.values,
                    "expected_delta": main_delta,
                }
            )
            if main_delta > pd.Timedelta(0):
                est_missing = (bad / main_delta).astype("int64") - 1
                irregular_intervals["estimated_missing_between"] = est_missing.clip(lower=0).values
            else:
                irregular_intervals["estimated_missing_between"] = None

    # Braki i nadmiarowe timestampy względem pełnego zakresu
    missing_ts = extra_ts = None
    if main_delta is not None and len(ts) > 1:
        expected_range = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=main_delta)
        missing_ts = expected_range.difference(ts)
        extra_ts = ts[~ts.isin(expected_range)]

    summary = {
        "n_timestamp_rows": int(ts_mask.sum()),
        "n_valid_timestamps": int(len(ts)),
        "n_invalid_timestamps": int(len(invalid_ts)),
        "all_unique": bool(ts.is_unique),
        "n_duplicates": int(len(dup)),
        "main_delta": main_delta,
        "n_unique_deltas": int(deltas.nunique()) if not deltas.empty else 0,
        "expected_total_points_if_full_range": (
            int(((ts.iloc[-1] - ts.iloc[0]) / main_delta) + 1)
            if main_delta not in (None, pd.Timedelta(0)) and len(ts) > 1
            else None
        ),
    }

    return {
        "summary": summary,
        "invalid_timestamps": invalid_ts,
        "duplicates": dup,
        "deltas_counts": deltas.value_counts(),
        "irregular_intervals": irregular_intervals,
        "missing_timestamps": missing_ts,
        "extra_timestamps": extra_ts,
    }


def report_nan_runs(meas_df: pd.DataFrame, top_k: int = 3):
    """
    Raportuje najdłuższe ciągi braków danych (NaN) dla każdej stacji.

    Zwraca słownik z:
    - liczbą wszystkich stacji,
    - liczbą stacji zawierających choć jeden NaN,
    - mapą: kod_stacji -> DataFrame z top_k najdłuższymi lukami (w dniach).
    """

    df = meas_df.copy()
    total_stations = df.shape[1]

    nan_runs = {}
    stations_with_any_nan = 0

    for col in df.columns:
        s = df[col]
        mask = s.isna()

        if not mask.any():
            continue

        stations_with_any_nan += 1

        # Grupujemy kolejne bloki True/False; interesują nas tylko bloki NaN
        block_id = mask.ne(mask.shift()).cumsum()
        nan_blocks = s[mask].groupby(block_id[mask])

        segments = []
        for _, grp in nan_blocks:
            idx = grp.index
            segments.append(
                {
                    "start": idx[0],
                    "end": idx[-1],
                    "length": len(idx) / 24,  # długość w dniach (dla danych godzinowych)
                }
            )

        if segments:
            seg_df = (
                pd.DataFrame(segments)
                .sort_values("length", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )
            nan_runs[col] = seg_df

    return {
        "total_stations": total_stations,
        "stations_with_any_nan": stations_with_any_nan,
        "nan_runs": nan_runs,
    }

def shift_midnight_to_previous_day(index: pd.Index) -> pd.DatetimeIndex:
    """
    Przesuwa dokładnie północ (00:00:00) na koniec poprzedniego dnia.

    Zwraca nowy indeks, zachowując kolejność elementów (bez przestawiania).
    """

    ts = pd.to_datetime(index)
    midnight_mask = ts == ts.normalize()
    # Odejmujemy 1 ns, aby północ "należała" do poprzedniego dnia,
    # ale pozostała chronologicznie po ostatnich pomiarach dnia.
    return ts.where(~midnight_mask, ts - pd.Timedelta(nanoseconds=1))

def monthly_avg_with_nan_threshold(
    data: pd.DataFrame,
    max_nan_per_month: int,
) -> pd.DataFrame:
    """
    Liczy średnie miesięczne i maskuje miesiące z nadmiarem braków danych.

    - Midnight (00:00) przypisywany jest do poprzedniego dnia.
    - Jeśli liczba NaN w miesiącu przekracza próg, wynik jest maskowany.
    """

    work_df = data.copy()
    work_df.index = shift_midnight_to_previous_day(work_df.index)

    # Średnia miesięczna i liczba braków
    monthly_mean = work_df.resample("ME").mean()
    monthly_nan_count = work_df.isna().resample("ME").sum()

    # Maskowanie miesięcy zbyt "dziurawych" (NaN count > threshold)
    monthly_mean = monthly_mean.mask(monthly_nan_count > max_nan_per_month)
    return monthly_mean


def average_by_city(
    measurements_df: pd.DataFrame,
    metadata_combined: pd.DataFrame,
    code_col: str = "Kod stacji",
    city_col: str = "Miejscowość",
) -> pd.DataFrame:
    """
    Agreguje pomiary do poziomu miasta (średnia po stacjach w tym samym mieście).

    Sprawdza, czy wszystkie stacje mają przypisane miasto w metadanych.
    """

    df = measurements_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    meta = metadata_combined.copy()
    meta[code_col] = meta[code_col].astype(str).str.strip()

    # Mapowanie stacja -> miasto
    meta_valid = meta.loc[meta[city_col].notna(), [code_col, city_col]]
    station_to_city = meta_valid.drop_duplicates(subset=code_col).set_index(code_col)[city_col]

    # Walidacja braków w metadanych
    missing_in_meta = sorted(set(df.columns) - set(meta[code_col]))
    if missing_in_meta:
        raise AssertionError(
            "Stations present in data but missing in metadata_combined: "
            + ", ".join(missing_in_meta)
        )

    missing_city = sorted(set(df.columns) - set(station_to_city.index))
    if missing_city:
        raise AssertionError(
            "Stations present in data but without city (NaN) "
            f"in metadata_combined[{city_col}]: "
            + ", ".join(missing_city)
        )

    # Grupowanie po mieście (kolumny)
    col_cities = df.columns.map(station_to_city)
    df_city = df.groupby(axis=1, by=col_cities).mean()

    return df_city


def count_days_over_threshold(meas_df: pd.DataFrame,
                              threshold: float,
                              years=(2014, 2019, 2024)) -> pd.DataFrame:
    """
    Zlicza liczbę dni w roku, w których średnia dobowa przekracza próg.

    - Midnight (00:00) przypisywany jest do poprzedniego dnia.
    - Wynik zwracany jest w układzie: stacja x rok.
    """

    df = meas_df.copy()
    df.index = shift_midnight_to_previous_day(df.index).normalize()

    # Średnia dobowa i przekroczenia progu
    daily_mean = df.groupby(df.index).mean()
    daily_over = daily_mean.gt(threshold)

    # Zliczanie przekroczeń w podziale na lata
    days_per_year = daily_over.groupby(daily_over.index.year).sum()

    result = days_per_year.reindex(index=years, fill_value=0).T
    result.index.name = "station"
    result.columns.name = "year"
    return result
