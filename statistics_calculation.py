import pandas as pd


def analyze_raw_df(df, meta_keys=None):

    labels = df.iloc[:, 0]

    def get_meta_row(name):
        rows = df.loc[labels == name]
        if rows.empty:
            return None
        return rows.iloc[0, 1:]

    nr_row          = get_meta_row("Nr")
    kod_stacji_row  = get_meta_row("Kod stacji")
    kod_stan_row    = get_meta_row("Kod stanowiska")
    czas_row        = get_meta_row("Czas uśredniania")
    wskaznik_row    = get_meta_row("Wskaźnik")

    def uniq_info(s):
        if s is None:
            return {"all_unique": None, "duplicates": None}
        dup = s[s.duplicated(keep=False)]
        return {
            "all_unique": bool(s.is_unique),
            "duplicates": dup if not dup.empty else None,
        }

    nr_info         = uniq_info(nr_row)
    kod_stacji_info = uniq_info(kod_stacji_row)
    kod_stan_info   = uniq_info(kod_stan_row)

    def uniques_list(s):
        if s is None:
            return None
        return pd.Series(s.dropna().unique()).tolist()

    czas_unique     = uniques_list(czas_row)
    wskaznik_unique = uniques_list(wskaznik_row)

    timestamp_mask = ~labels.isin(meta_keys)
    meas = df.loc[timestamp_mask, 1:]

    nan_per_station = meas.isna().mean() * 100

    if kod_stacji_row is not None:
        station_codes = kod_stacji_row.reindex(meas.columns)
    else:
        station_codes = pd.Series(index=meas.columns, dtype=object)

    top10_nan = nan_per_station.sort_values(ascending=False).head(10)
    top10_nan_stations = pd.DataFrame({
        "station_column": top10_nan.index,
        "station_code": station_codes.reindex(top10_nan.index).values,
        "nan_percent": top10_nan.values,
    }).reset_index(drop=True)

    num_timestamps = int(timestamp_mask.sum())
    num_metadata   = int((~timestamp_mask).sum())
    num_stations   = df.shape[1] - 1

    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "timestamp_counts": {
            "timestamp_rows": num_timestamps,
            "metadata_rows": num_metadata,
            "stations": num_stations,
        },
        "uniqueness": {
            "Nr": nr_info,
            "Kod stacji": kod_stacji_info,
            "Kod stanowiska": kod_stan_info,
        },
        "unique_values": {
            "Czas uśredniania": czas_unique,
            "Wskaźnik": wskaznik_unique,
        },
        "nan_by_station_top10": top10_nan_stations,
    }


def check_timestamps(df, meta_keys=None, expected_freq=None):

    labels = df.iloc[:, 0]

    ts_mask = ~labels.isin(meta_keys)
    ts_raw = labels[ts_mask]

    ts_parsed = pd.to_datetime(ts_raw, errors="coerce")

    invalid_ts = ts_raw[ts_parsed.isna()]

    ts = ts_parsed.dropna()
    ts_sorted = ts.sort_values()

    all_unique = ts_sorted.is_unique
    dup = ts_sorted[ts_sorted.duplicated(keep=False)]

    deltas = ts_sorted.diff().dropna()

    if expected_freq is None:
        if deltas.empty:
            main_delta = None
        else:
            main_delta = deltas.value_counts().idxmax()
    else:
        main_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(expected_freq))

    irregular_intervals = None
    if main_delta is not None and not deltas.empty:
        bad = deltas[deltas != main_delta]
        if not bad.empty:
            records = []
            for idx, delta in bad.items():
                pos = ts_sorted.index.get_loc(idx)
                prev_ts = ts_sorted.iloc[pos - 1]
                est_missing = int(delta / main_delta) - 1 if main_delta > pd.Timedelta(0) else None
                records.append({
                    "prev_timestamp": prev_ts,
                    "current_timestamp": ts_sorted.loc[idx],
                    "actual_delta": delta,
                    "expected_delta": main_delta,
                    "estimated_missing_between": max(est_missing, 0) if est_missing is not None else None,
                })
            irregular_intervals = pd.DataFrame(records)

    missing_ts = extra_ts = None
    if main_delta is not None and len(ts_sorted) > 1:
        expected_range = pd.date_range(ts_sorted.iloc[0], ts_sorted.iloc[-1], freq=main_delta)
        missing_ts = expected_range.difference(ts_sorted)
        extra_ts = ts_sorted[~ts_sorted.isin(expected_range)]

    summary = {
        "n_timestamp_rows": int(ts_mask.sum()),
        "n_valid_timestamps": int(len(ts)),
        "n_invalid_timestamps": int(len(invalid_ts)),
        "all_unique": bool(all_unique),
        "n_duplicates": int(len(dup)),
        "main_delta": main_delta,
        "n_unique_deltas": int(deltas.nunique()) if not deltas.empty else 0,
        "expected_total_points_if_full_range": (
            int(((ts_sorted.iloc[-1] - ts_sorted.iloc[0]) / main_delta) + 1)
            if main_delta not in (None, pd.Timedelta(0)) and len(ts_sorted) > 1
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

    df = meas_df.copy()
    total_stations = df.shape[1]

    nan_runs = {}
    stations_with_any_nan = 0

    for col in df.columns:
        s = df[col]
        m = s.isna()

        if not m.any():
            continue

        stations_with_any_nan += 1

        g = (m.ne(m.shift()).cumsum()) * m

        segments = []
        nan_groups = g[m] 

        for gid, grp in nan_groups.groupby(nan_groups):
            idx = grp.index
            start = idx[0]
            end = idx[-1]
            length = len(idx)

            segments.append(
                {"start": start, "end": end, "length": length/24}
            )

        if segments:
            seg_df = pd.DataFrame(segments).sort_values(
                "length", ascending=False
            ).head(top_k)
            nan_runs[col] = seg_df.reset_index(drop=True)

    report = {
        "total_stations": total_stations,
        "stations_with_any_nan": stations_with_any_nan,
        "nan_runs": nan_runs,
    }

    return report


def monthly_avg_with_nan_threshold(
    data: pd.DataFrame,
    max_nan_per_month: int,
) -> pd.DataFrame:

    work_df = data.copy()
    work_df.index = pd.to_datetime(work_df.index)

    # północ -- poprzedni dzień
    midnight_mask = work_df.index == work_df.index.normalize()
    effective_index = work_df.index - pd.to_timedelta(
        midnight_mask.astype(int), unit="D"
    )
    work_df.index = effective_index

    monthly_mean = work_df.resample("M").mean()

    monthly_nan_count = work_df.isna().resample("M").sum()

    bad_mask = monthly_nan_count > max_nan_per_month
    monthly_mean = monthly_mean.mask(bad_mask)

    return monthly_mean


def hourly_to_daily_30d_sma(df_hourly: pd.DataFrame,
                            window_days: int = 30,
                            min_periods: int | None = None) -> pd.DataFrame:

    if min_periods is None:
        min_periods = window_days

    df = df_hourly.copy()

    ts = pd.to_datetime(df.index)
    df.index = ts

    # Północ - poprzedni dzień
    midnight_mask = (
        (ts.hour == 0) &
        (ts.minute == 0) &
        (ts.second == 0) &
        (ts.microsecond == 0)
    )
    effective_ts = ts.where(~midnight_mask, ts - pd.Timedelta(days=1))

    df.index = effective_ts.normalize()

    daily = df.resample("D").mean()

    daily_sma = daily.rolling(window=window_days,
                              min_periods=min_periods).mean()

    return daily_sma


def average_by_city(
    measurements_df: pd.DataFrame,
    metadata_combined: pd.DataFrame,
    code_col: str = "Kod stacji",
    city_col: str = "Miejscowość",
) -> pd.DataFrame:

    df = measurements_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    meta = metadata_combined.copy()
    meta[code_col] = meta[code_col].astype(str).str.strip()

    meas_codes = set(df.columns)
    meta_codes_all = set(meta[code_col])
    meta_codes_with_city = set(meta.loc[meta[city_col].notna(), code_col])

    missing_in_meta = sorted(meas_codes - meta_codes_all)
    if missing_in_meta:
        raise AssertionError(
            "Stations present in data but missing in metadata_combined: "
            + ", ".join(missing_in_meta)
        )

    missing_city = sorted(meas_codes - meta_codes_with_city)
    if missing_city:
        raise AssertionError(
            "Stations present in data but without city (NaN) "
            f"in metadata_combined[{city_col}]: "
            + ", ".join(missing_city)
        )

    meta_valid = (
        meta.loc[meta[city_col].notna(), [code_col, city_col]]
        .drop_duplicates(subset=code_col, keep="first")
    )
    station_to_city = meta_valid.set_index(code_col)[city_col].to_dict()

    col_cities = pd.Series({code: station_to_city[code] for code in df.columns},
                           name="city")

    df_city = df.groupby(axis=1, by=col_cities).mean()

    return df_city


def count_days_over_threshold(meas_df: pd.DataFrame,
                              threshold: float,
                              years=(2014, 2019, 2024)) -> pd.DataFrame:
    df = meas_df.copy()
    ts = pd.to_datetime(df.index)
    df.index = ts

    # północ -- poprzedni dzień
    midnight_mask = (
        (ts.hour == 0) &
        (ts.minute == 0) &
        (ts.second == 0) &
        (ts.microsecond == 0)
    )
    effective_ts = ts.where(~midnight_mask, ts - pd.Timedelta(days=1))
    day_index = effective_ts.normalize()
    df.index = day_index

    # średnia dobowa > threshold
    daily_mean = df.groupby(df.index).mean()
    daily_over = daily_mean.gt(threshold)

    days_per_year = daily_over.groupby(daily_over.index.year).sum()

    result = days_per_year.reindex(index=years, fill_value=0).T
    result.index.name = "station"
    result.columns.name = "year"
    return result
