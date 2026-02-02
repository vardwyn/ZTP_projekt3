import pandas as pd
import pytest

from pm25.statistics_calculation import (
    analyze_raw_df,
    check_timestamps,
    monthly_avg_with_nan_threshold,
    average_by_city,
    count_days_over_threshold,
)


def make_raw_df():
    return pd.DataFrame(
        [
            ["Nr", 1, 2],
            ["Kod stacji", "A1", "B2"],
            ["Kod stanowiska", "A1-1", "B2-1"],
            ["Czas uśredniania", "1g", "1g"],
            ["Wskaźnik", "PM2.5", "PM2.5"],
            ["2024-01-01 00:00:00", 1.0, None],
            ["2024-01-01 01:00:00", 2.0, 3.0],
        ]
    )


def test_analyze_raw_df_basic():
    df = make_raw_df()
    meta_keys = ["Nr", "Kod stacji", "Kod stanowiska", "Czas uśredniania", "Wskaźnik"]

    result = analyze_raw_df(df, meta_keys)

    assert result["shape"]["rows"] == 7
    assert result["shape"]["columns"] == 3
    assert result["timestamp_counts"]["timestamp_rows"] == 2
    assert result["timestamp_counts"]["metadata_rows"] == 5
    assert result["timestamp_counts"]["stations"] == 2
    assert result["unique_values"]["Czas uśredniania"] == ["1g"]
    assert result["unique_values"]["Wskaźnik"] == ["PM2.5"]


def test_analyze_raw_df_missing_meta_keys_safe():
    df = make_raw_df()

    result = analyze_raw_df(df, meta_keys=None)

    assert result["timestamp_counts"]["timestamp_rows"] == 7
    assert result["timestamp_counts"]["metadata_rows"] == 0


def test_analyze_raw_df_missing_metadata_rows():
    df = pd.DataFrame(
        [
            ["2024-01-01 00:00:00", 1.0, 2.0],
            ["2024-01-01 01:00:00", 3.0, 4.0],
        ]
    )
    meta_keys = ["Nr", "Kod stacji"]

    result = analyze_raw_df(df, meta_keys)

    assert result["unique_values"]["Czas uśredniania"] is None
    assert result["unique_values"]["Wskaźnik"] is None


def test_check_timestamps_basic_hourly():
    df = make_raw_df()
    meta_keys = ["Nr", "Kod stacji", "Kod stanowiska", "Czas uśredniania", "Wskaźnik"]

    result = check_timestamps(df, meta_keys, expected_freq="h")

    assert result["summary"]["n_timestamp_rows"] == 2
    assert result["summary"]["n_invalid_timestamps"] == 0
    assert result["summary"]["main_delta"] == pd.Timedelta(hours=1)


def test_check_timestamps_invalid_and_duplicates():
    df = pd.DataFrame(
        [
            ["Nr", 1, 2],
            ["Kod stacji", "A1", "B2"],
            ["2024-01-01 00:00:00", 1.0, 2.0],
            ["not-a-date", 3.0, 4.0],
            ["2024-01-01 00:00:00", 5.0, 6.0],
        ]
    )
    meta_keys = ["Nr", "Kod stacji"]

    result = check_timestamps(df, meta_keys, expected_freq="h")

    assert result["summary"]["n_invalid_timestamps"] == 1
    assert result["summary"]["n_duplicates"] == 2


def test_check_timestamps_empty_or_single():
    df = pd.DataFrame(
        [
            ["Nr", 1, 2],
            ["Kod stacji", "A1", "B2"],
            ["2024-01-01 00:00:00", 1.0, 2.0],
        ]
    )
    meta_keys = ["Nr", "Kod stacji"]

    result = check_timestamps(df, meta_keys)

    assert result["summary"]["n_timestamp_rows"] == 1
    assert result["summary"]["main_delta"] is None


def test_monthly_avg_with_nan_threshold_masks():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-02-01 00:00:00",
        ]
    )
    df = pd.DataFrame({"A1": [None, 10.0, None]}, index=idx)

    monthly = monthly_avg_with_nan_threshold(df, max_nan_per_month=0)

    # Styczeń ma 1 NaN -> maskowany
    assert pd.isna(monthly.loc["2024-01-31", "A1"])


def test_monthly_avg_with_nan_threshold_keeps_good_months():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-02-01 00:00:00",
            "2024-02-01 01:00:00",
        ]
    )
    df = pd.DataFrame({"A1": [1.0, 3.0, 2.0, 4.0]}, index=idx)

    monthly = monthly_avg_with_nan_threshold(df, max_nan_per_month=1)

    # 2024-01-01 00:00 przesuwa się do grudnia 2023, a 2024-02-01 00:00 do stycznia
    assert monthly.loc["2024-01-31", "A1"] == (3.0 + 2.0) / 2
    assert monthly.loc["2024-02-29", "A1"] == 4.0


def test_average_by_city_basic():
    meas = pd.DataFrame(
        {
            "A1": [1.0, 2.0],
            "B2": [3.0, 4.0],
            "C3": [5.0, 6.0],
        }
    )
    meta = pd.DataFrame(
        {
            "Kod stacji": ["A1", "B2", "C3"],
            "Miejscowość": ["X", "X", "Y"],
        }
    )

    city = average_by_city(meas, meta)

    assert list(city.columns) == ["X", "Y"]
    assert city["X"].tolist() == [(1.0 + 3.0) / 2, (2.0 + 4.0) / 2]
    assert city["Y"].tolist() == [5.0, 6.0]


def test_average_by_city_with_whitespace_codes():
    meas = pd.DataFrame({" A1 ": [1.0], "B2": [3.0]})
    meta = pd.DataFrame({"Kod stacji": ["A1", " B2 "], "Miejscowość": ["X", "X"]})

    city = average_by_city(meas, meta)

    assert city["X"].tolist() == [(1.0 + 3.0) / 2]


def test_average_by_city_missing_city_raises():
    meas = pd.DataFrame({"A1": [1.0], "B2": [2.0]})
    meta = pd.DataFrame({"Kod stacji": ["A1", "B2"], "Miejscowość": ["X", None]})

    with pytest.raises(AssertionError):
        average_by_city(meas, meta)


def test_average_by_city_missing_station_in_meta_raises():
    meas = pd.DataFrame({"A1": [1.0], "B2": [2.0]})
    meta = pd.DataFrame({"Kod stacji": ["A1"], "Miejscowość": ["X"]})

    with pytest.raises(AssertionError):
        average_by_city(meas, meta)


def test_count_days_over_threshold_basic():
    idx = pd.to_datetime(
        [
            "2024-01-01 23:00:00",
            "2024-01-02 00:00:00",
            "2024-01-02 01:00:00",
        ]
    )
    df = pd.DataFrame({"A1": [10.0, 50.0, 10.0]}, index=idx)

    result = count_days_over_threshold(df, threshold=20.0, years=(2024,))

    assert result.loc["A1", 2024] == 1


def test_count_days_over_threshold_multiple_years_and_stations():
    idx = pd.to_datetime(
        [
            "2023-12-31 23:00:00",
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-12-31 23:00:00",
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "A1": [10.0, 50.0, 10.0, 10.0, 50.0, 10.0],
            "B2": [0.0, 0.0, 0.0, 30.0, 30.0, 30.0],
        },
        index=idx,
    )

    result = count_days_over_threshold(df, threshold=20.0, years=(2023, 2024, 2025))

    assert result.loc["A1", 2023] == 1
    assert result.loc["A1", 2024] == 1
    assert result.loc["A1", 2025] == 0

    assert result.loc["B2", 2023] == 0
    assert result.loc["B2", 2024] == 1
    assert result.loc["B2", 2025] == 1
