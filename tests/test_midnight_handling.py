import pandas as pd

from statistics_calculation import (
    shift_midnight_to_previous_day,
    hourly_to_daily_30d_sma,
    count_days_over_threshold,
    monthly_avg_with_nan_threshold,
)


def test_shift_midnight_to_previous_day_preserves_order():
    idx = pd.to_datetime(
        [
            "2024-01-01 23:00:00",
            "2024-01-02 00:00:00",
            "2024-01-02 01:00:00",
        ]
    )

    shifted = shift_midnight_to_previous_day(idx)

    assert list(shifted) == sorted(shifted)
    assert shifted[1].date() == pd.Timestamp("2024-01-01").date()
    assert shifted[1] > shifted[0]


def test_shift_midnight_to_previous_day_keeps_relative_order_in_unsorted_index():
    idx = pd.to_datetime(
        [
            "2024-01-02 01:00:00",
            "2024-01-02 00:00:00",
            "2024-01-01 23:00:00",
        ]
    )

    shifted = shift_midnight_to_previous_day(idx)

    # Kolejność elementów nie powinna się zmieniać (pozycje są te same)
    assert list(shifted) == [
        pd.Timestamp("2024-01-02 01:00:00"),
        pd.Timestamp("2024-01-01 23:59:59.999999999"),
        pd.Timestamp("2024-01-01 23:00:00"),
    ]
    assert shifted[1].date() == pd.Timestamp("2024-01-01").date()


def test_count_days_over_threshold_midnight_in_previous_day():
    idx = pd.to_datetime(
        [
            "2024-01-01 23:00:00",
            "2024-01-02 00:00:00",
            "2024-01-02 01:00:00",
        ]
    )
    df = pd.DataFrame({"A1": [10.0, 50.0, 10.0]}, index=idx)

    result = count_days_over_threshold(df, threshold=20.0, years=(2024,))

    # 2024-01-01 [10, 50] -> mean 30 > threshold => 1 day
    # 2024-01-02 [10] -> mean 10 <= threshold => 0 days
    assert result.loc["A1", 2024] == 1


def test_count_days_over_threshold_multi_station_multi_year():
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

    # A1: 2023-12-31 [10,50] -> mean 30 => 1 day in 2023
    #      2024-01-01 [10] -> 0 days in 2024
    #      2024-12-31 [10,50] -> 1 day in 2024
    #      2025-01-01 [10] -> 0 days in 2025
    assert result.loc["A1", 2023] == 1
    assert result.loc["A1", 2024] == 1
    assert result.loc["A1", 2025] == 0

    # B2: 2023-12-31 [0,0] -> 0
    #     2024-12-31 [30,30] -> 1 day in 2024
    #     2025-01-01 [30] -> 1 day in 2025
    assert result.loc["B2", 2023] == 0
    assert result.loc["B2", 2024] == 1
    assert result.loc["B2", 2025] == 1


def test_hourly_to_daily_30d_sma_midnight_in_previous_day():
    idx = pd.to_datetime(
        [
            "2024-01-01 23:00:00",
            "2024-01-02 00:00:00",
            "2024-01-02 01:00:00",
        ]
    )
    df = pd.DataFrame({"A1": [10.0, 50.0, 10.0]}, index=idx)

    daily = hourly_to_daily_30d_sma(df, window_days=1, min_periods=1)

    assert daily.loc["2024-01-01", "A1"] == 30.0
    assert daily.loc["2024-01-02", "A1"] == 10.0


def test_hourly_to_daily_30d_sma_multiple_stations():
    idx = pd.date_range("2024-01-01 22:00:00", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "A1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B2": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=idx,
    )

    daily = hourly_to_daily_30d_sma(df, window_days=1, min_periods=1)

    # 2024-01-01: 22:00, 23:00, 00:00 (shifted)
    assert daily.loc["2024-01-01", "A1"] == (1 + 2 + 3) / 3
    # 2024-01-02: 01:00, 02:00
    assert daily.loc["2024-01-02", "A1"] == (4 + 5) / 2


def test_monthly_avg_with_nan_threshold_midnight_in_previous_day():
    idx = pd.to_datetime(
        [
            "2024-01-31 23:00:00",
            "2024-02-01 00:00:00",
        ]
    )
    df = pd.DataFrame({"A1": [10.0, 50.0]}, index=idx)

    monthly = monthly_avg_with_nan_threshold(df, max_nan_per_month=100)

    jan = monthly.loc["2024-01-31", "A1"]
    assert jan == 30.0


def test_monthly_avg_with_nan_threshold_masks_bad_months():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-02-01 00:00:00",
        ]
    )
    df = pd.DataFrame({"A1": [None, 10.0, None]}, index=idx)

    monthly = monthly_avg_with_nan_threshold(df, max_nan_per_month=0)

    # Styczeń ma 1 NaN -> maskowany; luty ma 1 NaN (po przesunięciu 00:00 do stycznia)
    assert pd.isna(monthly.loc["2024-01-31", "A1"])
