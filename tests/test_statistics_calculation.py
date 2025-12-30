import pandas as pd

from statistics_calculation import analyze_raw_df, check_timestamps


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
