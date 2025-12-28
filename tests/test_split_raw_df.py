import pandas as pd
import pytest

from data_preparation import split_raw_df_to_metadata_and_measurements


def make_raw_df():
    # Minimalny surowy df
    return pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["2024-01-01 00:00", 1.0, 2.0],
            ["2024-01-01 01:00", 3.0, 4.0],
        ]
    )


def test_split_basic_structure():
    raw = make_raw_df()
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    stations, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert list(stations.columns) == meta_keys
    assert stations["Kod stacji"].tolist() == ["S1", "S2"]
    assert list(measurements.columns) == ["S1", "S2"]
    assert measurements.shape == (2, 2)
    assert pd.api.types.is_datetime64_any_dtype(measurements.index)


def test_split_with_no_meta_keys_treats_all_as_data():
    raw = make_raw_df()
    meta_keys = []

    stations, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert stations.empty
    assert measurements.shape[0] == raw.shape[0]
    assert list(measurements.columns) == list(raw.columns[1:])


def test_split_meta_missing_kod_stacji_uses_data_columns():
    raw = pd.DataFrame(
        [
            ["Czas uśredniania", "1g", "1g"],
            ["2024-01-01 00:00", 1.0, 2.0],
        ]
    )
    meta_keys = ["Czas uśredniania"]

    stations, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert "Czas uśredniania" in stations.columns
    assert list(measurements.columns) == list(raw.columns[1:])


def test_split_duplicate_station_codes_allowed():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S1"],
            ["Czas uśredniania", "1g", "1g"],
            ["2024-01-01 00:00", 1.0, 2.0],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    stations, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert list(measurements.columns) == ["S1", "S1"]
    assert measurements.shape == (1, 2)


def test_split_invalid_timestamps_become_nat():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["not-a-date", 1.0, 2.0],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    _, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert measurements.index.isna().any()
