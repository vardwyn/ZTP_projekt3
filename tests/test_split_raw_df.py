import pandas as pd
import pytest

from data_preparation import split_raw_df_to_metadata_and_measurements


def make_raw_df():
    # Minimalny surowy df
    return pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["2024-01-01 00:00:00", 1.0, 2.0],
            ["2024-01-01 01:00:00", 3.0, 4.0],
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
    assert set(measurements.index.year) == {2024}


def test_split_with_no_meta_keys_treats_all_as_data():
    raw = pd.DataFrame(
        [
            ["2024-01-01 00:00:00", 1.0, 2.0],
            ["2024-01-01 01:00:00", 3.0, 4.0],
        ]
    )
    meta_keys = []

    stations, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert stations.empty
    assert measurements.shape[0] == raw.shape[0]
    assert list(measurements.columns) == list(raw.columns[1:])


def test_split_meta_missing_kod_stacji_uses_data_columns():
    raw = pd.DataFrame(
        [
            ["Czas uśredniania", "1g", "1g"],
            ["2024-01-01 00:00:00", 1.0, 2.0],
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
            ["2024-01-01 00:00:00", 1.0, 2.0],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    stations, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert list(measurements.columns) == ["S1", "S1"]
    assert measurements.shape == (1, 2)


def test_split_invalid_timestamps_raises():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["not-a-date", 1.0, 2.0],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    with pytest.raises(AssertionError):
        split_raw_df_to_metadata_and_measurements(raw, meta_keys)


def test_split_parses_microseconds_timestamps():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["2018-02-21 16:00:00.123456", 1.0, 2.0],
            ["2018-02-21 17:00:00.000001", 3.0, 4.0],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    _, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert measurements.index.isna().sum() == 0
    assert measurements.index[0] == pd.Timestamp("2018-02-21 16:00:00")


def test_split_rejects_iso_t_separator():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["2018-02-21T16:00:00", 1.0, 2.0],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    with pytest.raises(AssertionError):
        split_raw_df_to_metadata_and_measurements(raw, meta_keys)


def test_split_rejects_rows_from_other_years():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["2015-12-31 23:00:00", 1.0, 2.0],
            ["2016-01-01 00:00:00", 3.0, 4.0],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    with pytest.warns(UserWarning, match="Wykryto wiersze spoza głównego roku"):
        _, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert set(measurements.index.year) == {2015}


def test_split_converts_decimal_commas_to_numbers():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["2024-01-01 00:00:00", "1,5", "2.75"],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    _, measurements = split_raw_df_to_metadata_and_measurements(raw, meta_keys)

    assert measurements.iloc[0, 0] == 1.5
    assert measurements.iloc[0, 1] == 2.75
    assert measurements.dtypes.apply(pd.api.types.is_numeric_dtype).all()


def test_split_rejects_non_numeric_measurements():
    raw = pd.DataFrame(
        [
            ["Kod stacji", "S1", "S2"],
            ["Czas uśredniania", "1g", "1g"],
            ["2024-01-01 00:00:00", "bad", "2.75"],
        ]
    )
    meta_keys = ["Kod stacji", "Czas uśredniania"]

    with pytest.raises(AssertionError):
        split_raw_df_to_metadata_and_measurements(raw, meta_keys)
