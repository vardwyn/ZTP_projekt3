import pandas as pd
import pytest

from pm25.data_preparation import (
    build_station_code_mapping,
    update_station_names_metadata,
    update_station_names_data,
)


def make_updated_metadata():
    return pd.DataFrame(
        {
            "Kod stacji": [" A1 ", "B2", "C3"],
            "Stary Kod stacji": ["A0, A-1", None, " C0 "],
        }
    )


def test_build_station_code_mapping_handles_old_codes_and_whitespace():
    updated_metadata = make_updated_metadata()

    mapping = build_station_code_mapping(updated_metadata, verbose=False)

    assert mapping["A1"] == "A1"
    assert mapping["B2"] == "B2"
    assert mapping["C3"] == "C3"
    assert mapping["A0"] == "A1"
    assert mapping["A-1"] == "A1"
    assert mapping["C0"] == "C3"


def test_update_station_names_metadata_updates_and_merges():
    updated_metadata = make_updated_metadata()
    mapping = build_station_code_mapping(updated_metadata, verbose=False)

    metadata_df = pd.DataFrame(
        {
            "Kod stacji": ["A0", "B2", "C3"],
            "Other": [1, 2, 3],
        }
    )

    out = update_station_names_metadata(metadata_df, updated_metadata, mapping, label="test")

    assert out["Kod stacji"].tolist() == ["A1", "B2", "C3"]
    assert "Stary Kod stacji" in out.columns
    assert out["Other"].tolist() == [1, 2, 3]


def test_update_station_names_metadata_raises_on_unmatched():
    updated_metadata = make_updated_metadata()
    mapping = build_station_code_mapping(updated_metadata, verbose=False)

    metadata_df = pd.DataFrame({"Kod stacji": ["A0", "X9"]})

    with pytest.raises(ValueError) as exc:
        update_station_names_metadata(metadata_df, updated_metadata, mapping, label="test")

    assert "X9" in str(exc.value)


def test_update_station_names_data_renames_columns():
    updated_metadata = make_updated_metadata()
    mapping = build_station_code_mapping(updated_metadata, verbose=False)

    data_df = pd.DataFrame(
        {
            "A0": [1.0, 2.0],
            "B2": [3.0, 4.0],
            "C0": [5.0, 6.0],
        }
    )

    out = update_station_names_data(data_df, mapping, label="test")

    assert list(out.columns) == ["A1", "B2", "C3"]


def test_update_station_names_data_allows_duplicate_targets():
    updated_metadata = pd.DataFrame(
        {
            "Kod stacji": ["A1"],
            "Stary Kod stacji": ["A0, A-1"],
        }
    )
    mapping = build_station_code_mapping(updated_metadata, verbose=False)

    data_df = pd.DataFrame(
        {
            "A0": [1.0, 2.0],
            "A-1": [3.0, 4.0],
        }
    )

    out = update_station_names_data(data_df, mapping, label="test")

    assert list(out.columns) == ["A1", "A1"]


def test_update_station_names_data_strips_and_keeps_unmapped():
    updated_metadata = make_updated_metadata()
    mapping = build_station_code_mapping(updated_metadata, verbose=False)

    data_df = pd.DataFrame(
        {
            " A0 ": [1.0, 2.0],
            " X9 ": [3.0, 4.0],
        }
    )

    out = update_station_names_data(data_df, mapping, label="test")

    assert list(out.columns) == ["A1", "X9"]
