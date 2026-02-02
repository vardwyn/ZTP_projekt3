import pandas as pd
import pytest

from pm25.data_preparation import extend_metadata_with_station_info


def make_metadata():
    metadata_df = pd.DataFrame(
        {
            "Kod stacji": ["A1", "B2", "C3"],
            "Other": [1, 2, 3],
        }
    )

    updated_metadata = pd.DataFrame(
        {
            "Kod stacji": ["A1", "B2", "C3"],
            "Województwo": ["X", "Y", "Z"],
            "Miejscowość": ["M1", "M2", "M3"],
        }
    )
    return metadata_df, updated_metadata


def test_extend_metadata_basic_merge():
    metadata_df, updated_metadata = make_metadata()
    extra_cols = ["Województwo", "Miejscowość"]

    out = extend_metadata_with_station_info(
        metadata_df, updated_metadata, extra_cols, label="test"
    )

    assert out["Kod stacji"].tolist() == ["A1", "B2", "C3"]
    assert out["Other"].tolist() == [1, 2, 3]
    assert out["Województwo"].tolist() == ["X", "Y", "Z"]
    assert out["Miejscowość"].tolist() == ["M1", "M2", "M3"]


def test_extend_metadata_strips_codes():
    metadata_df, updated_metadata = make_metadata()
    metadata_df["Kod stacji"] = [" A1 ", "B2", "C3 "]
    updated_metadata["Kod stacji"] = ["A1", " B2", "C3"]
    extra_cols = ["Województwo"]

    out = extend_metadata_with_station_info(
        metadata_df, updated_metadata, extra_cols, label="test"
    )

    assert out["Kod stacji"].tolist() == ["A1", "B2", "C3"]
    assert out["Województwo"].tolist() == ["X", "Y", "Z"]


def test_extend_metadata_missing_code_raises_assertion():
    metadata_df, updated_metadata = make_metadata()
    metadata_df.loc[0, "Kod stacji"] = "X9"
    extra_cols = ["Województwo"]

    with pytest.raises(AssertionError):
        extend_metadata_with_station_info(
            metadata_df, updated_metadata, extra_cols, label="test"
        )


def test_extend_metadata_missing_extra_column_raises_key_error():
    metadata_df, updated_metadata = make_metadata()
    extra_cols = ["NieIstnieje"]

    with pytest.raises(KeyError):
        extend_metadata_with_station_info(
            metadata_df, updated_metadata, extra_cols, label="test"
        )
