import pandas as pd

from data_preparation import combine_metadata_frames, combine_data_frames


def test_combine_metadata_frames_basic_intersection():
    df1 = pd.DataFrame({"Kod stacji": ["A1", "B2", "C3"], "X": [1, 2, 3]})
    df2 = pd.DataFrame({"Kod stacji": ["B2", "C3", "D4"], "X": [4, 5, 6]})

    combined, common = combine_metadata_frames([df1, df2])

    assert common == {"B2", "C3"}
    assert combined["Kod stacji"].tolist() == ["B2", "C3"]
    assert list(combined.columns) == ["Kod stacji", "X"]


def test_combine_metadata_frames_strips_codes():
    df1 = pd.DataFrame({"Kod stacji": [" A1 ", "B2"], "X": [1, 2]})
    df2 = pd.DataFrame({"Kod stacji": ["A1", " B2 "], "X": [3, 4]})

    combined, common = combine_metadata_frames([df1, df2])

    assert common == {"A1", "B2"}
    assert combined["Kod stacji"].tolist() == ["A1", "B2"]


def test_combine_metadata_frames_single_input():
    df1 = pd.DataFrame({"Kod stacji": ["A1", "B2"], "X": [1, 2]})

    combined, common = combine_metadata_frames([df1])

    assert common == {"A1", "B2"}
    assert combined["Kod stacji"].tolist() == ["A1", "B2"]


def test_combine_metadata_frames_duplicates_preserved_in_base():
    df1 = pd.DataFrame({"Kod stacji": ["A1", "A1", "B2"], "X": [1, 2, 3]})
    df2 = pd.DataFrame({"Kod stacji": ["A1", "B2"], "X": [4, 5]})

    combined, common = combine_metadata_frames([df1, df2])

    assert common == {"A1", "B2"}
    assert combined["Kod stacji"].tolist() == ["A1", "A1", "B2"]


def test_combine_data_frames_basic_inner_columns():
    df1 = pd.DataFrame({"A1": [1, 2], "B2": [3, 4], "C3": [5, 6]})
    df2 = pd.DataFrame({"B2": [7, 8], "C3": [9, 10], "D4": [11, 12]})

    combined, common_cols = combine_data_frames([df1, df2])

    assert list(common_cols) == ["B2", "C3"]
    assert list(combined.columns) == ["B2", "C3"]
    assert combined.shape[0] == 4


def test_combine_data_frames_single_input():
    df1 = pd.DataFrame({"A1": [1, 2], "B2": [3, 4]})

    combined, common_cols = combine_data_frames([df1])

    assert list(common_cols) == ["A1", "B2"]
    assert combined.shape == df1.shape


def test_combine_data_frames_empty_intersection():
    df1 = pd.DataFrame({"A1": [1, 2]})
    df2 = pd.DataFrame({"B2": [3, 4]})

    combined, common_cols = combine_data_frames([df1, df2])

    assert common_cols == []
    assert combined.shape[1] == 0
