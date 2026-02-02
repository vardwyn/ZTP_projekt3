import pandas as pd

from pm25.statistics_calculation import report_nan_runs


def test_report_nan_runs_basic():
    idx = pd.date_range("2024-01-01 00:00:00", periods=6, freq="h")
    df = pd.DataFrame(
        {
            "A1": [1.0, None, None, 4.0, None, 6.0],
            "B2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=idx,
    )

    report = report_nan_runs(df, top_k=2)

    assert report["total_stations"] == 2
    assert report["stations_with_any_nan"] == 1
    assert "A1" in report["nan_runs"]
    assert "B2" not in report["nan_runs"]

    runs = report["nan_runs"]["A1"]
    assert len(runs) == 2
    assert runs.iloc[0]["length"] >= runs.iloc[1]["length"]


def test_report_nan_runs_all_nan():
    idx = pd.date_range("2024-01-01 00:00:00", periods=3, freq="h")
    df = pd.DataFrame({"A1": [None, None, None]}, index=idx)

    report = report_nan_runs(df, top_k=1)

    runs = report["nan_runs"]["A1"]
    assert runs.iloc[0]["length"] == 3 / 24
