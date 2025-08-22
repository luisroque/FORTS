import pandas as pd
import pytest

from forts.load_data.m1 import M1Dataset
from forts.load_data.m3 import M3Dataset
from forts.load_data.m4 import M4Dataset
from forts.load_data.m5 import M5Dataset
from forts.load_data.tourism import TourismDataset
from forts.load_data.traffic import TrafficDataset


@pytest.mark.parametrize(
    "DatasetClass, group",
    [
        pytest.param(TourismDataset, "Monthly", id="TourismDataset-Monthly"),
        pytest.param(M1Dataset, "Monthly", id="M1Dataset-Monthly"),
        pytest.param(M1Dataset, "Quarterly", id="M1Dataset-Quarterly"),
        pytest.param(M3Dataset, "Monthly", id="M3Dataset-Monthly"),
        pytest.param(M3Dataset, "Quarterly", id="M3Dataset-Quarterly"),
        pytest.param(M3Dataset, "Yearly", id="M3Dataset-Yearly"),
        pytest.param(M4Dataset, "Monthly", id="M4Dataset-Monthly"),
        pytest.param(M4Dataset, "Quarterly", id="M4Dataset-Quarterly"),
        pytest.param(TrafficDataset, "Daily", id="TrafficDataset-Daily"),
        pytest.param(M5Dataset, "Daily", id="M5Dataset-Daily"),
    ],
)
def test_dataset_integrity(DatasetClass, group):
    """
    Tests that each dataset loader returns a DataFrame with the correct
    structure and that the 'ds' column is of datetime type with correct frequency.
    """
    # Instantiate the dataset class
    try:
        dataset = DatasetClass()
    except Exception as e:
        pytest.fail(f"Failed to instantiate {DatasetClass.__name__}: {e}")

    # Load the data
    try:
        if group:
            df = dataset.load(group=group)
        else:
            df = dataset.load()
    except Exception as e:
        pytest.fail(
            f"Failed to load data from {DatasetClass.__name__} (group: {group}): {e}"
        )

    # --- Assertions ---
    assert isinstance(
        df, pd.DataFrame
    ), f"{DatasetClass.__name__} did not return a pandas DataFrame."
    assert not df.empty, f"DataFrame from {DatasetClass.__name__} is empty."

    expected_columns = {"unique_id", "ds", "y"}
    assert expected_columns.issubset(
        df.columns
    ), f"DataFrame from {DatasetClass.__name__} is missing required columns."

    # Verify that the 'ds' column is of datetime type
    assert pd.api.types.is_datetime64_any_dtype(
        df["ds"]
    ), f"The 'ds' column in {DatasetClass.__name__} is not a datetime type."

    # Verify frequency if the dataset provides that information
    if group and hasattr(dataset, "frequency_pd") and group in dataset.frequency_pd:
        expected_freq = dataset.frequency_pd[group]
        # We check frequency for each individual time series.
        for unique_id, series_df in df.groupby("unique_id", observed=False):
            # infer_freq needs at least 2 points, and more to be reliable.
            if len(series_df) > 2:
                series_df = series_df.sort_values("ds")
                inferred_freq = pd.infer_freq(series_df["ds"])

                if inferred_freq is not None:
                    # M, ME, MS -> M. Q, QE, QS -> Q.
                    expected_base_freq = expected_freq[0]
                    inferred_base_freq = inferred_freq[0]
                    assert expected_base_freq == inferred_base_freq, (
                        f"Inferred frequency '{inferred_freq}' for unique_id '{unique_id}' "
                        f"in {DatasetClass.__name__} (group: {group}) does not match "
                        f"expected frequency '{expected_freq}' (base freq mismatch: "
                        f"{inferred_base_freq} vs {expected_base_freq})."
                    )

    # Verify that 'y' column is numeric
    assert pd.api.types.is_numeric_dtype(
        df["y"]
    ), f"The 'y' column in {DatasetClass.__name__} is not a numeric type."

    # Verify 'unique_id' is not empty and is of a consistent type
    assert (
        df["unique_id"].notna().all()
    ), f"The 'unique_id' column in {DatasetClass.__name__} contains null values."
