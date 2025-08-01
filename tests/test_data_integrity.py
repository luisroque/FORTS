import importlib
import inspect
import pkgutil

import pandas as pd
import pytest

import forts.load_data as load_data_pkg
from forts.load_data.base import LoadDataset


def discover_dataset_classes():
    """Dynamically discovers all LoadDataset subclasses in the forts.load_data package."""
    dataset_classes = []
    for _, name, _ in pkgutil.iter_modules(load_data_pkg.__path__):
        if name not in ("base", "config"):
            module = importlib.import_module(f"forts.load_data.{name}")
            for _, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, LoadDataset)
                    and obj is not LoadDataset
                ):
                    dataset_classes.append(obj)
    return dataset_classes


@pytest.mark.parametrize("DatasetClass", discover_dataset_classes())
def test_dataset_integrity(DatasetClass):
    """
    Tests that each dataset loader returns a DataFrame with the correct
    structure and that the 'ds' column is of datetime type.
    """
    # Instantiate the dataset class
    try:
        dataset = DatasetClass()
    except Exception as e:
        pytest.fail(f"Failed to instantiate {DatasetClass.__name__}: {e}")

    # Load the data
    try:
        # For datasets that require a 'group', we provide a default one.
        # This is a simplification for the purpose of this test.
        if "group" in inspect.signature(dataset.load).parameters:
            df = dataset.load(group=dataset.data_group[0])
        else:
            df = dataset.load()
    except Exception as e:
        pytest.fail(f"Failed to load data from {DatasetClass.__name__}: {e}")

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

    # Verify that 'y' column is numeric
    assert pd.api.types.is_numeric_dtype(
        df["y"]
    ), f"The 'y' column in {DatasetClass.__name__} is not a numeric type."

    # Verify 'unique_id' is not empty and is of a consistent type
    assert (
        df["unique_id"].notna().all()
    ), f"The 'unique_id' column in {DatasetClass.__name__} contains null values."
