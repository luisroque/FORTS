import pandas as pd

from forts.gcs_utils import get_gcs_fs
from forts.load_data.base import LoadDataset


class M1Dataset(LoadDataset):
    DATASET_NAME = "M1"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/m1"

    horizons_map = {
        "Quarterly": 2,
        "Monthly": 8,
    }

    frequency_map = {
        "Quarterly": 4,
        "Monthly": 12,
    }

    context_length = {
        "Quarterly": 4,
        "Monthly": 12,
    }

    min_samples = {
        "Quarterly": 22,
        "Monthly": 52,
    }

    frequency_pd = {
        "Quarterly": "QE",
        "Monthly": "ME",
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    def load(self, group=None, min_n_instances=None):
        if group is None:
            raise ValueError("The 'group' parameter is required for this dataset.")
        gcs_path = f"{self.DATASET_PATH}/{group.lower()}.parquet"
        try:
            with get_gcs_fs().open(gcs_path, "rb") as f:
                df = pd.read_parquet(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {gcs_path}. "
                "Please run the seeding script: python scripts/seed_gcs_datasets.py"
            )

        if min_n_instances is not None:
            df = self.prune_df_by_size(df, min_n_instances)

        df["ds"] = pd.to_datetime(df["ds"])
        return df
