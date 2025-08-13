import pandas as pd

from forts.gcs_utils import gcs_read_parquet
from forts.load_data.base import LoadDataset


class M4Dataset(LoadDataset):
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/m4"
    DATASET_NAME = "M4"

    horizons_map = {
        "Quarterly": 8,
        "Monthly": 12,
    }

    frequency_map = {
        "Quarterly": 4,
        "Monthly": 12,
    }

    context_length = {
        "Quarterly": 10,
        "Monthly": 24,
    }

    min_samples = {
        "Quarterly": (8 + 10 + 1) * 2,
        "Monthly": (24 + 12 + 1) * 2,
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
            ds = gcs_read_parquet(gcs_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {gcs_path}. "
                "Please run the seeding script: python scripts/seed_gcs_datasets.py"
            )

        if min_n_instances is not None:
            ds = self.prune_df_by_size(ds, min_n_instances)

        ds["ds"] = pd.to_datetime(ds["ds"])
        return ds
