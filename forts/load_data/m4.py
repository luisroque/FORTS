import pandas as pd

from forts.gcs_utils import get_gcs_fs
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

    @classmethod
    def load_data(cls, group, min_n_instances=None):
        gcs_path = f"{cls.DATASET_PATH}/{group.lower()}.parquet"
        try:
            with get_gcs_fs().open(gcs_path, "rb") as f:
                ds = pd.read_parquet(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {gcs_path}. "
                "Please run the seeding script: python scripts/seed_gcs_datasets.py"
            )

        if min_n_instances is not None:
            ds = cls.prune_df_by_size(ds, min_n_instances)

        return ds
