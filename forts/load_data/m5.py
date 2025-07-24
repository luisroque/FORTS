import pandas as pd

from forts.gcs_utils import get_gcs_fs
from forts.load_data.base import LoadDataset


class M5Dataset(LoadDataset):
    DATASET_NAME = "M5"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/m5"

    @classmethod
    def load_data(cls, group=None):
        gcs_path = f"{cls.DATASET_PATH}/daily.parquet"
        try:
            with get_gcs_fs().open(gcs_path, "rb") as f:
                ds = pd.read_parquet(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {gcs_path}. "
                "Please run the seeding script: python scripts/seed_gcs_datasets.py"
            )
        return ds
