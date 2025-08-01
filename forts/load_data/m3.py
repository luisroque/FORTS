import pandas as pd

from forts.gcs_utils import get_gcs_fs
from forts.load_data.base import LoadDataset


class M3Dataset(LoadDataset):
    DATASET_NAME = "M3"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/m3"

    def load(self, group=None):
        if group is None:
            raise ValueError("The 'group' parameter is required for this dataset.")

        gcs_path = f"{self.DATASET_PATH}/{group.lower()}.parquet"
        try:
            with get_gcs_fs().open(gcs_path, "rb") as f:
                ds = pd.read_parquet(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {gcs_path}. "
                "Please run the seeding script: python scripts/seed_gcs_datasets.py"
            )
        return ds
