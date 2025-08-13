from forts.gcs_utils import gcs_read_parquet
from forts.load_data.base import LoadDataset


class M5Dataset(LoadDataset):
    DATASET_NAME = "M5"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/m5"

    def load(self, group=None):
        gcs_path = f"{self.DATASET_PATH}/daily.parquet"
        try:
            ds = gcs_read_parquet(gcs_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {gcs_path}. "
                "Please run the seeding script: python scripts/seed_gcs_datasets.py"
            )
        return ds
