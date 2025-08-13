from forts.gcs_utils import gcs_read_parquet
from forts.load_data.base import LoadDataset


class M3Dataset(LoadDataset):
    DATASET_NAME = "M3"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/m3"

    def load(self, group=None):
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
        return ds
