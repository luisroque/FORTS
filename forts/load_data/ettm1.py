import os

import pandas as pd
from datasetsforecast.long_horizon import LongHorizon

from forts.load_data.base import LoadDataset


class ETTm1Dataset(LoadDataset):
    DATASET_NAME = "ETTm1"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}"
    LOCAL_CACHE_PATH = "assets/datasets"

    def load(self, group=None):
        # Check local cache first
        local_dir = os.path.join(self.LOCAL_CACHE_PATH, "longhorizon/datasets")
        if os.path.exists(os.path.join(local_dir, self.DATASET_NAME)):
            print(f"Loading {self.DATASET_NAME} from local cache: {local_dir}")
            ds, *_ = LongHorizon.load(local_dir, self.DATASET_NAME)
        else:
            print(f"Loading {self.DATASET_NAME} from GCS: {self.DATASET_PATH}")
            ds, *_ = LongHorizon.load(self.DATASET_PATH, self.DATASET_NAME)

        ds["ds"] = pd.to_datetime(ds["ds"])
        return ds
