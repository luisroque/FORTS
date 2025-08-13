import os

import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData

from forts.load_data.base import LoadDataset


class Wiki2Dataset(LoadDataset):
    DATASET_NAME = "Wiki2"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}"
    LOCAL_CACHE_PATH = "assets/datasets"

    def load(self, group=None):
        # Check local cache first
        local_dir = os.path.join(self.LOCAL_CACHE_PATH, "hierarchical")
        if os.path.exists(os.path.join(local_dir, self.DATASET_NAME)):
            print(f"Loading {self.DATASET_NAME} from local cache: {local_dir}")
            ds, *_ = HierarchicalData.load(local_dir, self.DATASET_NAME)
        else:
            print(f"Loading {self.DATASET_NAME} from GCS: {self.DATASET_PATH}")
            ds, *_ = HierarchicalData.load(self.DATASET_PATH, self.DATASET_NAME)

        ds["ds"] = pd.to_datetime(ds["ds"])
        return ds
