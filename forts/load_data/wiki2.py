import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData

from forts.load_data.base import LoadDataset


class Wiki2Dataset(LoadDataset):
    DATASET_NAME = "Wiki2"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/Wiki2"

    def load(self, group=None):
        ds, *_ = HierarchicalData.load(self.DATASET_PATH, self.DATASET_NAME)
        ds["ds"] = pd.to_datetime(ds["ds"])
        return ds
