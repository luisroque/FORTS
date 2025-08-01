import pandas as pd
from datasetsforecast.long_horizon import LongHorizon

from forts.load_data.base import LoadDataset


class ETTm2Dataset(LoadDataset):
    DATASET_NAME = "ETTm2"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}"

    def load(self, group=None):
        ds, *_ = LongHorizon.load(self.DATASET_PATH, self.DATASET_NAME)
        ds["ds"] = pd.to_datetime(ds["ds"])
        return ds
