import pandas as pd
from datasetsforecast.long_horizon import LongHorizon

from forts.load_data.base import LoadDataset


class WeatherDataset(LoadDataset):
    DATASET_NAME = "Weather"
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}"

    def load(self):
        ds, *_ = LongHorizon.load(self.DATASET_PATH, self.DATASET_NAME)
        ds["ds"] = pd.to_datetime(ds["ds"])
        return ds
