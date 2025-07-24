import numpy as np
import pandas as pd

from forts.gcs_utils import get_gcs_fs
from forts.load_data.base import LoadDataset


class TourismDataset(LoadDataset):
    DATASET_PATH = f"{LoadDataset.DATASET_PATH}/tourism"
    DATASET_NAME = "Tourism"
    DIR_NAME = "27-3-Athanasopoulos1"

    frequency_pd = {"Yearly": "Y", "Quarterly": "QS", "Monthly": "ME"}

    @classmethod
    def load_data(cls, group):
        assert group in cls.data_group

        ds = {}
        gcs_fs = get_gcs_fs()

        base_path = f"{cls.DATASET_PATH}/{cls.DIR_NAME}"
        train_path = f"{base_path}/{group.lower()}_in.csv"
        test_path = f"{base_path}/{group.lower()}_oos.csv"

        try:
            with gcs_fs.open(train_path) as f:
                train = pd.read_csv(f, header=0, delimiter=",")
            with gcs_fs.open(test_path) as f:
                test = pd.read_csv(f, header=0, delimiter=",")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {base_path}. "
                "Please run the seeding script: python scripts/seed_gcs_datasets.py"
            )

        if group == "Yearly":
            train_meta = train[:2]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[2:].reset_index(drop=True).T
            train = train[2:].reset_index(drop=True).T
        else:
            train_meta = train[:3]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[3:].reset_index(drop=True).T
            train = train[3:].reset_index(drop=True).T

        train_set = [ts[:ts_length] for ts, ts_length in zip(train.values, meta_length)]
        test_set = [ts[:ts_length] for ts, ts_length in zip(test.values, meta_length)]

        for i, idx in enumerate(train.index):
            ds[idx] = np.concatenate([train_set[i], test_set[i]])

        max_len = np.max([len(x) for k, x in ds.items()])
        idx = pd.date_range(
            end=pd.Timestamp("2023-11-01"),
            periods=max_len,
            freq=cls.frequency_pd[group],
        )

        ds = {
            k: pd.Series(series, index=idx[-len(series) :]) for k, series in ds.items()
        }
        df = pd.concat(ds, axis=1)
        df = df.reset_index().melt("index").dropna().reset_index(drop=True)
        df.columns = ["ds", "unique_id", "y"]

        return df
