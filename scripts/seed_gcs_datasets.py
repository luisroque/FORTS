import os
import tempfile
import time
import zipfile
from io import BytesIO

import pandas as pd
import requests
from datasetsforecast.hierarchical import HierarchicalData
from datasetsforecast.long_horizon import LongHorizon
from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4
from datasetsforecast.m5 import M5
from gluonts.dataset.repository.datasets import get_dataset

from forts.gcs_utils import gcs_path_exists, get_datasets_path, get_gcs_fs

GCS_DATASETS_PATH = get_datasets_path()
GCS_FS = get_gcs_fs()

LOCAL_CACHE_PATH = "assets/datasets"


def seed_m1_dataset(group: str, temp_dir: str):
    """Downloads M1 data, saves as parquet, and uploads to GCS."""
    gcs_path = f"{GCS_DATASETS_PATH}/m1/{group.lower()}.parquet"
    if gcs_path_exists(gcs_path):
        print(f"M1 {group} data already exists in GCS. Skipping.")
        return

    print(f"Processing M1 {group} dataset...")
    dataset = get_dataset(f"m1_{group.lower()}", regenerate=False)
    df_list = []
    for i, series in enumerate(dataset.train):
        s = pd.Series(
            series["target"],
            index=pd.date_range(
                start=series["start"].to_timestamp(),
                freq=series["start"].freq,
                periods=len(series["target"]),
            ),
        )
        s_df = s.reset_index()
        s_df.columns = ["ds", "y"]
        s_df["unique_id"] = f"ID{i}"
        df_list.append(s_df)

    df = pd.concat(df_list).reset_index(drop=True)
    local_path = os.path.join(temp_dir, f"m1_{group.lower()}.parquet")
    df.to_parquet(local_path)
    GCS_FS.put(local_path, gcs_path)
    print(f"Successfully uploaded M1 {group} to {gcs_path}")


def seed_tourism_dataset():
    """Downloads Tourism data, extracts, and uploads to GCS, using a local cache."""
    gcs_path = f"{GCS_DATASETS_PATH}/tourism/27-3-Athanasopoulos1"
    if gcs_path_exists(gcs_path):
        print("Tourism data already exists in GCS. Skipping.")
        return

    print("Processing Tourism dataset...")

    local_tourism_path = os.path.join(LOCAL_CACHE_PATH, "tourism")
    os.makedirs(local_tourism_path, exist_ok=True)
    local_zip_path = os.path.join(local_tourism_path, "27-3-Athanasopoulos1.zip")

    if not os.path.exists(local_zip_path):
        print(f"Did not find local cache at {local_zip_path}. Downloading...")
        url = "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=60)
                response.raise_for_status()
                with open(local_zip_path, "wb") as f:
                    f.write(response.content)
                print(
                    f"Successfully downloaded and cached tourism data at {local_zip_path}"
                )
                break  # Success
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 == max_retries:
                    raise
                time.sleep(5)
    else:
        print(f"Found local cache for Tourism data at {local_zip_path}.")

    with open(local_zip_path, "rb") as f_zip:
        zip_bytes = BytesIO(f_zip.read())

    with zipfile.ZipFile(zip_bytes) as z:
        for file_info in z.infolist():
            if file_info.is_dir():
                continue
            gcs_file_path = f"{gcs_path}/{os.path.basename(file_info.filename)}"
            with z.open(file_info) as f_in:
                GCS_FS.pipe(gcs_file_path, f_in.read())

    print(f"Successfully uploaded Tourism data to {gcs_path}")


def seed_datasetsforecast_dataset(
    name: str, group: str = None, group_is_name: bool = False
):
    """
    Downloads data using datasetsforecast and uploads the directory to GCS,
    mirroring the local assets/datasets structure.
    """
    # 1. Determine subdirectory and construct GCS path
    sub_dir = ""
    if name in [
        "ECL",
        "ETTh1",
        "ETTh2",
        "ETTm1",
        "ETTm2",
        "Weather",
    ]:
        sub_dir = "longhorizon/datasets"
    elif name in [
        "Labour",
        "Traffic",
        "Wiki2",
        "Tourism",
        "TourismLarge",
        "TourismSmall",
    ]:
        sub_dir = "hierarchical"

    gcs_path = (
        os.path.join(GCS_DATASETS_PATH, sub_dir, name)
        if sub_dir
        else os.path.join(GCS_DATASETS_PATH, name)
    )

    # 2. Check GCS
    if gcs_path_exists(gcs_path):
        print(f"{name} data already exists in GCS at {gcs_path}. Skipping.")
        return

    # 3. Check local cache (this is the final destination, so if it exists, something was downloaded)
    local_source_dir = (
        os.path.join(LOCAL_CACHE_PATH, sub_dir, name)
        if sub_dir
        else os.path.join(LOCAL_CACHE_PATH, name)
    )

    if not os.path.exists(local_source_dir):
        print(
            f"Did not find {name} in local cache. Downloading to {LOCAL_CACHE_PATH}..."
        )
        loader_map = {
            "M3": M3,
            "M4": M4,
            "M5": M5,
            "LongHorizon": LongHorizon,
            "Hierarchical": HierarchicalData,
        }
        loader_key = name
        if sub_dir == "longhorizon":
            loader_key = "LongHorizon"
        if sub_dir == "hierarchical":
            loader_key = "Hierarchical"

        Loader = loader_map[loader_key]
        load_group = group if not group_is_name else name

        # This will download to LOCAL_CACHE_PATH/longhorizon etc.
        if load_group:
            Loader.load(LOCAL_CACHE_PATH, group=load_group)
        else:
            Loader.load(LOCAL_CACHE_PATH)
    else:
        print(f"Found {name} in local cache at {local_source_dir}")

    # 4. Upload from local cache to GCS
    if os.path.exists(local_source_dir):
        print(f"Uploading {name} from {local_source_dir} to {gcs_path}...")
        GCS_FS.put(local_source_dir, gcs_path, recursive=True)
        print(f"Successfully uploaded {name} to {gcs_path}")
    else:
        print(
            f"[ERROR] Could not find downloaded data for {name} at {local_source_dir}"
        )


def seed_m3_dataset(group: str, temp_dir: str):
    """Downloads M3 data, saves as parquet, and uploads to GCS."""
    gcs_path = f"{GCS_DATASETS_PATH}/m3/{group.lower()}.parquet"
    if gcs_path_exists(gcs_path):
        print(f"M3 {group} data already exists in GCS. Skipping.")
        return

    print(f"Processing M3 {group} dataset...")
    df, *_ = M3.load(temp_dir, group=group)

    local_path = os.path.join(temp_dir, f"m3_{group.lower()}.parquet")
    df.to_parquet(local_path)
    GCS_FS.put(local_path, gcs_path)
    print(f"Successfully uploaded M3 {group} to {gcs_path}")


def seed_m4_dataset(group: str, temp_dir: str):
    """Downloads M4 data, saves as parquet, and uploads to GCS."""
    gcs_path = f"{GCS_DATASETS_PATH}/m4/{group.lower()}.parquet"
    if gcs_path_exists(gcs_path):
        print(f"M4 {group} data already exists in GCS. Skipping.")
        return

    print(f"Processing M4 {group} dataset...")
    df, *_ = M4.load(temp_dir, group=group)
    df["ds"] = df["ds"].astype(int)

    if group == "Quarterly":
        df = df.query('unique_id!="Q23425"').reset_index(drop=True)

    unq_periods = df["ds"].sort_values().unique()

    freq_pd = {"Quarterly": "Q", "Monthly": "M"}
    dates = pd.date_range(
        end="2024-03-01", periods=len(unq_periods), freq=freq_pd[group]
    )

    new_ds = {k: v for k, v in zip(unq_periods, dates)}

    df["ds"] = df["ds"].map(new_ds)

    local_path = os.path.join(temp_dir, f"m4_{group.lower()}.parquet")
    df.to_parquet(local_path)
    GCS_FS.put(local_path, gcs_path)
    print(f"Successfully uploaded M4 {group} to {gcs_path}")


def seed_m5_dataset(temp_dir: str):
    """Downloads M5 data, saves as parquet, and uploads to GCS."""
    gcs_path = f"{GCS_DATASETS_PATH}/m5/daily.parquet"
    if gcs_path_exists(gcs_path):
        print(f"M5 data already exists in GCS. Skipping.")
        return

    print(f"Processing M5 dataset...")
    df, *_ = M5.load(temp_dir)

    local_path = os.path.join(temp_dir, "m5_daily.parquet")
    df.to_parquet(local_path)
    GCS_FS.put(local_path, gcs_path)
    print(f"Successfully uploaded M5 to {gcs_path}")


def main():
    """Main function to seed all datasets."""
    parser = argparse.ArgumentParser(description="Seed datasets in GCS.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specify a single dataset to seed. If not provided, all datasets will be seeded.",
        choices=[
            "M1",
            "M3",
            "M4",
            "M5",
            "Tourism",
            "ECL",
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Labour",
            "Traffic",
            "TourismLarge",
            "TourismSmall",
            "Weather",
            "Wiki2",
        ],
    )
    args = parser.parse_args()

    print("--- Starting GCS Dataset Seeding ---")

    os.makedirs(LOCAL_CACHE_PATH, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        if args.dataset is None or args.dataset == "M1":
            seed_m1_dataset("Quarterly", temp_dir)
            seed_m1_dataset("Monthly", temp_dir)
        if args.dataset is None or args.dataset == "M3":
            seed_m3_dataset("Monthly", temp_dir)
            seed_m3_dataset("Quarterly", temp_dir)
            seed_m3_dataset("Yearly", temp_dir)
        if args.dataset is None or args.dataset == "M4":
            seed_m4_dataset("Monthly", temp_dir)
            seed_m4_dataset("Quarterly", temp_dir)
        if args.dataset is None or args.dataset == "M5":
            seed_m5_dataset(temp_dir)

    if args.dataset is None or args.dataset == "Tourism":
        seed_tourism_dataset()

    if args.dataset is None or args.dataset == "M3":
        seed_datasetsforecast_dataset("M3", group="Monthly")
    if args.dataset is None or args.dataset == "M4":
        seed_datasetsforecast_dataset("M4", group="Monthly")
    if args.dataset is None or args.dataset == "M5":
        seed_datasetsforecast_dataset("M5")
    if args.dataset is None or args.dataset == "ECL":
        seed_datasetsforecast_dataset("ECL", group_is_name=True)
    if args.dataset is None or args.dataset == "ETTh1":
        seed_datasetsforecast_dataset("ETTh1", group_is_name=True)
    if args.dataset is None or args.dataset == "ETTh2":
        seed_datasetsforecast_dataset("ETTh2", group_is_name=True)
    if args.dataset is None or args.dataset == "ETTm1":
        seed_datasetsforecast_dataset("ETTm1", group_is_name=True)
    if args.dataset is None or args.dataset == "ETTm2":
        seed_datasetsforecast_dataset("ETTm2", group_is_name=True)
    if args.dataset is None or args.dataset == "Labour":
        seed_datasetsforecast_dataset("Labour", group_is_name=True)
    if args.dataset is None or args.dataset == "Traffic":
        seed_datasetsforecast_dataset("Traffic", group_is_name=True)
    if args.dataset is None or args.dataset == "Tourism":
        seed_datasetsforecast_dataset("Tourism", group_is_name=True)
    if args.dataset is None or args.dataset == "TourismLarge":
        seed_datasetsforecast_dataset("TourismLarge", group_is_name=True)
    if args.dataset is None or args.dataset == "TourismSmall":
        seed_datasetsforecast_dataset("TourismSmall", group_is_name=True)
    if args.dataset is None or args.dataset == "Weather":
        seed_datasetsforecast_dataset("Weather", group_is_name=True)
    if args.dataset is None or args.dataset == "Wiki2":
        seed_datasetsforecast_dataset("Wiki2", group_is_name=True)

    print("\n--- GCS Dataset Seeding Complete ---")
    print(
        f"All datasets should now be available under gs://{os.environ.get('GCS_BUCKET', 'your-gcs-bucket-name')}/forts-experiments/datasets/"
    )


if __name__ == "__main__":
    import argparse

    main()
