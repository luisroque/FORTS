import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pandas as pd
from tqdm import tqdm

from forts.gcs_utils import gcs_list_files, gcs_read_csv, gcs_write_csv, get_gcs_path

LOCAL_CACHE_DIR = "assets/results_cache"


def get_local_cache_path(gcs_path: str) -> str:
    """Converts a GCS path to a local cache path."""
    if not gcs_path.startswith("gs://"):
        return gcs_path
    relative_path = gcs_path[5:]  # Remove "gs://"
    return os.path.join(LOCAL_CACHE_DIR, relative_path)


def get_cached_file_list(
    base_path: str, cache_duration_seconds: int = 3600
) -> List[str]:
    """Gets a list of files from GCS, using a local cache for the file list itself."""
    list_cache_dir = os.path.join(LOCAL_CACHE_DIR, "_file_lists")
    os.makedirs(list_cache_dir, exist_ok=True)

    sanitized_path = base_path.replace("gs://", "").replace("/", "_")
    list_cache_file = os.path.join(list_cache_dir, f"{sanitized_path}.json")

    if os.path.exists(list_cache_file):
        try:
            cache_age = time.time() - os.path.getmtime(list_cache_file)
            if cache_age < cache_duration_seconds:
                with open(list_cache_file, "r") as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass  # Fall through to refetch

    files = gcs_list_files(base_path, extension=".csv")
    with open(list_cache_file, "w") as f:
        json.dump(files, f)
    return files


def read_csv_with_cache(gcs_path: str) -> pd.DataFrame:
    """Reads a CSV from GCS, using a local cache."""
    local_path = get_local_cache_path(gcs_path)
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception:
            # Fall through to re-download if local file is corrupt
            pass  # nosec

    df = gcs_read_csv(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)
    return df


def main():
    """Main function to process efficiency results."""
    base_paths = [
        get_gcs_path("model_weights/coreset/hypertuning"),
        get_gcs_path("model_weights/basic_forecasting/hypertuning_basic_forecasting"),
        get_gcs_path("model_weights/in_domain/hypertuning"),
        get_gcs_path("model_weights/out_domain/hypertuning"),
    ]
    results_out_dir = get_gcs_path("results/results_summary")

    performance_files = []
    for base_path in base_paths:
        print(f"Searching for performance files in: {base_path}")
        files = get_cached_file_list(base_path)
        performance_files.extend(files)
        print(f"Found {len(files)} performance files in {base_path}.")

    print(f"Found {len(performance_files)} total performance files.")

    results_combined = []

    with ThreadPoolExecutor() as executor:
        # Use a wrapper to add file metadata during processing
        def process_file(gcs_path):
            file_name = os.path.basename(gcs_path)
            df = read_csv_with_cache(gcs_path)
            parts = file_name.replace("_results.csv", "").split("_")

            # Extract method from the end of the filename
            if "coreset" in gcs_path:
                if "TimeGEN" in parts:
                    timegen_idx = parts.index("TimeGEN")
                    method = "_".join(parts[timegen_idx:])
                else:
                    method = parts[-1]
                dataset = "MIXED"
                group = "MIXED"
            else:
                if "TimeGEN" in parts:
                    timegen_idx = parts.index("TimeGEN")
                    method = "_".join(parts[timegen_idx:])
                else:
                    method = parts[-1]
                dataset = parts[0]
                group = parts[1]

            df["Method"] = method
            df["Dataset"] = dataset
            df["Group"] = group
            df["gcs_path"] = gcs_path

            return df[
                ["Dataset", "Group", "Method", "time_total_s", "loss", "gcs_path"]
            ].dropna()

        results_list = list(
            tqdm(
                executor.map(process_file, performance_files),
                total=len(performance_files),
                desc="Processing efficiency files",
            )
        )
        results_combined = [r for r in results_list if r is not None]

    if not results_combined:
        print("No valid performance results found.")
        return

    full_df = pd.concat(results_combined, ignore_index=True)
    idx = full_df.groupby(["Dataset", "Group", "Method"])["loss"].idxmin()
    min_loss_df = full_df.loc[idx]

    # Add the number of files used in the computation for each method
    file_counts = (
        full_df.groupby("Method")["gcs_path"]
        .nunique()
        .reset_index(name="Number of Files")
    )

    results_df = min_loss_df.groupby("Method")["time_total_s"].sum().reset_index()
    results_df = pd.merge(results_df, file_counts, on="Method")
    results_df = results_df.sort_values(by="Method").reset_index(drop=True)

    # Normalize by AutoTimeGEN
    autotimegen_row = results_df[results_df["Method"] == "AutoTimeGEN"]
    if not autotimegen_row.empty:
        autotimegen_time = autotimegen_row["time_total_s"].iloc[0]
        results_df["Normalized Time (vs AutoTimeGEN)"] = (
            results_df["time_total_s"] / autotimegen_time
        ).round(3)
    else:
        # Fallback to normalizing by the fastest algorithm if AutoTimeGEN is not found
        min_time = results_df["time_total_s"].min()
        results_df["Normalized Time (vs Fastest)"] = (
            results_df["time_total_s"] / min_time
        ).round(3)

    # Save to GCS
    gcs_csv_path = f"{results_out_dir}/training_time_stats_per_method.csv"
    gcs_write_csv(results_df, gcs_csv_path)

    # Save locally
    local_csv_path = get_local_cache_path(gcs_csv_path)
    os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
    results_df.to_csv(local_csv_path, index=False)

    print(f"Efficiency results saved to {gcs_csv_path} and {local_csv_path}")


if __name__ == "__main__":
    main()
