import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from forts.gcs_utils import (
    gcs_list_files,
    gcs_read_csv,
    gcs_read_json,
    gcs_write_csv,
    get_gcs_path,
)

BASE_OUT_DOMAIN = get_gcs_path("results/results_forecast_out_domain")
BASE_IN_DOMAIN = get_gcs_path("results/results_forecast_in_domain")
BASE_BASIC_FORECASTING = get_gcs_path("results/results_forecast_basic_forecasting")
BASE_CORESET = get_gcs_path("results/results_forecast_coreset")
SUMMARY_DIR = get_gcs_path("results/results_summary")
FM_RESULTS_DIR = get_gcs_path("results/results_forecast_FM")
LOCAL_CACHE_DIR = "assets/results_cache"


FM_PATHS = {
    "moirai": f"{FM_RESULTS_DIR}/moirai_results.csv",
    "timemoe": f"{FM_RESULTS_DIR}/timemoe_results.csv",
}


def get_local_cache_path(gcs_path: str) -> str:
    """Converts a GCS path to a local cache path."""
    if not gcs_path.startswith("gs://"):
        return gcs_path
    relative_path = gcs_path[5:]  # Remove "gs://"
    return os.path.join(LOCAL_CACHE_DIR, relative_path)


def load_json_files(base_path: str, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Load all JSON files from a GCS directory into a DataFrame, using a local cache.
    """
    files = gcs_list_files(base_path, extension=".json")

    if max_files is not None:
        files = files[:max_files]

    def process_file(gcs_file_path: str):
        local_path = get_local_cache_path(gcs_file_path)

        # If file is cached, read from local
        if os.path.exists(local_path):
            try:
                with open(local_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading local file {local_path}, will refetch: {e}")
                # Fall through to re-download if local file is corrupt

        # If not cached or local read failed, download from GCS, save, then return content
        try:
            data = gcs_read_json(gcs_file_path)
            if data is not None:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "w") as f:
                    json.dump(data, f)
            return data
        except Exception as e:
            print(f"Error processing GCS file {gcs_file_path}: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_file, files),
                total=len(files),
                desc=f"Processing files in {os.path.basename(base_path)}",
            )
        )

    data = [item for item in results if item is not None]

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df


def rename_columns(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """
    Rename columns for out-domain or in-domain data to unified format.
    """
    metric_suffix = f"Per Series_{domain}"
    rename_map = {
        f"Forecast SMAPE MEAN (last window) {metric_suffix}": "SMAPE Mean",
        f"Forecast MASE MEAN (last window) {metric_suffix}": "MASE Mean",
        f"Forecast MAE MEAN (last window) {metric_suffix}": "MAE Mean",
        f"Forecast RMSE MEAN (last window) {metric_suffix}": "RMSE Mean",
        f"Forecast RMSSE MEAN (last window) {metric_suffix}": "RMSSE Mean",
        "Dataset": "Dataset Target",
        "Group": "Dataset Group Target",
    }
    return df.rename(columns=rename_map)


def align_columns(df: pd.DataFrame, reference_cols: List[str]) -> pd.DataFrame:
    """
    Align the DataFrame columns to a reference column list.
    Missing columns will be added with NA values.
    """
    for col in reference_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[reference_cols]


def add_source_target_pair_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column for identifying source-target dataset pairs.
    """
    return df.assign(
        **{
            "Source-Target Pair": df["Dataset Source"]
            + " ("
            + df["Dataset Group Source"]
            + ") → "
            + df["Dataset Target"]
            + " ("
            + df["Dataset Group Target"]
            + ")"
        }
    )


def load_fm_result(path_fm: str, reference_cols: List[str]) -> pd.DataFrame:
    """
    Load and process a single forecast model (FM) result file.
    """
    try:
        fm_df = gcs_read_csv(path_fm)
    except FileNotFoundError:
        print(f"Info: FM results file not found at {path_fm}. Skipping.")
        return pd.DataFrame()

    for col in reference_cols:
        if col not in fm_df.columns:
            fm_df[col] = pd.NA

    fm_df["Dataset Source"] = "MIXED"
    fm_df["Dataset Group Source"] = fm_df.apply(
        lambda x: f"ALL_BUT_{x['Dataset Target']}_{x['Dataset Group Target']}",
        axis=1,
    )
    return fm_df.reindex(columns=reference_cols)


def summarize_metric(
    df: pd.DataFrame,
    metric: str,
    mode: str,
    aggregate_by: List[str],
    rank_within: Optional[List[str]] = None,
    filter_same_seasonality: bool = False,
    src_seas_col: str = "Dataset Group Source",
    tgt_seas_col: str = "Dataset Group Target",
    out_path: Optional[str] = None,
    fname: Optional[str] = None,
    rank_method: str = "min",
    agg_func=np.nanmean,
) -> pd.DataFrame:
    """
    Summarize a metric either by rank or mean.
    """
    work = df.copy()

    if filter_same_seasonality:
        work = work[work[src_seas_col] == work[tgt_seas_col]]

    if mode == "rank":
        if not rank_within:
            raise ValueError("`rank_within` must be given when mode='rank'")
        work["Rank"] = work.groupby(rank_within)[metric].rank(method=rank_method)
        summary = work.groupby(aggregate_by)["Rank"].apply(agg_func).reset_index()
        summary.rename(columns={"Rank": "Rank"}, inplace=True)
        sort_by = ["Rank"] if aggregate_by == ["Method"] else aggregate_by + ["Rank"]
    elif mode == "mean":
        summary = work.groupby(aggregate_by)[metric].apply(agg_func).reset_index()
        summary.rename(columns={metric: metric}, inplace=True)
        sort_by = [metric] if aggregate_by == ["Method"] else aggregate_by + [metric]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    summary.sort_values(by=sort_by, inplace=True)

    if out_path:
        if fname is None:
            stem = "_".join(aggregate_by)
            fname = f"{mode}_{metric.replace(' ','_').lower()}_{stem}.csv"
        gcs_write_csv(summary, f"{out_path}/{fname}")

    return summary


def generate_latex_summary(
    basic_df: pd.DataFrame,
    in_domain_df: pd.DataFrame,
    single_source_df: pd.DataFrame,
    multi_source_df: pd.DataFrame,
    metric: str,
    out_path: str,
) -> pd.DataFrame:
    """
    Generates a summary table for LaTeX.
    """
    dfs = [basic_df, in_domain_df, single_source_df, multi_source_df]
    if not any(not df.empty for df in dfs):
        print("Info: No data available to generate LaTeX summary.")
        return pd.DataFrame()

    all_methods = pd.concat(
        [
            df[["Method"]]
            for df in [basic_df, in_domain_df, single_source_df, multi_source_df]
            if not df.empty
        ]
    ).drop_duplicates()

    scenarios = {
        "Full-shot": basic_df,
        "In-domain": in_domain_df,
        "Single-source": single_source_df,
        "Multi-source": multi_source_df,
    }

    if all_methods.empty:
        print("Info: No methods found to generate LaTeX summary.")
        return pd.DataFrame()

    final_summary = all_methods
    count_summary = all_methods.copy()

    for name, df in scenarios.items():
        if df.empty:
            mase = pd.DataFrame({"Method": [], f"{name}_MASE": []})
            rank = pd.DataFrame({"Method": [], f"{name}_Rank": []})
            counts = pd.DataFrame({"Method": [], f"{name}_Count": []})
        else:
            mase = summarize_metric(df, metric, "mean", aggregate_by=["Method"]).rename(
                columns={metric: f"{name}_MASE"}
            )
            rank = summarize_metric(
                df,
                metric,
                "rank",
                aggregate_by=["Method"],
                rank_within=["Source-Target Pair"],
            ).rename(columns={"Rank": f"{name}_Rank"})
            counts = df.groupby("Method").size().reset_index(name=f"{name}_Count")

        summary = pd.merge(mase, rank, on="Method", how="outer")
        final_summary = pd.merge(final_summary, summary, on="Method", how="left")
        count_summary = pd.merge(count_summary, counts, on="Method", how="left")

    final_summary = final_summary.sort_values(by="Method").reset_index(drop=True)
    count_summary = count_summary.sort_values(by="Method").reset_index(drop=True)

    gcs_write_csv(final_summary, f"{out_path}/latex_summary.csv")
    gcs_write_csv(count_summary, f"{out_path}/latex_count_summary.csv")

    return final_summary


def main(max_files: Optional[int] = None, output_path: Optional[str] = None):
    print("Loading data from GCS directories...")

    out_df = load_json_files(BASE_OUT_DOMAIN, max_files)
    print(f"Out-domain results: {len(out_df)} rows loaded")

    in_df = load_json_files(BASE_IN_DOMAIN, max_files)
    print(f"In-domain results: {len(in_df)} rows loaded")

    basic_for_df = load_json_files(BASE_BASIC_FORECASTING, max_files)
    print(f"Basic forecasting results: {len(basic_for_df)} rows loaded")

    coreset_df = load_json_files(BASE_CORESET, max_files)
    print(f"Coreset results: {len(coreset_df)} rows loaded")

    if "Dataset Source" not in out_df.columns:
        out_df["Dataset Source"] = "None"
    if "Dataset" not in out_df.columns:
        out_df["Dataset"] = "None"

    out_df = out_df[out_df["Dataset Source"] != out_df["Dataset"]]
    out_df = out_df[out_df["Dataset Source"] != "MIXED"]

    out_df = rename_columns(out_df, "out_domain")
    in_df = rename_columns(in_df, "in_domain")
    basic_for_df = rename_columns(basic_for_df, "basic_forecasting")
    coreset_df = rename_columns(coreset_df, "out_domain")

    common_cols = [
        "Dataset Source",
        "Dataset Group Source",
        "Dataset Target",
        "Dataset Group Target",
        "Method",
        "SMAPE Mean",
        "MASE Mean",
        "MAE Mean",
        "RMSE Mean",
        "RMSSE Mean",
    ]

    in_df["Dataset Source"] = "None"
    in_df["Dataset Group Source"] = "None"
    in_df = align_columns(in_df, common_cols)
    basic_for_df["Dataset Source"] = "None"
    basic_for_df["Dataset Group Source"] = "None"
    basic_for_df = align_columns(basic_for_df, common_cols)
    out_df = align_columns(out_df, common_cols)
    coreset_df = align_columns(coreset_df, common_cols)

    moirai_df = load_fm_result(FM_PATHS["moirai"], common_cols)
    timemoe_df = load_fm_result(FM_PATHS["timemoe"], common_cols)
    multi_source_df = pd.concat([coreset_df, moirai_df, timemoe_df], ignore_index=True)

    print(f"Moirai results: {len(moirai_df)} rows")
    print(f"TimeMOE results: {len(timemoe_df)} rows")
    print(f"Multi-source combined: {len(multi_source_df)} rows")

    out_df = add_source_target_pair_column(out_df)
    in_df = add_source_target_pair_column(in_df)
    basic_for_df = add_source_target_pair_column(basic_for_df)
    multi_source_df = add_source_target_pair_column(multi_source_df)

    # summarize results
    metric = "MASE Mean"

    # Use provided output path or default
    final_output_path = output_path if output_path is not None else SUMMARY_DIR

    generate_latex_summary(
        basic_df=basic_for_df,
        in_domain_df=in_df,
        single_source_df=out_df,
        multi_source_df=multi_source_df,
        metric=metric,
        out_path=final_output_path,
    )


if __name__ == "__main__":
    main()
