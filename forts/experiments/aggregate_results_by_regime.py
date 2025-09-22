import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from forts.gcs_utils import gcs_read_csv, gcs_write_csv, get_gcs_path


def load_granular_summaries(summary_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all granular summary files from a directory.

    Args:
        summary_dir: Directory containing the granular summary files

    Returns:
        Dictionary mapping regime names to their corresponding DataFrames
    """
    granular_files = {
        "full_shot": "full_shot_granular_summary.csv",
        "in_domain": "in_domain_granular_summary.csv",
        "single_source": "single_source_granular_summary.csv",
        "multi_source": "multi_source_granular_summary.csv",
    }

    summaries = {}

    for regime, filename in granular_files.items():
        file_path = f"{summary_dir}/{filename}"
        try:
            df = gcs_read_csv(file_path)
            if not df.empty:
                summaries[regime] = df
                print(
                    f"Loaded {regime}: {df.shape[0]} methods × {df.shape[1]-1} datasets"
                )
            else:
                print(f"Warning: Empty dataframe for {regime}")
        except Exception as e:
            print(f"Warning: Could not load {regime} from {file_path}: {e}")

    return summaries


def compute_regime_statistics(summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute aggregate statistics for each regime.

    Args:
        summaries: Dictionary mapping regime names to DataFrames

    Returns:
        DataFrame with regime-level statistics
    """
    regime_stats = []

    for regime_name, df in summaries.items():
        if df.empty:
            continue

        # Get numeric columns (exclude Method column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Compute statistics across all datasets for each method
        method_means = df[numeric_cols].mean(
            axis=1
        )  # Mean across datasets for each method

        # Compute regime-level statistics
        stats = {
            "Regime": regime_name,
            "Num_Methods": len(df),
            "Num_Datasets": len(numeric_cols),
            "Overall_Mean": method_means.mean(),
            "Overall_Std": method_means.std(),
            "Overall_Median": method_means.median(),
            "Best_Method_Score": method_means.min(),
            "Best_Method": df.loc[method_means.idxmin(), "Method"],
            "Worst_Method_Score": method_means.max(),
            "Worst_Method": df.loc[method_means.idxmax(), "Method"],
            "Score_Range": method_means.max() - method_means.min(),
        }

        regime_stats.append(stats)

    return pd.DataFrame(regime_stats)


def compute_method_rankings_across_regimes(
    summaries: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute method rankings across all regimes.

    Args:
        summaries: Dictionary mapping regime names to DataFrames

    Returns:
        DataFrame with method performance across regimes
    """
    all_methods = set()
    for df in summaries.values():
        if not df.empty:
            all_methods.update(df["Method"].tolist())

    method_performance = []

    for method in sorted(all_methods):
        perf = {"Method": method}

        for regime_name, df in summaries.items():
            if df.empty or method not in df["Method"].values:
                perf[f"{regime_name}_mean"] = np.nan
                perf[f"{regime_name}_rank"] = np.nan
            else:
                method_row = df[df["Method"] == method]
                numeric_cols = method_row.select_dtypes(include=[np.number]).columns
                method_mean = method_row[numeric_cols].mean(axis=1).iloc[0]

                # Compute rank within regime (1 = best)
                regime_means = df[numeric_cols].mean(axis=1)
                method_rank = (regime_means <= method_mean).sum()

                perf[f"{regime_name}_mean"] = round(method_mean, 3)
                perf[f"{regime_name}_rank"] = method_rank

        # Compute overall statistics
        mean_cols = [col for col in perf.keys() if col.endswith("_mean")]
        means = [perf[col] for col in mean_cols if not pd.isna(perf[col])]

        if means:
            perf["Overall_Mean"] = round(np.mean(means), 3)
            perf["Overall_Std"] = round(np.std(means), 3)
            perf["Regimes_Participated"] = len(means)
        else:
            perf["Overall_Mean"] = np.nan
            perf["Overall_Std"] = np.nan
            perf["Regimes_Participated"] = 0

        method_performance.append(perf)

    df_performance = pd.DataFrame(method_performance)

    # Sort by overall mean (ascending, better methods first)
    # Handle NaN values by putting them at the end manually
    df_performance = df_performance.sort_values(
        "Overall_Mean", ascending=True
    ).reset_index(drop=True)

    return df_performance


def compute_dataset_difficulty_analysis(
    summaries: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Analyze dataset difficulty across regimes.

    Args:
        summaries: Dictionary mapping regime names to DataFrames

    Returns:
        DataFrame with dataset difficulty analysis
    """
    # Get all datasets across all regimes
    all_datasets = set()
    for df in summaries.values():
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            all_datasets.update(numeric_cols.tolist())

    dataset_analysis = []

    for dataset in sorted(all_datasets):
        analysis = {"Dataset": dataset}

        for regime_name, df in summaries.items():
            if df.empty or dataset not in df.columns:
                analysis[f"{regime_name}_mean"] = np.nan
                analysis[f"{regime_name}_std"] = np.nan
                analysis[f"{regime_name}_methods"] = 0
            else:
                dataset_scores = df[dataset].dropna()
                analysis[f"{regime_name}_mean"] = (
                    round(dataset_scores.mean(), 3)
                    if not dataset_scores.empty
                    else np.nan
                )
                analysis[f"{regime_name}_std"] = (
                    round(dataset_scores.std(), 3)
                    if not dataset_scores.empty
                    else np.nan
                )
                analysis[f"{regime_name}_methods"] = len(dataset_scores)

        # Compute overall statistics
        mean_cols = [col for col in analysis.keys() if col.endswith("_mean")]
        means = [analysis[col] for col in mean_cols if not pd.isna(analysis[col])]

        if means:
            analysis["Overall_Mean"] = round(np.mean(means), 3)
            analysis["Overall_Std"] = round(np.std(means), 3)
            analysis["Regimes_Available"] = len(means)
        else:
            analysis["Overall_Mean"] = np.nan
            analysis["Overall_Std"] = np.nan
            analysis["Regimes_Available"] = 0

        dataset_analysis.append(analysis)

    df_analysis = pd.DataFrame(dataset_analysis)

    # Sort by overall mean (ascending, easier datasets first)
    # Handle NaN values by putting them at the end manually
    df_analysis = df_analysis.sort_values("Overall_Mean", ascending=True).reset_index(
        drop=True
    )

    return df_analysis


def save_results(
    regime_stats: pd.DataFrame,
    method_performance: pd.DataFrame,
    dataset_analysis: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Save all aggregate results to CSV files.

    Args:
        regime_stats: Regime-level statistics
        method_performance: Method performance across regimes
        dataset_analysis: Dataset difficulty analysis
        output_dir: Directory to save results
    """
    results = {
        "regime_statistics.csv": regime_stats,
        "method_performance_across_regimes.csv": method_performance,
        "dataset_difficulty_analysis.csv": dataset_analysis,
    }

    for filename, df in results.items():
        if not df.empty:
            # Save to GCS
            gcs_path = f"{output_dir}/{filename}"
            gcs_write_csv(df, gcs_path)

            # Save locally
            local_dir = output_dir.replace("gs://", "assets/results_cache/")
            os.makedirs(local_dir, exist_ok=True)
            local_path = f"{local_dir}/{filename}"
            df.to_csv(local_path, index=False)

            print(f"Saved {filename}: {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            print(f"Warning: Empty dataframe for {filename}")


def main(summary_dir: Optional[str] = None, output_dir: Optional[str] = None):
    """
    Main function to aggregate results by regime.

    Args:
        summary_dir: Directory containing granular summary files
        output_dir: Directory to save aggregate results
    """
    # Use default paths if not provided
    if summary_dir is None:
        summary_dir = get_gcs_path("results/results_summary")

    if output_dir is None:
        output_dir = summary_dir

    print(f"Loading granular summaries from: {summary_dir}")
    print(f"Saving aggregate results to: {output_dir}")
    print("=" * 60)

    # Load granular summaries
    summaries = load_granular_summaries(summary_dir)

    if not summaries:
        print("Error: No granular summary files found!")
        return

    print("\nComputing regime statistics...")
    regime_stats = compute_regime_statistics(summaries)

    print("\nComputing method performance across regimes...")
    method_performance = compute_method_rankings_across_regimes(summaries)

    print("\nComputing dataset difficulty analysis...")
    dataset_analysis = compute_dataset_difficulty_analysis(summaries)

    print("\nSaving results...")
    save_results(regime_stats, method_performance, dataset_analysis, output_dir)

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"- Processed {len(summaries)} regimes")
    print(f"- Analyzed {len(method_performance)} methods")
    print(f"- Analyzed {len(dataset_analysis)} datasets")
    print("- Generated 3 aggregate summary files")
    print("=" * 60)

    return {
        "regime_stats": regime_stats,
        "method_performance": method_performance,
        "dataset_analysis": dataset_analysis,
    }


if __name__ == "__main__":
    main()
