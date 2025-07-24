import re

import pandas as pd

from forts.experiments.run_pipeline import DATASET_GROUP_FREQ
from forts.gcs_utils import gcs_list_files, gcs_read_csv, gcs_read_json, get_gcs_path
from forts.model_pipeline.model_pipeline import _ModelListMixin


def get_experiment_grid():
    experiments = []
    models = [name for name, _ in _ModelListMixin().get_model_list()]

    all_pairs = [(ds, grp) for ds, grps in DATASET_GROUP_FREQ.items() for grp in grps]

    for model in models:
        # Basic forecasting and in-domain
        for dataset, subgroups in DATASET_GROUP_FREQ.items():
            for subgroup in subgroups:
                experiments.append(
                    {
                        "type": "basic_forecasting",
                        "source_dataset": "None",
                        "source_group": "None",
                        "target_dataset": dataset,
                        "target_group": subgroup,
                        "model": model,
                        "finetuned": False,
                    }
                )
                experiments.append(
                    {
                        "type": "in_domain",
                        "source_dataset": "None",
                        "source_group": "None",
                        "target_dataset": dataset,
                        "target_group": subgroup,
                        "model": model,
                        "finetuned": False,
                    }
                )

        # Out-domain
        for source_ds, source_grp in all_pairs:
            for target_ds, target_grp in all_pairs:
                if (source_ds, source_grp) != (target_ds, target_grp):
                    # Standard
                    experiments.append(
                        {
                            "type": "out_domain",
                            "source_dataset": source_ds,
                            "source_group": source_grp,
                            "target_dataset": target_ds,
                            "target_group": target_grp,
                            "model": model,
                            "finetuned": False,
                        }
                    )
                    # Finetuned
                    experiments.append(
                        {
                            "type": "out_domain",
                            "source_dataset": source_ds,
                            "source_group": source_grp,
                            "target_dataset": target_ds,
                            "target_group": target_grp,
                            "model": model,
                            "finetuned": True,
                        }
                    )

        # Coreset (leave-one-out)
        for held_out_ds, held_out_grp in all_pairs:
            # Standard
            experiments.append(
                {
                    "type": "coreset",
                    "source_dataset": "MIXED",
                    "source_group": f"ALL_BUT_{held_out_ds}_{held_out_grp}",
                    "target_dataset": held_out_ds,
                    "target_group": held_out_grp,
                    "model": model,
                    "finetuned": False,
                }
            )
            # Finetuned
            experiments.append(
                {
                    "type": "coreset",
                    "source_dataset": "MIXED",
                    "source_group": f"ALL_BUT_{held_out_ds}_{held_out_grp}",
                    "target_dataset": held_out_ds,
                    "target_group": held_out_grp,
                    "model": model,
                    "finetuned": True,
                }
            )

    return pd.DataFrame(experiments)


def load_summary_results(suffix):
    summary_dir = get_gcs_path("results_forecast_out_domain_summary")
    fname = f"results_all_seasonalities_all_combinations_mase_mean{suffix}.csv"
    fpath = f"{summary_dir}/{fname}"
    try:
        return gcs_read_csv(fpath)
    except FileNotFoundError:
        pass


def load_finetuning_results():
    results = []
    base_path = get_gcs_path("results_forecast_fine_tuning")
    all_files = gcs_list_files(base_path, extension=".json")

    for fpath in all_files:
        fname = fpath.split("/")[-1]
        match = re.match(
            r"(.+?)_(.+?)_(Auto.+?)_(\d+)_trained_on_(.+?)_(.+?)_finetuning.json",
            fname,
        )
        if match:
            (
                target_ds,
                target_grp,
                model,
                _,
                source_ds,
                source_grp,
            ) = match.groups()
            data = gcs_read_json(fpath)
            mase = data.get("Forecast MASE MEAN (last window) Per Series_out_domain")
            results.append(
                {
                    "source_dataset": source_ds,
                    "source_group": source_grp,
                    "target_dataset": target_ds,
                    "target_group": target_grp,
                    "model": model,
                    "MASE Mean": mase,
                }
            )
    return pd.DataFrame(results)


def load_raw_results(mode):
    results = []
    base_path = get_gcs_path(f"results_forecast_{mode}")
    all_files = gcs_list_files(base_path, extension=".json")

    for fpath in all_files:
        fname = fpath.split("/")[-1]
        match = re.match(r"(.+?)_(.+?)_(Auto.+?)_(\d+).json", fname)
        if match:
            (
                target_ds,
                target_grp,
                model,
                _,
            ) = match.groups()
            data = gcs_read_json(fpath)
            mase = data.get(f"Forecast MASE MEAN (last window) Per Series_{mode}")
            results.append(
                {
                    "source_dataset": "None",
                    "source_group": "None",
                    "target_dataset": target_ds,
                    "target_group": target_grp,
                    "model": model,
                    "MASE Mean": mase,
                }
            )
    return pd.DataFrame(results)


def main():
    grid = get_experiment_grid()
    grid["MASE Mean"] = pd.NA
    grid["Num Results"] = 0

    # Load results
    basic_results = load_raw_results("basic_forecasting")
    in_domain_results = load_raw_results("in_domain")
    out_domain_results = load_summary_results("_out_domain")
    coreset_results = load_summary_results("_out_domain_coreset")
    finetuning_results = load_finetuning_results()

    for idx, row in grid.iterrows():
        df = None
        if row["type"] == "basic_forecasting":
            df = basic_results
        elif row["type"] == "in_domain":
            df = in_domain_results
        elif row["type"] == "out_domain" and not row["finetuned"]:
            df = out_domain_results
        elif row["type"] == "out_domain" and row["finetuned"]:
            df = finetuning_results
        elif row["type"] == "coreset" and not row["finetuned"]:
            df = coreset_results
        elif row["type"] == "coreset" and row["finetuned"]:
            # Assuming coreset finetuning results are also in the fine-tuning folder
            df = finetuning_results

        if df is not None:
            mask = (
                (
                    df[
                        (
                            "source_dataset"
                            if "source_dataset" in df.columns
                            else "Dataset Source"
                        )
                    ]
                    == row["source_dataset"]
                )
                & (
                    df[
                        (
                            "source_group"
                            if "source_group" in df.columns
                            else "Dataset Group Source"
                        )
                    ]
                    == row["source_group"]
                )
                & (
                    df[
                        (
                            "target_dataset"
                            if "target_dataset" in df.columns
                            else "Dataset Target"
                        )
                    ]
                    == row["target_dataset"]
                )
                & (
                    df[
                        (
                            "target_group"
                            if "target_group" in df.columns
                            else "Dataset Group Target"
                        )
                    ]
                    == row["target_group"]
                )
                & (df["model" if "model" in df.columns else "Method"] == row["model"])
            )
            result_row = df[mask]
            if not result_row.empty:
                mase_value = result_row["MASE Mean"].mean()
                grid.loc[idx, "MASE Mean"] = mase_value
                grid.loc[idx, "Num Results"] = result_row["MASE Mean"].count()

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        for exp_type in [
            "basic_forecasting",
            "in_domain",
            "out_domain",
            "coreset",
        ]:
            print(f"--- {exp_type.replace('_', ' ').title()} Results ---")
            df_slice = grid[grid["type"] == exp_type].copy()

            if exp_type == "basic_forecasting" or exp_type == "in_domain":
                df_slice.rename(
                    columns={
                        "target_dataset": "Dataset",
                        "target_group": "Group",
                        "model": "Method",
                    },
                    inplace=True,
                )
                print(
                    df_slice[
                        ["Dataset", "Group", "Method", "MASE Mean", "Num Results"]
                    ].to_string(index=False)
                )
            elif exp_type == "out_domain":
                df_slice.rename(
                    columns={
                        "source_dataset": "Source Dataset",
                        "source_group": "Source Group",
                        "model": "Method",
                    },
                    inplace=True,
                )
                summary_df = (
                    df_slice.groupby(
                        ["Source Dataset", "Source Group", "Method", "finetuned"]
                    )[["MASE Mean", "Num Results"]]
                    .agg({"MASE Mean": "mean", "Num Results": "sum"})
                    .reset_index()
                )
                print(summary_df.to_string(index=False))

            elif exp_type == "coreset":
                df_slice.rename(
                    columns={
                        "target_dataset": "Held-out Dataset",
                        "target_group": "Held-out Group",
                        "model": "Method",
                    },
                    inplace=True,
                )
                print(
                    df_slice[
                        [
                            "Held-out Dataset",
                            "Held-out Group",
                            "Method",
                            "finetuned",
                            "MASE Mean",
                            "Num Results",
                        ]
                    ].to_string(index=False)
                )
            print("\n" + "=" * 50 + "\n")

    print("--- Overall Experiment Progress ---")
    grid["Completed"] = grid["Num Results"] > 0
    progress_summary = (
        grid.groupby(["type", "finetuned"])
        .agg(Completed=("Completed", "sum"), Total=("Completed", "size"))
        .reset_index()
    )
    progress_summary["Progress"] = (
        progress_summary["Completed"] / progress_summary["Total"] * 100
    ).map("{:.2f}%".format)
    progress_summary.rename(
        columns={"type": "Experiment Type", "finetuned": "Finetuned"},
        inplace=True,
    )
    print(progress_summary.to_string(index=False))


if __name__ == "__main__":
    main()
