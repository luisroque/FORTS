import argparse
import os

import pandas as pd

from forts.gcs_utils import gcs_path_exists, get_gcs_path


def _pad_for_unsupported_models(
    df: pd.DataFrame, freq: str, required_length: int
) -> pd.DataFrame:
    """
    Manually pads the start of each time series for models that
    do not support start_padding_enabled, only if the series
    is shorter than the required_length.
    """
    if df.empty:
        return df

    padded_dfs = []
    for uid, group in df.groupby("unique_id"):
        group = group.sort_values("ds")
        current_length = len(group)

        if current_length < required_length:
            padding_size = required_length - current_length
            first_ds = group["ds"].iloc[0]
            first_y = group["y"].iloc[0]

            pad_dates = pd.date_range(
                end=first_ds, periods=padding_size + 1, freq=freq
            )[:-1]

            if not pad_dates.empty:
                pad_df = pd.DataFrame({"unique_id": uid, "ds": pad_dates, "y": first_y})
                padded_dfs.append(pd.concat([pad_df, group], ignore_index=True))
            else:
                padded_dfs.append(group)
        else:
            padded_dfs.append(group)

    return pd.concat(padded_dfs, ignore_index=True)


def get_model_list(model_name=None):
    """
    Returns the list of models that are used in the experiments.
    If model_name is specified, it returns a list containing only that model.
    """
    # Avoid circular import
    from forts.model_pipeline.model_pipeline import _ModelListMixin

    model_list = _ModelListMixin().get_model_list()

    if model_name:
        model_list = [m for m in model_list if m[0] == model_name]
        if not model_list:
            raise ValueError(f"Model '{model_name}' not found in the model list.")

    return model_list


def check_results_exist(
    dataset,
    dataset_group,
    model_name,
    horizon,
    mode,
    finetune,
    dataset_source=None,
    dataset_group_source=None,
    test_mode=False,
):
    if test_mode:
        results_folder = get_gcs_path("results/test_results")
    elif finetune:
        results_folder = get_gcs_path("results/results_forecast_fine_tuning")
    elif mode == "out_domain" and dataset_source == "MIXED":
        results_folder = get_gcs_path("results/results_forecast_coreset")
    else:
        results_folder = get_gcs_path(f"results/results_forecast_{mode}")

    if mode == "basic_forecasting":
        results_file = (
            f"{results_folder}/{dataset}_{dataset_group}_{model_name}_{horizon}.json"
        )
    elif dataset_source:
        if finetune:
            results_file = (
                f"{results_folder}/{dataset}_{dataset_group}_{model_name}_{horizon}_"
                f"trained_on_{dataset_source}_{dataset_group_source}_finetuning.json"
            )
        else:
            results_file = (
                f"{results_folder}/{dataset}_{dataset_group}_{model_name}_{horizon}_"
                f"trained_on_{dataset_source}_{dataset_group_source}.json"
            )
    else:
        results_file = (
            f"{results_folder}/{dataset}_{dataset_group}_{model_name}_{horizon}.json"
        )
    return gcs_path_exists(results_file), results_file


def extract_frequency(dataset_group):
    """Safely extracts frequency from dataset group."""
    freq = dataset_group[1]["FREQ"]
    return freq


def extract_horizon(dataset_group):
    """Safely extracts horizon from dataset group."""
    h = dataset_group[1]["H"]
    return h


def extract_score(dataset_group):
    """Safely extracts frequency from dataset group."""
    score = dataset_group[1]["final_score"]
    return score


def has_final_score_in_tuple(tpl):
    """Check if the second element is a dictionary and contains 'final_score'"""
    return isinstance(tpl[1], dict) and "final_score" in tpl[1]


def set_device(use_gpu: bool):
    """
    Configures whether PyTorch can see the GPU.
    """
    if not use_gpu:
        print("Forcing CPU usage (GPU disabled by user).")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        print("Using GPU if available")


def cmd_parser():
    parser = argparse.ArgumentParser(
        description="Run synthetic data generation using FORTS."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration",
    )
    parser.add_argument(
        "--transfer-learning",
        action="store_true",
        help="Perform transfer learning (TL).",
    )
    parser.add_argument(
        "--coreset",
        action="store_true",
        help="Perform transfer learning with coreset dataset "
        "and a leave-one-out strategy.",
    )
    parser.add_argument(
        "--basic-forecasting",
        action="store_true",
        help="Perform basic forecasting.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Enable fine-tuning on the target dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specify a single model to run (e.g., AutoNHITS).",
    )
    args = parser.parse_args()

    return args
