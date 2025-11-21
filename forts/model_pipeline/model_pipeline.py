import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from neuralforecast.auto import (
    AutoiTransformer,
    AutoKAN,
    AutoNBEATS,
    AutoNHITS,
    AutoPatchTST,
    AutoTFT,
    AutoTimeMixer,
    AutoTSMixer,
    AutoxLSTM,
)
from neuralforecast.models.timemixer import TimeMixer
from neuralforecast.models.xlstm import xLSTM
from ray import tune

from forts.config import MAX_EVALS
from forts.experiments.helper import _pad_for_unsupported_models
from forts.gcs_utils import (
    _get_local_fallback_path,
    gcs_write_csv,
    get_model_weights_path,
)
from forts.model_pipeline.auto.AutoModels import AutoTimeMOE
from forts.model_pipeline.core.core_extension import CustomNeuralForecast
from forts.visualization.model_visualization import plot_generated_vs_original


class UnivariateTimeMixer(TimeMixer):
    SAMPLING_TYPE = "windows"
    MULTIVARIATE = False

    def __init__(self, n_series, **kwargs):
        # force n_series to 1 for univariate mode internals
        # force valid_batch_size=1 to avoid NaN predictions bug in neuralforecast
        # when processing batched predictions with MULTIVARIATE=False
        kwargs["valid_batch_size"] = 1
        kwargs["limit_val_batches"] = 64
        super().__init__(n_series=1, **kwargs)
        self.enc_in = 1
        self.c_out = 1


class AutoUnivariateTimeMixer(AutoTimeMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_model = UnivariateTimeMixer


class FixedxLSTM(xLSTM):
    """
    Custom xLSTM wrapper that sets validation and inference parameters to avoid NaN predictions.
    Similar to UnivariateTimeMixer, this forces specific batch sizes and limits
    validation batches to prevent NaN issues during prediction.

    Also enforces minimum input_size >= 3*h to prevent tensor dimension errors.
    """

    def __init__(self, **kwargs):
        # Force valid_batch_size=1 to avoid NaN predictions bug
        kwargs["valid_batch_size"] = 1
        kwargs["limit_val_batches"] = 64

        # Ensure input_size >= 3*h (xLSTM requirement)
        h = kwargs.get("h")
        if h is not None and "input_size" in kwargs:
            min_input_size = 3 * h
            if kwargs["input_size"] < min_input_size:
                print(
                    f"[FixedxLSTM] Adjusting input_size from {kwargs['input_size']} "
                    f"to {min_input_size} (xLSTM requires input_size >= 3*h)"
                )
                kwargs["input_size"] = min_input_size

        # Set inference_input_size to match input_size for consistent behavior
        if "inference_input_size" not in kwargs and "input_size" in kwargs:
            kwargs["inference_input_size"] = kwargs["input_size"]

        super().__init__(**kwargs)


class AutoFixedxLSTM(AutoxLSTM):
    """
    Auto wrapper for FixedxLSTM that uses the fixed version as the underlying model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_model = FixedxLSTM


AutoModelType = Union[
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
    AutoTimeMOE,
    AutoxLSTM,
    AutoFixedxLSTM,
    AutoTimeMixer,
    AutoUnivariateTimeMixer,
    AutoNBEATS,
]


class _ModelListMixin:
    """
    Mixin that provides a `get_model_list()` method.
    Subclasses can override `MODEL_LIST` or `get_model_list`
    to control exactly which Auto-models are trained.
    """

    MODEL_LIST: list[tuple[str, AutoModelType]] = [
        ("AutoNHITS", AutoNHITS),
        ("AutoKAN", AutoKAN),
        ("AutoPatchTST", AutoPatchTST),
        ("AutoiTransformer", AutoiTransformer),
        ("AutoTSMixer", AutoTSMixer),
        ("AutoTFT", AutoTFT),
        ("AutoTimeMOE", AutoTimeMOE),
        ("AutoxLSTM", AutoFixedxLSTM),
        ("AutoTimeMixer", AutoUnivariateTimeMixer),
        ("AutoNBEATS", AutoNBEATS),
    ]

    def get_model_list(self):
        return self.MODEL_LIST

    def _get_best_valid_trial(self, results: pd.DataFrame, model_name: str):
        """
        Get the best trial that has a valid (non-zero, non-NaN, non-Inf) loss.

        Parameters
        ----------
        results : pd.DataFrame
            Results dataframe from Ray Tune hyperparameter search
        model_name : str
            Name of the model for logging

        Returns
        -------
        pd.Series or None
            The best valid trial, or None if no valid trials exist
        """
        # Filter out invalid trials
        valid_results = results[
            (results["loss"] > 0)
            & (results["loss"].notna())
            & (~results["loss"].isin([np.inf, -np.inf]))
        ]

        if len(valid_results) == 0:
            return None

        # Return the trial with the lowest valid loss
        best_valid_trial = valid_results.loc[valid_results["loss"].idxmin()]
        return best_valid_trial

    def _validate_trained_model(
        self, model, model_name: str, trainval_data: pd.DataFrame
    ) -> None:
        """
        Validate a trained model to detect degenerate solutions.

        This method performs two checks:
        1. Checks if the best trial has suspicious loss values (0, NaN, Inf)
        2. Runs a sanity check by making predictions on a small sample to detect NaN outputs

        Parameters
        ----------
        model : CustomNeuralForecast
            The trained model to validate
        model_name : str
            Name of the model for logging
        trainval_data : pd.DataFrame
            Training/validation data for sanity check predictions

        Raises
        ------
        ValueError
            If all trials have invalid loss values or if the model produces >50% NaN predictions
        """
        # Check 1: Validate loss values from hyperparameter tuning
        results = model.models[0].results.get_dataframe()
        best_trial = results.loc[results["loss"].idxmin()]
        best_loss = best_trial["loss"]

        if best_loss == 0.0 or np.isnan(best_loss) or np.isinf(best_loss):
            print(
                f"[WARNING] Ray Tune selected a trial with suspicious loss={best_loss} for {model_name}!"
            )

            # Try to find the best valid trial
            best_valid_trial = self._get_best_valid_trial(results, model_name)

            if best_valid_trial is not None:
                valid_loss = best_valid_trial["loss"]
                invalid_count = len(results) - len(
                    results[
                        (results["loss"] > 0)
                        & (results["loss"].notna())
                        & (~results["loss"].isin([np.inf, -np.inf]))
                    ]
                )
                print(
                    f"[INFO] Found best valid trial with loss={valid_loss:.4f}. "
                    f"Discarded {invalid_count}/{len(results)} trials with invalid loss. "
                    f"NOTE: The model was already trained with the degenerate trial. "
                    f"Recommend re-running with adjusted hyperparameters to avoid loss=0 trials."
                )
            else:
                raise ValueError(
                    f"All {len(results)} trials for {model_name} resulted in invalid loss values "
                    "(0, NaN, or Inf). This model configuration is not working. "
                    "Please check the data, model settings, or try a different model."
                )

        # Check 2: Sanity check - make predictions on a small sample to detect NaN outputs
        sample_size = min(3, len(trainval_data["unique_id"].unique()))
        sample_ids = trainval_data["unique_id"].unique()[:sample_size]
        sample_df = trainval_data[trainval_data["unique_id"].isin(sample_ids)].copy()

        sanity_preds = model.predict(df=sample_df)
        pred_col = str(model.models[0])
        if pred_col in sanity_preds.columns:
            nan_count = sanity_preds[pred_col].isna().sum()
            total_count = len(sanity_preds)
            if nan_count > 0:
                nan_pct = (nan_count / total_count) * 100
                print(
                    f"[WARNING] Sanity check: {model_name} produced {nan_count}/{total_count} "
                    f"({nan_pct:.1f}%) NaN predictions on training sample! "
                    "This model may not generalize well."
                )
                if nan_pct > 50:
                    error_msg = (
                        f"{model_name} produced {nan_pct:.1f}% NaN predictions on training data. "
                        "This model is not usable. Please try different hyperparameters or a different model."
                    )
                    print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)


class ModelPipeline(_ModelListMixin):
    """
    pipeline that:
      - Re-uses an existing DataPipeline instance for data splits/freq/horizon.
      - Hyper-tunes and trains models
      - Predict functions for different strategies
    """

    def __init__(self, data_pipeline):
        """
        data_pipeline : DataPipeline
            A fully initialized instance of existing DataPipeline,
            used to retrieve train/val/test splits, freq, horizon, etc
        """
        self.hp = data_pipeline
        self.freq = self.hp.freq
        self.h = self.hp.h

        # TRAIN+VAL (transfer learning)
        self.trainval_long = (
            self.hp.original_trainval_long[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # TRAIN+VAL (basic forecasting)
        self.trainval_long_basic_forecast = (
            self.hp.original_trainval_long_basic_forecast[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # TEST data (transfer learning)
        self.test_long = (
            self.hp.original_test_long[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # TEST (basic forecasting)
        self.test_long_basic_forecast = (
            self.hp.original_test_long_basic_forecast[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # combined basic-forecast dataset (train+test)
        self.original_long_basic_forecast = pd.concat(
            [self.trainval_long_basic_forecast, self.test_long_basic_forecast],
            ignore_index=True,
        )

        self.models = {}

    def hyper_tune_and_train(
        self,
        dataset_source,
        dataset_group_source,
        max_evals=MAX_EVALS,
        mode="in_domain",
        test_mode: bool = False,
        max_steps: int = None,
        model_list=None,
    ):
        """
        Trains and hyper-tunes all six models.
        Each data_pipeline does internal time-series cross-validation to select
        its best hyperparameters.
        """
        valid_modes = {
            "in_domain",
            "out_domain",
            "out_domain_coreset",
            "basic_forecasting",
        }
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Supported modes are: "
                f"{sorted(valid_modes)}."
            )

        path_mode = "coreset" if mode == "out_domain_coreset" else mode
        mode = "out_domain" if mode in ("out_domain", "out_domain_coreset") else mode

        if mode == "basic_forecasting":
            trainval_long = self.trainval_long_basic_forecast
            mode_suffix = "_basic_forecasting"
        else:
            trainval_long = self.trainval_long
            mode_suffix = ""

        # Filter out series that are too short for the validation set.
        # The length of a series must be greater than the validation size.
        val_size = self.h
        group_sizes = trainval_long.groupby("unique_id")["unique_id"].transform("size")
        original_series_count = len(trainval_long["unique_id"].unique())
        trainval_long = trainval_long[group_sizes > val_size]
        filtered_series_count = len(trainval_long["unique_id"].unique())

        if original_series_count > filtered_series_count:
            print(
                f"Filtered out {original_series_count - filtered_series_count} series "
                f"from the training data because they were shorter than the validation size ({val_size})."
            )

        num_cpus = os.cpu_count() - 3 if os.cpu_count() > 3 else 1
        gpus = 1 if torch.cuda.is_available() else 0

        print(
            f"Available resources: {os.cpu_count()} CPUs, {torch.cuda.device_count()} GPUs"
        )
        print(f"Using {num_cpus} CPUs and {gpus} GPUs for training.")

        models_to_train = (
            model_list if model_list is not None else self.get_model_list()
        )

        weights_folder = get_model_weights_path()
        if test_mode:
            weights_folder = f"{weights_folder}/test"
        else:
            weights_folder = f"{weights_folder}/{path_mode}"

        save_dir = f"{weights_folder}/hypertuning{mode_suffix}"

        for name, ModelClass in models_to_train:
            print(f"\n=== Handling {name} ===")
            if name in ("AutoTSMixer", "AutoiTransformer", "AutoTimeMixer"):
                n_series_arg = 1

                init_kwargs = dict(
                    h=self.h,
                    n_series=n_series_arg,
                    num_samples=max_evals,
                    verbose=True,
                    cpus=num_cpus,
                    gpus=gpus,
                )
                base_config = ModelClass.get_default_config(
                    h=self.h,
                    backend="ray",
                    n_series=n_series_arg,
                )
                base_config["scaler_type"] = tune.choice([None, "standard"])
                base_config["log_every_n_steps"] = 5
            else:
                init_kwargs = dict(
                    h=self.h,
                    num_samples=max_evals,
                    verbose=True,
                    cpus=num_cpus,
                    gpus=gpus,
                )
                base_config = ModelClass.get_default_config(h=self.h, backend="ray")
                base_config["start_padding_enabled"] = True
                base_config["scaler_type"] = tune.choice([None, "standard"])
                base_config["log_every_n_steps"] = 5
            if max_steps is None:
                base_config["max_steps"] = 1000
            else:
                base_config["max_steps"] = max_steps

            if mode == "out_domain":
                base_config["input_size"] = self.h

            init_kwargs["config"] = base_config

            nf_save_path = f"{save_dir}/{dataset_source}_{dataset_group_source}_{name}_neuralforecast"

            local_trainval_long = trainval_long.copy()
            if name in ("AutoTSMixer", "AutoiTransformer"):
                # These models do not support start_padding_enabled, so we pad manually
                # We need to determine the max input_size that will be tuned
                if "input_size" in base_config and isinstance(
                    base_config["input_size"], tune.search.sample.Categorical
                ):
                    max_input_size = max(base_config["input_size"].categories)
                else:
                    max_input_size = self.h * 2

                required_length = max_input_size + self.h
                if self.freq != "mixed":
                    local_trainval_long = _pad_for_unsupported_models(
                        local_trainval_long, self.freq, required_length
                    )

            try:
                print(
                    f"Attempting to load saved model for {name} from {nf_save_path}..."
                )
                auto_model = ModelClass(**init_kwargs)
                nf = CustomNeuralForecast(models=[auto_model], freq=self.freq)

                load_path = nf_save_path
                local_path = _get_local_fallback_path(nf_save_path)
                if os.path.exists(local_path):
                    load_path = local_path

                model = nf.load(path=load_path)
                print("Load successful.")
            except FileNotFoundError:
                print(
                    f"No saved {name} found at {nf_save_path}. "
                    "Training & tuning from scratch..."
                )
                auto_model = ModelClass(**init_kwargs)
                model = CustomNeuralForecast(models=[auto_model], freq=self.freq)
                model.fit(
                    df=local_trainval_long,
                    val_size=self.h,
                )

                # Check if Ray Tune selected a degenerate trial and refit with best valid one
                results = model.models[0].results.get_dataframe()
                best_trial = results.loc[results["loss"].idxmin()]
                best_loss = best_trial["loss"]

                if best_loss == 0.0 or np.isnan(best_loss) or np.isinf(best_loss):
                    print(
                        f"[WARNING] Ray Tune selected a degenerate trial with loss={best_loss} for {name}!"
                    )

                    # Find the best valid trial
                    best_valid_trial = self._get_best_valid_trial(results, name)

                    if best_valid_trial is None:
                        print(
                            f"[ERROR] All {len(results)} trials for {name} resulted in invalid loss values."
                        )
                        print(
                            f"[SKIP] Not saving {name} - no usable configuration found."
                        )
                        continue

                    # Extract the valid configuration
                    trial_dict = best_valid_trial.to_dict()
                    valid_loss = trial_dict.pop("loss", None)

                    # Extract only the config parameters (those with 'config/' prefix)
                    valid_config = {}
                    for key, value in trial_dict.items():
                        if key.startswith("config/"):
                            # Remove the 'config/' prefix
                            param_name = key.replace("config/", "")
                            valid_config[param_name] = value

                    # Remove metadata parameters
                    valid_config.pop("step_size", None)
                    valid_config.pop("start_padding_enabled", None)
                    valid_config.pop("log_every_n_steps", None)
                    valid_config.pop("valid_loss", None)
                    valid_config.pop("h", None)  # h is passed separately

                    # ensure input_size >= 3 * h for xLSTM-based models
                    if "input_size" in valid_config:
                        min_input_size = 3 * self.h
                        if valid_config["input_size"] < min_input_size:
                            print(
                                f"[FIX] Adjusting input_size from {valid_config['input_size']} "
                                f"to {min_input_size} for {name} (xLSTM requires input_size >= 3*h)"
                            )
                            valid_config["input_size"] = min_input_size

                    invalid_count = len(results) - len(
                        results[
                            (results["loss"] > 0)
                            & (results["loss"].notna())
                            & (~results["loss"].isin([np.inf, -np.inf]))
                        ]
                    )

                    print(
                        f"[INFO] Refitting {name} with best valid trial "
                        f"(loss={valid_loss:.4f}). "
                        f"Discarded {invalid_count}/{len(results)} degenerate trials."
                    )

                    # Get the underlying model class and instantiate directly
                    # We don't use the Auto wrapper here because it would run Ray Tune again
                    actual_model_class = model.models[0].cls_model

                    # Preserve the results from the original Auto model for validation
                    original_results = model.models[0].results

                    # Create a new instance with the valid configuration
                    # The underlying model will handle input_size validation
                    refitted_model = actual_model_class(h=self.h, **valid_config)

                    # Attach the results to the refitted model for validation
                    refitted_model.results = original_results

                    # Refit with the valid configuration
                    model_refitted = CustomNeuralForecast(
                        models=[refitted_model], freq=self.freq
                    )
                    model_refitted.fit(df=local_trainval_long, val_size=self.h)
                    model = model_refitted

                    print(f"[SUCCESS] {name} refitted with valid configuration.")

                # Validate the fitted model - sanity check for NaN predictions
                # This will raise ValueError if the model produces >50% NaN
                try:
                    self._validate_trained_model(model, name, local_trainval_long)
                except ValueError as e:
                    print(f"[ERROR] Validation failed for {name}: {e}")
                    print(
                        f"[SKIP] Not saving {name} - model produces too many NaN predictions."
                    )
                    # Don't add to self.models and continue to next model
                    continue

                try:
                    model.save(path=nf_save_path, overwrite=True, save_dataset=False)
                except Exception as e:
                    print(f"Failed to save to GCS: {e}. Saving to local fallback.")
                    local_path = _get_local_fallback_path(nf_save_path)
                    model.save(path=local_path, overwrite=True, save_dataset=False)

                print(f"Saved {name} NeuralForecast object to {nf_save_path}")

                # Save hyperparameter tuning results
                results = model.models[0].results.get_dataframe()
                results_file = f"{save_dir}/{dataset_source}_{dataset_group_source}_{name}_results.csv"
                gcs_write_csv(results, results_file)
                print(f"Saved tuning results to {results_file}")

            self.models[name] = model

        print("\nAll Auto-models have been trained/tuned or loaded from disk.\n")

    def finetune(
        self,
        model_name: str,
        nf_model: CustomNeuralForecast,
        dataset_source: str,
        dataset_group_source: str,
        test_mode: bool = False,
        max_steps: int = 10,
    ):
        """
        Fine-tunes a given model for 10 epochs on the new target training data.
        """
        target_train_df = self.trainval_long
        if target_train_df.empty:
            print(f"[Fine-tuning SKIP] No training data for {model_name}.")
            return nf_model

        print(f"\n=== Fine-tuning {model_name} for 10 epochs... ===")

        # Get the underlying auto model and update its config for fine-tuning
        auto_model = nf_model.models[0]

        auto_model.max_steps = max_steps
        auto_model.val_check_steps = 10

        # Create a new NeuralForecast object for the fine-tuning process
        finetune_nf = CustomNeuralForecast(models=[auto_model], freq=self.freq)

        # Fit the model on the new data
        finetune_nf.fit(df=target_train_df, val_size=self.h)

        # Save the fine-tuned model
        weights_folder = get_model_weights_path()
        if test_mode:
            save_dir = f"{weights_folder}/test/hypertuning_finetuned"
        else:
            save_dir = f"{weights_folder}/finetuned/hypertuning"
        nf_save_path = f"{save_dir}/{dataset_source}_{dataset_group_source}_{model_name}_neuralforecast"
        try:
            finetune_nf.save(path=nf_save_path, overwrite=True, save_dataset=False)
        except Exception as e:
            print(f"Failed to save to GCS: {e}. Saving to local fallback.")
            local_path = _get_local_fallback_path(nf_save_path)
            finetune_nf.save(path=local_path, overwrite=True, save_dataset=False)

        print(
            f"Saved fine-tuned {model_name} " f"NeuralForecast object to {nf_save_path}"
        )

        print(f"✓ Fine-tuning complete for {model_name}.")
        return finetune_nf

    @staticmethod
    def _mark_context_rows(
        group: pd.DataFrame, window_size_source: int, horizon: int, mode: str
    ) -> pd.DataFrame:
        """
        Given rows for a single unique_id (already sorted by ds),
        slice off the last horizon points so we only keep the context portion.
        Return an empty DataFrame if not enough data.
        """
        n = len(group)
        if n < window_size_source + horizon:
            return pd.DataFrame(columns=group.columns)

        if mode == "out_domain":
            horizon = min(window_size_source, horizon)

        last_window_end = n - horizon
        return group.iloc[:last_window_end].copy()

    def _preprocess_context(
        self,
        window_size: int,
        test_set: pd.DataFrame,
        window_size_source: int = None,
        mode: str = None,
    ) -> pd.DataFrame:
        if not window_size_source:
            window_size_source = window_size

        df_test = test_set.sort_values(["unique_id", "ds"]).copy()

        horizon = (
            min(window_size_source, window_size)
            if mode == "out_domain"
            else window_size
        )

        required_size = window_size_source + horizon
        sizes = df_test.groupby("unique_id")["unique_id"].transform("size")
        pos_from_end = df_test.groupby("unique_id").cumcount(ascending=False)

        mask = (sizes >= required_size) & (pos_from_end >= horizon)
        df_context = df_test[mask].reset_index(drop=True)

        if "y_true" in df_context.columns:
            df_context = df_context.rename(columns={"y_true": "y"})

        return df_context[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    @staticmethod
    def _mark_prediction_rows(group: pd.DataFrame, horizon: int) -> pd.DataFrame:

        last_window_end = horizon
        return group.iloc[:last_window_end].copy()

    def predict_from_last_window_one_pass(
        self,
        model: CustomNeuralForecast,
        window_size: int,
        window_size_source: int,
        dataset_target: str,
        dataset_group_target: str,
        dataset_source: str,
        dataset_group_source: str,
        freq: str,
        h: int,
        mode: str = "in_domain",
    ) -> pd.DataFrame:
        """
        Predicts exactly the last horizon h points for each test series in a single pass.
        """
        model_name = str(model.models[0])

        dataset_desc = f"{dataset_source}-{dataset_group_source}"
        dataset_name_for_title = dataset_source
        dataset_group_for_title = dataset_group_source

        if mode == "in_domain":
            df_y_preprocess = self._preprocess_context(
                window_size=window_size, test_set=self.test_long
            )
            df_y_hat = model.predict(df=df_y_preprocess, freq=freq)

            df_y = self.test_long
        elif mode == "out_domain":
            df_y_preprocess = self._preprocess_context(
                window_size_source=window_size_source,
                window_size=window_size,
                test_set=self.original_long_basic_forecast,
                mode=mode,
            )

            if df_y_preprocess.empty:
                # no series has enough context, so nothing to predict
                print(
                    f"[SKIP] '{dataset_source or dataset_target}' – "
                    "no series meets the context requirement "
                    f"(window = {window_size_source or window_size})."
                )
                return pd.DataFrame()

            df_y_hat_raw = model.predict(df=df_y_preprocess, freq=freq)

            # Replaces the groupby().apply() with a more efficient version
            df_y_hat = df_y_hat_raw.loc[
                df_y_hat_raw.groupby("unique_id").cumcount() < window_size
            ].copy()
            df_y_hat = df_y_hat.reset_index(drop=True)

            df_y_hat.sort_values(["unique_id", "ds"], inplace=True)

            df_y = self.original_long_basic_forecast

            dataset_desc = (
                f"{dataset_source}-{dataset_group_source}"
                f"_to_"
                f"{dataset_target}-{dataset_group_target}"
            )
            dataset_name_for_title = f"{dataset_source}→{dataset_target}"
            dataset_group_for_title = f"{dataset_group_source}→{dataset_group_target}"
        elif mode == "basic_forecasting":
            df_y_hat = model.predict(df=self.trainval_long_basic_forecast, freq=freq)

            df_y = self.original_long_basic_forecast
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Supported modes are: "
                "'in_domain', 'out_domain', 'basic_forecasting'."
            )

        # Drop NaN predictions
        if not df_y_hat.empty:
            df_y_hat = df_y_hat.dropna(subset=[model_name])

        df_y_hat.rename(columns={model_name: "y"}, inplace=True)
        df_y_hat["y"] = df_y_hat["y"].clip(lower=0)
        df_y_hat = df_y_hat.groupby("unique_id", group_keys=False, observed=False).tail(
            h
        )

        if "y_true" in df_y.columns:
            df_y = df_y.rename(columns={"y_true": "y"})

        suffix_name = f"{model_name}-last-window-one-pass_{mode}_{dataset_desc}"
        title = (
            f"{model_name} • Last-window one-pass "
            f"({mode.replace('_', ' ')}) — "
            f"{dataset_name_for_title} [{dataset_group_for_title}]"
        )

        plot_generated_vs_original(
            synth_data=df_y_hat[["unique_id", "ds", "y"]],
            original_data=df_y[["unique_id", "ds", "y"]],
            dataset_name=dataset_name_for_title,
            dataset_group=dataset_group_for_title,
            model_name=model_name,
            n_series=8,
            suffix_name=suffix_name,
            title=title,
        )

        df_y.rename(columns={"y": "y_true"}, inplace=True)

        df_y_y_hat = df_y.merge(df_y_hat, on=["unique_id", "ds"], how="left")

        return df_y_y_hat


class ModelPipelineCoreset(ModelPipeline):
    """
    A lightweight version of ModelPipeline that trains the
    best-performing models on a mixed training set (the coreset)
    """

    def __init__(
        self,
        long_df: pd.DataFrame,
        freq: str,
        h: int,
    ):
        self.freq = freq
        self.h = h

        # coreset itself (train + val)
        self.trainval_long = long_df.sort_values(["unique_id", "ds"])

        # dummy placeholders so inherited code that references them does not fail
        self.trainval_long_basic_forecast = self.trainval_long
        self.original_long_basic_forecast = self.trainval_long

        empty = pd.DataFrame(columns=self.trainval_long.columns)
        self.test_long = empty
        self.test_long_basic_forecast = empty

        self.models = {}
