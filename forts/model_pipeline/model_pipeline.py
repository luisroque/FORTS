import os
from typing import Union

import pandas as pd
import torch
from neuralforecast.auto import (
    AutoiTransformer,
    AutoKAN,
    AutoNHITS,
    AutoPatchTST,
    AutoTFT,
    AutoTSMixer,
)
from ray import tune

from forts.experiments.helper import _pad_for_unsupported_models
from forts.gcs_utils import (
    _get_local_fallback_path,
    gcs_write_csv,
    get_model_weights_path,
)
from forts.model_pipeline.auto.AutoModels import AutoTimeMOE
from forts.model_pipeline.core.core_extension import CustomNeuralForecast
from forts.visualization.model_visualization import plot_generated_vs_original

AutoModelType = Union[
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
    AutoTimeMOE,
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
    ]

    def get_model_list(self):
        return self.MODEL_LIST


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
        max_evals=20,
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

        num_cpus = os.cpu_count() - 3 if os.cpu_count() > 3 else 1
        gpus = 1 if torch.cuda.is_available() else 0

        print(
            f"Available resources: {os.cpu_count()} CPUs, {torch.cuda.device_count()} GPUs"
        )
        print(f"Using {num_cpus} CPUs and {gpus} GPUs for training.")

        if model_list is None:
            model_list = self.get_model_list()

        weights_folder = get_model_weights_path()
        if test_mode:
            weights_folder = f"{weights_folder}/test"
        else:
            weights_folder = f"{weights_folder}/{path_mode}"

        save_dir = f"{weights_folder}/hypertuning{mode_suffix}"

        for name, ModelClass in model_list:
            print(f"\n=== Handling {name} ===")
            if name in ("AutoTSMixer", "AutoiTransformer"):
                init_kwargs = dict(
                    h=self.h,
                    n_series=1,
                    num_samples=max_evals,
                    verbose=True,
                    cpus=num_cpus,
                    gpus=gpus,
                )
                base_config = ModelClass.get_default_config(
                    h=self.h,
                    backend="ray",
                    n_series=1,
                )
                base_config["scaler_type"] = tune.choice([None, "standard"])
                base_config["log_every_n_steps"] = 10
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
                base_config["log_every_n_steps"] = 10
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

                try:
                    model.save(path=nf_save_path, overwrite=True, save_dataset=False)
                except Exception as e:
                    print(f"Failed to save to GCS: {e}. Saving to local fallback.")
                    local_path = _get_local_fallback_path(nf_save_path)
                    model.save(path=local_path, overwrite=True, save_dataset=False)

                print(f"Saved {name} NeuralForecast object to {nf_save_path}")

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

        df_context = df_test.groupby(
            "unique_id", group_keys=True, as_index=False
        ).apply(
            lambda g: self._mark_context_rows(
                group=g,
                window_size_source=window_size_source,
                horizon=window_size,
                mode=mode,
            )
        )
        df_context = df_context.reset_index(drop=True)

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

            df_y_hat = df_y_hat_raw.groupby(
                "unique_id", group_keys=True, as_index=False
            ).apply(lambda g: self._mark_prediction_rows(group=g, horizon=window_size))
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

        df_y_hat.rename(columns={model_name: "y"}, inplace=True)
        df_y_hat["y"] = df_y_hat["y"].clip(lower=0)
        df_y_hat = df_y_hat.groupby("unique_id", group_keys=False).tail(h)

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
