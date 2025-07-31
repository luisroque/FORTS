from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ray import tune

import forts
from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.experiments.helper import _pad_for_unsupported_models
from forts.model_pipeline.model_pipeline import AutoiTransformer, ModelPipeline


@pytest.fixture
def mock_data_pipeline():
    mock = MagicMock()
    mock.freq = "D"
    mock.h = 1

    # Create a toy dataframe with one series shorter than the required length
    data = {
        "unique_id": ["series1"] * 5 + ["series2"] * 15,
        "ds": pd.to_datetime(
            list(pd.date_range(start="2023-01-01", periods=5))
            + list(pd.date_range(start="2023-01-01", periods=15))
        ),
        "y": [1.0] * 20,
    }
    df = pd.DataFrame(data)

    mock.original_trainval_long_basic_forecast = df
    mock.original_trainval_long = df
    mock.original_test_long = pd.DataFrame(columns=["unique_id", "ds", "y"])
    mock.original_test_long_basic_forecast = pd.DataFrame(
        columns=["unique_id", "ds", "y"]
    )

    return mock


def test_conditional_padding(mock_data_pipeline):
    df = mock_data_pipeline.original_trainval_long
    freq = mock_data_pipeline.freq
    required_length = 10

    padded_df = _pad_for_unsupported_models(df, freq, required_length)

    padded_lengths = padded_df.groupby("unique_id").size()

    assert padded_lengths["series1"] == 10
    assert padded_lengths["series2"] == 15


@patch("forts.model_pipeline.model_pipeline.AutoiTransformer.get_default_config")
def test_itransformer_with_short_series_and_padding(
    mock_get_config, mock_data_pipeline, tmp_path
):
    # Mock the config to ensure input_size is large enough to trigger padding for "series1"
    # required_length will be max_input_size + h = 10 + 1 = 11.
    # "series1" has length 5, so it should be padded.
    mock_config = {
        "n_series": 1,
        "input_size": tune.choice([10]),
        "scaler_type": tune.choice([None, "standard"]),
        "log_every_n_steps": 10,
    }
    mock_get_config.return_value = mock_config

    pipeline = ModelPipeline(mock_data_pipeline)

    with (
        patch(
            "forts.model_pipeline.model_pipeline.get_model_weights_path",
            return_value=str(tmp_path),
        ),
        patch("forts.model_pipeline.model_pipeline.gcs_write_csv", return_value=None),
        patch("forts.model_pipeline.core.core_extension.CustomNeuralForecast.save"),
    ):
        try:
            pipeline.hyper_tune_and_train(
                dataset_source="toy_dataset",
                dataset_group_source="toy_group",
                max_evals=1,
                mode="basic_forecasting",
                test_mode=True,
                model_list=[("AutoiTransformer", AutoiTransformer)],
                max_steps=1,
            )
        except Exception as e:
            pytest.fail(f"hyper_tune_and_train raised an exception with padding: {e}")


def test_coreset_e2e_with_mixed_freq_padding_and_large_horizon(mocker, tmp_path):
    """
    Tests that the coreset experiment runs end-to-end with AutoiTransformer
    when source datasets have mixed frequencies (Yearly, Monthly, Daily) and
    the target dataset has a large horizon (h=24). This scenario is designed
    to trigger the window size problem if padding is not handled correctly.
    """
    mocker.patch("sys.argv", ["", "--coreset", "--model", "AutoiTransformer"])
    mocker.patch(
        "forts.model_pipeline.model_pipeline.get_model_weights_path",
        return_value=tmp_path,
    )
    mocker.patch("forts.model_pipeline.model_pipeline.gcs_write_csv")
    mocker.patch("forts.model_pipeline.core.core_extension.CustomNeuralForecast.save")
    mocker.patch("forts.experiments.run_pipeline.evaluation_pipeline_forts_forecast")
    mocker.patch(
        "forts.experiments.run_pipeline.check_results_exist",
        return_value=(False, "dummy_path"),
    )
    pad_spy = mocker.spy(forts.experiments.run_pipeline, "_pad_for_unsupported_models")

    coreset_dataset_group = {
        "M4": {"Monthly": {"FREQ": "ME", "H": 24}},
        "Traffic": {"Daily": {"FREQ": "D", "H": 30}},
        "M3": {"Yearly": {"FREQ": "YE", "H": 4}},
    }
    mocker.patch(
        "forts.experiments.run_pipeline.DATASET_GROUP_FREQ", coreset_dataset_group
    )

    # --- Mock DataPipeline instances ---
    mock_dp_monthly = MagicMock(spec=DataPipeline)
    mock_dp_monthly.h = 24
    mock_dp_monthly.freq = "ME"
    mock_dp_monthly.dataset_name = "M4"
    mock_dp_monthly.dataset_group = "Monthly"
    mock_dp_monthly.period = 12
    mock_dp_monthly.original_trainval_long = pd.DataFrame(
        {
            "unique_id": ["m4_monthly1"] * 100,
            "ds": pd.to_datetime(
                pd.date_range(start="2015-01-01", periods=100, freq="ME")
            ),
            "y": range(100),
        }
    )
    mock_dp_monthly.original_trainval_long_basic_forecast = (
        mock_dp_monthly.original_trainval_long
    )
    mock_dp_monthly.original_test_long = pd.DataFrame(columns=["unique_id", "ds", "y"])
    mock_dp_monthly.original_test_long_basic_forecast = pd.DataFrame(
        columns=["unique_id", "ds", "y"]
    )

    mock_dp_daily = MagicMock(spec=DataPipeline)
    mock_dp_daily.h = 30
    mock_dp_daily.freq = "D"
    mock_dp_daily.dataset_name = "Traffic"
    mock_dp_daily.dataset_group = "Daily"
    mock_dp_daily.period = 365
    mock_dp_daily.original_trainval_long = pd.DataFrame(
        {
            "unique_id": ["traffic1"] * 100,
            "ds": pd.to_datetime(
                pd.date_range(start="2022-01-01", periods=100, freq="D")
            ),
            "y": range(100),
        }
    )
    mock_dp_daily.original_trainval_long_basic_forecast = (
        mock_dp_daily.original_trainval_long
    )
    mock_dp_daily.original_test_long = pd.DataFrame(columns=["unique_id", "ds", "y"])
    mock_dp_daily.original_test_long_basic_forecast = pd.DataFrame(
        columns=["unique_id", "ds", "y"]
    )

    mock_dp_yearly = MagicMock(spec=DataPipeline)
    mock_dp_yearly.h = 4
    mock_dp_yearly.freq = "YE"
    mock_dp_yearly.dataset_name = "M3"
    mock_dp_yearly.dataset_group = "Yearly"
    mock_dp_yearly.period = 1
    mock_dp_yearly.original_trainval_long = pd.DataFrame(
        {
            "unique_id": ["m3_yearly1"] * 10,
            "ds": pd.to_datetime(
                pd.date_range(start="2013-01-01", periods=10, freq="YE")
            ),
            "y": range(10),
        }
    )
    mock_dp_yearly.original_trainval_long_basic_forecast = (
        mock_dp_yearly.original_trainval_long
    )
    mock_dp_yearly.original_test_long = pd.DataFrame(columns=["unique_id", "ds", "y"])
    mock_dp_yearly.original_test_long_basic_forecast = pd.DataFrame(
        columns=["unique_id", "ds", "y"]
    )

    def data_pipeline_side_effect(*args, **kwargs):
        name = kwargs.get("dataset_name")
        if name == "M4":
            return mock_dp_monthly
        if name == "Traffic":
            return mock_dp_daily
        if name == "M3":
            return mock_dp_yearly
        return MagicMock()

    mocker.patch(
        "forts.experiments.run_pipeline.DataPipeline",
        side_effect=data_pipeline_side_effect,
    )
    original_train_method = ModelPipeline.hyper_tune_and_train

    def mocked_train_method(self, *args, **kwargs):
        kwargs["max_evals"] = 1
        kwargs["max_steps"] = 1
        return original_train_method(self, *args, **kwargs)

    mocker.patch(
        "forts.model_pipeline.model_pipeline.ModelPipeline.hyper_tune_and_train",
        side_effect=mocked_train_method,
        autospec=True,
    )
    from forts.experiments.run_pipeline import main

    try:
        main()
    except Exception as e:
        pytest.fail(f"Coreset pipeline failed during end-to-end run: {e}")

    assert pad_spy.call_count == 6
    all_freqs = [call.kwargs["freq"] for call in pad_spy.call_args_list]
    assert all_freqs.count("ME") == 2
    assert all_freqs.count("D") == 2
    assert all_freqs.count("YE") == 2
