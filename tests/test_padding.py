from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ray import tune

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
    pipeline = ModelPipeline(mock_data_pipeline)
    df = mock_data_pipeline.original_trainval_long

    required_length = 10

    padded_df = pipeline._pad_for_unsupported_models(df, required_length)

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
