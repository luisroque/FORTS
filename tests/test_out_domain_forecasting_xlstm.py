import os
from unittest.mock import patch

import numpy as np
import pytest
from neuralforecast.auto import AutoxLSTM

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.model_pipeline import ModelPipeline


@pytest.fixture
def setup_pipelines():
    """Sets up source and target pipelines for out-domain testing."""
    # Source dataset: M3 Monthly
    source_dp = DataPipeline(
        dataset_name="M3",
        dataset_group="Monthly",
        freq="ME",
        horizon=12,
        window_size=24,
    )
    source_mp = ModelPipeline(data_pipeline=source_dp)
    source_mp.MODEL_LIST = [("AutoxLSTM", AutoxLSTM)]

    # Target dataset: Tourism Monthly (different from source)
    target_dp = DataPipeline(
        dataset_name="Tourism",
        dataset_group="Monthly",
        freq="ME",
        horizon=12,
        window_size=24,
    )
    target_mp = ModelPipeline(data_pipeline=target_dp)
    target_mp.MODEL_LIST = [("AutoxLSTM", AutoxLSTM)]

    return source_mp, target_mp, source_dp, target_dp


def test_out_domain_forecasting(setup_pipelines):
    """
    Tests the out_domain forecasting pipeline to ensure it runs without errors
    and produces a valid forecast without NaN values.

    This test mirrors the actual cloud experiments where a model trained on one
    dataset (M3) is evaluated on another dataset (Tourism).
    """
    source_mp, target_mp, source_dp, target_dp = setup_pipelines

    # Mock CustomNeuralForecast.load to raise FileNotFoundError
    # This forces the pipeline to train the model from scratch
    with patch(
        "forts.model_pipeline.model_pipeline.CustomNeuralForecast.load"
    ) as mock_load:
        mock_load.side_effect = FileNotFoundError

        # Train on source dataset (M3)
        source_mp.hyper_tune_and_train(
            max_evals=1,
            mode="out_domain",
            dataset_source="M3",
            dataset_group_source="Monthly",
            test_mode=True,
            max_steps=2,
        )

    # Get the trained model
    model_name, model = list(source_mp.models.items())[0]
    row_forecast = {}

    results_file = (
        f"assets/results_forecast_out_domain/Tourism_Monthly_{model_name}_12_"
        f"trained_on_M3_Monthly.json"
    )
    if os.path.exists(results_file):
        os.remove(results_file)

    # Evaluate on target dataset (Tourism) - this is the out-domain test
    evaluation_pipeline_forts_forecast(
        dataset="Tourism",
        dataset_group="Monthly",
        model=model,
        pipeline=target_mp,
        period=target_dp.period,
        horizon=target_dp.h,
        freq=target_dp.freq,
        row_forecast=row_forecast,
        window_size=target_dp.window_size,
        window_size_source=source_dp.window_size,
        dataset_source="M3",
        dataset_group_source="Monthly",
        mode="out_domain",
        test_mode=True,
    )

    # Check that results were produced
    assert "Forecast SMAPE MEAN (last window) Per Series_out_domain" in row_forecast
    forecast_value = row_forecast[
        "Forecast SMAPE MEAN (last window) Per Series_out_domain"
    ]

    # ensure we don't get None (which indicates NaN predictions)
    assert (
        forecast_value is not None
    ), f"Forecast resulted in None value. This indicates xLSTM produced all NaN predictions in out-domain transfer!"

    assert not np.isnan(
        forecast_value
    ), f"Forecast resulted in NaN value. This indicates xLSTM produced invalid predictions in out-domain transfer!"

    # Also check that other metrics were computed
    assert "Forecast MASE MEAN (last window) Per Series_out_domain" in row_forecast
    assert "Forecast MAE MEAN (last window) Per Series_out_domain" in row_forecast

    print(f"\nOut-domain transfer test passed!")
    print(f"   SMAPE: {forecast_value}")
