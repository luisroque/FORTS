import os
from unittest.mock import patch

import numpy as np
import pytest
from neuralforecast.auto import AutoxLSTM

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.model_pipeline import ModelPipeline


@pytest.fixture
def setup_pipeline():
    """Sets up a DataPipeline and ModelPipeline for a small dataset."""
    dp = DataPipeline(
        dataset_name="Tourism",
        dataset_group="Monthly",
        freq="ME",
        horizon=12,
        window_size=24,
    )
    mp = ModelPipeline(data_pipeline=dp)
    mp.MODEL_LIST = [("AutoxLSTM", AutoxLSTM)]
    return dp, mp


def test_in_domain_forecasting(setup_pipeline):
    """
    Tests the in_domain forecasting pipeline to ensure it runs without errors
    and produces a valid forecast without NaN values.
    """
    dp, mp = setup_pipeline

    # Mock CustomNeuralForecast.load to raise FileNotFoundError
    # This forces the pipeline to train the model from scratch
    with patch(
        "forts.model_pipeline.model_pipeline.CustomNeuralForecast.load"
    ) as mock_load:
        mock_load.side_effect = FileNotFoundError

        # Use a minimal number of evaluations for speed
        mp.hyper_tune_and_train(
            max_evals=1,
            mode="in_domain",
            dataset_source="Tourism",
            dataset_group_source="Monthly",
            test_mode=True,
            max_steps=2,
        )

    # Evaluate the first model
    model_name, model = list(mp.models.items())[0]
    row_forecast = {}

    results_file = (
        f"assets/results_forecast_in_domain/Tourism_Monthly_{model_name}_12.json"
    )
    if os.path.exists(results_file):
        os.remove(results_file)

    evaluation_pipeline_forts_forecast(
        dataset="Tourism",
        dataset_group="Monthly",
        model=model,
        pipeline=mp,
        period=dp.period,
        horizon=dp.h,
        freq=dp.freq,
        row_forecast=row_forecast,
        window_size=dp.window_size,
        window_size_source=dp.window_size,
        mode="in_domain",
        test_mode=True,
    )

    # Check that some results were produced
    assert "Forecast SMAPE MEAN (last window) Per Series_in_domain" in row_forecast
    forecast_value = row_forecast[
        "Forecast SMAPE MEAN (last window) Per Series_in_domain"
    ]
    assert forecast_value is not None
    assert not np.isnan(
        forecast_value
    ), f"Forecast resulted in NaN value. This indicates xLSTM produced invalid predictions."
