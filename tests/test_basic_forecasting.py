import os

import pytest
from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.model_pipeline import ModelPipeline


class TestModelPipeline(ModelPipeline):
    MODEL_LIST = [("AutoNHITS", AutoNHITS)]


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
    mp = TestModelPipeline(data_pipeline=dp)
    return dp, mp


def test_basic_forecasting(setup_pipeline):
    """
    Tests the basic forecasting pipeline to ensure it runs without errors
    and produces a valid forecast.
    """
    dp, mp = setup_pipeline

    # Use a minimal number of evaluations for speed
    mp.hyper_tune_and_train(
        max_evals=1,
        mode="basic_forecasting",
        dataset_source="Tourism",
        dataset_group_source="Monthly",
        test_mode=True,
        max_steps=2,
    )

    # Evaluate the first model
    model_name, model = list(mp.models.items())[0]
    row_forecast = {}

    results_file = f"assets/results_forecast_basic_forecasting/Tourism_Monthly_{model_name}_12.json"
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
        mode="basic_forecasting",
        test_mode=True,
    )

    # Check that some results were produced
    assert (
        "Forecast SMAPE MEAN (last window) Per Series_basic_forecasting" in row_forecast
    )
    assert (
        row_forecast["Forecast SMAPE MEAN (last window) Per Series_basic_forecasting"]
        is not None
    )
