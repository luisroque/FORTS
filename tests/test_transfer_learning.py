import os

import pytest
from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.model_pipeline import ModelPipeline


class TestModelPipeline(ModelPipeline):
    MODEL_LIST = [("AutoNHITS", AutoNHITS)]


@pytest.fixture
def setup_pipelines():
    """Sets up source and target pipelines for transfer learning."""
    source_dp = DataPipeline(
        dataset_name="Labour",
        dataset_group="Monthly",
        freq="ME",
        horizon=12,
        window_size=24,
    )
    source_mp = TestModelPipeline(data_pipeline=source_dp)

    target_dp = DataPipeline(
        dataset_name="Tourism",
        dataset_group="Monthly",
        freq="ME",
        horizon=12,
        window_size=24,
    )
    target_mp = TestModelPipeline(data_pipeline=target_dp)

    return source_mp, target_mp, target_dp


def test_transfer_learning(setup_pipelines):
    """
    Tests the transfer learning pipeline.
    """
    source_mp, target_mp, target_dp = setup_pipelines

    source_mp.hyper_tune_and_train(
        max_evals=1,
        mode="out_domain",
        dataset_source="Labour",
        dataset_group_source="Monthly",
        test_mode=True,
        max_steps=2,
    )

    model_name, model = list(source_mp.models.items())[0]
    row_forecast = {}

    results_file = (
        f"assets/test_results/Tourism_Monthly_{model_name}_12_trained_on_"
        f"Labour_Monthly.json"
    )
    if os.path.exists(results_file):
        os.remove(results_file)

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
        window_size_source=source_mp.h,
        mode="out_domain",
        dataset_source="Labour",
        dataset_group_source="Monthly",
        test_mode=True,
    )

    assert "Forecast SMAPE MEAN (last window) Per Series_out_domain" in row_forecast
    assert (
        row_forecast["Forecast SMAPE MEAN (last window) Per Series_out_domain"]
        is not None
    )
