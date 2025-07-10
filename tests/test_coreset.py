import sys
from pathlib import Path
import pytest

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from forts.data_pipeline.data_pipeline_setup import DataPipeline, build_mixed_trainval
from forts.model_pipeline.model_pipeline import ModelPipeline, ModelPipelineCoreset
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast


@pytest.fixture
def setup_coreset_pipelines():
    """Sets up pipelines for coreset leave-one-out testing."""
    # Define a small set of datasets for the coreset
    coreset_datasets = {
        ("Labour", "Monthly"): {"FREQ": "ME", "H": 12},
        ("Tourism", "Monthly"): {"FREQ": "ME", "H": 12},
    }

    all_data_pipelines = {
        (ds, grp): DataPipeline(
            dataset_name=ds,
            dataset_group=grp,
            freq=meta["FREQ"],
            horizon=meta["H"],
            window_size=meta["H"] * 2,
        )
        for (ds, grp), meta in coreset_datasets.items()
    }
    return all_data_pipelines


def test_coreset_leave_one_out(setup_coreset_pipelines):
    """
    Tests the coreset leave-one-out evaluation pipeline.
    """
    all_data_pipelines = setup_coreset_pipelines

    # Use the first pipeline as the held-out target
    (target_ds, target_grp), target_dp = list(all_data_pipelines.items())[0]

    # Use the rest as the source
    source_pipelines = [
        dp
        for (ds, grp), dp in all_data_pipelines.items()
        if (ds, grp) != (target_ds, target_grp)
    ]

    mixed_trainval = build_mixed_trainval(
        source_pipelines,
        dataset_source="MIXED",
        dataset_group=f"ALL_BUT_{target_ds}_{target_grp}",
    )

    mixed_mp = ModelPipelineCoreset(
        mixed_trainval,
        freq="ME",
        h=target_dp.h,
    )
    mixed_mp.hyper_tune_and_train(
        max_evals=1,
        mode="out_domain_coreset",
        dataset_source="MIXED",
        dataset_group_source=f"ALL_BUT_{target_ds}_{target_grp}",
    )

    heldout_mp = ModelPipeline(target_dp)
    model_name, model = list(mixed_mp.models.items())[0]
    row_forecast = {}

    evaluation_pipeline_forts_forecast(
        dataset=target_ds,
        dataset_group=target_grp,
        model=model,
        pipeline=heldout_mp,
        period=target_dp.period,
        horizon=target_dp.h,
        freq=target_dp.freq,
        row_forecast=row_forecast,
        window_size=target_dp.window_size,
        window_size_source=target_dp.h,
        mode="out_domain",
        dataset_source="MIXED",
        dataset_group_source=f"ALL_BUT_{target_ds}_{target_grp}",
    )

    assert "Forecast SMAPE MEAN (last window) Per Series_out_domain" in row_forecast
    assert (
        row_forecast["Forecast SMAPE MEAN (last window) Per Series_out_domain"]
        is not None
    )
