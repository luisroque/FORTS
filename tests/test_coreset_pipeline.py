from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline, build_mixed_trainval
from forts.experiments.run_pipeline import DATASET_GROUP_FREQ
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.model_pipeline import ModelPipeline, ModelPipelineCoreset


def test_coreset_pipeline_works_m4_monthly(mocker, tmp_path):
    """
    Tests that the coreset pipeline runs without errors for M4 Monthly.
    """
    # Mocking the GCS interactions to run the test locally
    mocker.patch(
        "forts.model_pipeline.model_pipeline.get_model_weights_path",
        return_value=tmp_path,
    )
    mocker.patch("forts.model_pipeline.model_pipeline.gcs_write_csv")
    mocker.patch("forts.model_pipeline.core.core_extension.CustomNeuralForecast.save")

    # Parameters for a small run
    params = {
        "dataset_source": "M4",
        "dataset_group_source": "Monthly",
        "h": 24,
        "freq": "ME",
        "max_steps": 1,
        "max_evals": 1,
        "coreset_size": 2,
    }

    all_data_pipelines = {}
    for ds, groups in DATASET_GROUP_FREQ.items():
        for grp, meta in groups.items():
            all_data_pipelines[(ds, grp)] = DataPipeline(
                dataset_name=ds,
                dataset_group=grp,
                freq=meta["FREQ"],
                horizon=meta["H"],
                window_size=meta["H"] * 2,
            )

    target_ds, target_grp = "M4", "Monthly"
    target_data_pipeline = all_data_pipelines[(target_ds, target_grp)]

    source_pipelines = [
        data_pipeline
        for ds, data_pipeline in all_data_pipelines.items()
        if ds != (target_ds, target_grp)
    ]

    dataset_source = "MIXED"
    dataset_group = f"ALL_BUT_{target_ds}_{target_grp}"
    mixed_trainval = build_mixed_trainval(
        source_pipelines,
        dataset_source=dataset_source,
        dataset_group=dataset_group,
    )

    # Assert that the mixed_trainval DataFrame is not empty
    assert not mixed_trainval.empty
    # Assert that the target pipeline is the one we expect
    assert target_data_pipeline.dataset_name == params["dataset_source"]

    model_list = [("AutoNHITS", AutoNHITS)]

    mixed_mp = ModelPipelineCoreset(
        mixed_trainval,
        freq="mixed",
        h=target_data_pipeline.h,
    )
    mixed_mp.hyper_tune_and_train(
        max_evals=params["max_evals"],
        mode="out_domain_coreset",
        dataset_source=dataset_source,
        dataset_group_source=dataset_group,
        max_steps=params["max_steps"],
        model_list=model_list,
    )

    heldout_mp = ModelPipeline(target_data_pipeline)

    for model_name, nf_model in mixed_mp.models.items():
        row = {}

        evaluation_pipeline_forts_forecast(
            dataset=target_ds,
            dataset_group=target_grp,
            pipeline=heldout_mp,
            model=nf_model,
            horizon=target_data_pipeline.h,
            freq=target_data_pipeline.freq,
            period=target_data_pipeline.period,
            row_forecast=row,
            dataset_source=dataset_source,
            dataset_group_source=dataset_group,
            mode="out_domain",
            window_size=target_data_pipeline.h,
            window_size_source=target_data_pipeline.h,
            test_mode=True,
        )

        assert (
            row["Forecast SMAPE MEDIAN (last window) Per Series_out_domain"] is not None
        )


def test_coreset_pipeline_loads_from_gcs(mocker, tmp_path):
    """
    Tests that the coreset pipeline can load a pre-trained model from GCS.
    """
    # Mock GCS utilities to control GCS interactions
    mocker.patch("forts.model_pipeline.model_pipeline.gcs_write_csv")
    mocker.patch("forts.model_pipeline.core.core_extension.CustomNeuralForecast.save")

    # Make get_model_weights_path return a real GCS path
    mocker.patch(
        "forts.model_pipeline.model_pipeline.get_model_weights_path",
        return_value="gs://forts-ml-research-466308/forts-experiments/model_weights",
    )

    # Parameters for a small run
    params = {
        "dataset_source": "M4",
        "dataset_group_source": "Monthly",
        "h": 24,
        "freq": "ME",
        "max_steps": 1,
        "max_evals": 1,
        "coreset_size": 2,
    }

    all_data_pipelines = {}
    for ds, groups in DATASET_GROUP_FREQ.items():
        for grp, meta in groups.items():
            all_data_pipelines[(ds, grp)] = DataPipeline(
                dataset_name=ds,
                dataset_group=grp,
                freq=meta["FREQ"],
                horizon=meta["H"],
                window_size=meta["H"] * 2,
            )

    target_ds, target_grp = "M4", "Monthly"
    target_data_pipeline = all_data_pipelines[(target_ds, target_grp)]

    source_pipelines = [
        data_pipeline
        for ds, data_pipeline in all_data_pipelines.items()
        if ds != (target_ds, target_grp)
    ]

    dataset_source = "MIXED"
    dataset_group = f"ALL_BUT_{target_ds}_{target_grp}"
    mixed_trainval = build_mixed_trainval(
        source_pipelines,
        dataset_source=dataset_source,
        dataset_group=dataset_group,
    )

    # Assert that the mixed_trainval DataFrame is not empty
    assert not mixed_trainval.empty
    # Assert that the target pipeline is the one we expect
    assert target_data_pipeline.dataset_name == params["dataset_source"]

    model_list = [("AutoNHITS", AutoNHITS)]

    mixed_mp = ModelPipelineCoreset(
        mixed_trainval,
        freq="mixed",
        h=target_data_pipeline.h,
    )
    mixed_mp.hyper_tune_and_train(
        max_evals=params["max_evals"],
        mode="out_domain_coreset",
        dataset_source=dataset_source,
        dataset_group_source=dataset_group,
        max_steps=params["max_steps"],
        model_list=model_list,
    )

    heldout_mp = ModelPipeline(target_data_pipeline)

    for model_name, nf_model in mixed_mp.models.items():
        row = {}

        evaluation_pipeline_forts_forecast(
            dataset=target_ds,
            dataset_group=target_grp,
            pipeline=heldout_mp,
            model=nf_model,
            horizon=target_data_pipeline.h,
            freq=target_data_pipeline.freq,
            period=target_data_pipeline.period,
            row_forecast=row,
            dataset_source=dataset_source,
            dataset_group_source=dataset_group,
            mode="out_domain",
            window_size=target_data_pipeline.h,
            window_size_source=target_data_pipeline.h,
            test_mode=True,
        )

        assert (
            row["Forecast SMAPE MEDIAN (last window) Per Series_out_domain"] is not None
        )
