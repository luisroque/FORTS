from forts.data_pipeline.data_pipeline_setup import DataPipeline, build_mixed_trainval
from forts.experiments.run_pipeline import DATASET_GROUP_FREQ


def test_coreset_pipeline_works(mocker, tmp_path):
    """
    Tests that the coreset pipeline runs without errors.
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
        "dataset_source": "Tourism",
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
                window_size=meta["H"],
            )

    target_ds, target_grp = "Tourism", "Monthly"
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
