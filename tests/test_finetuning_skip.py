import pytest
from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.gcs_utils import gcs_delete_file, get_gcs_path
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.model_pipeline import ModelPipeline


class TestModelPipeline(ModelPipeline):
    MODEL_LIST = [("AutoNHITS", AutoNHITS)]


@pytest.mark.parametrize(
    "source_dataset, source_group, H_TL, source_freq, target_dataset, target_group, H, target_freq, should_skip",
    [
        # Case 1: Fine-tuning should be skipped (target horizon too small)
        ("Labour", "Monthly", 24, "ME", "Tourism", "Monthly", 12, "ME", True),
        # Case 2: Fine-tuning should run (different seasonalities, valid horizons)
        ("M3", "Yearly", 4, "Y", "M1", "Quarterly", 8, "Q", False),
        # Case 3: Fine-tuning should run (equal horizons)
        ("M3", "Monthly", 12, "ME", "M1", "Monthly", 12, "ME", False),
    ],
)
def test_finetuning_logic(
    capsys,
    source_dataset,
    source_group,
    H_TL,
    source_freq,
    target_dataset,
    target_group,
    H,
    target_freq,
    should_skip,
):
    """
    Tests that fine-tuning is correctly skipped or executed based on the
    relative horizons of the source and target datasets.
    """
    # Setup source pipeline
    source_dp = DataPipeline(
        dataset_name=source_dataset,
        dataset_group=source_group,
        freq=source_freq,
        horizon=H_TL,
        window_size=H_TL * 2,
    )
    source_mp = TestModelPipeline(data_pipeline=source_dp)

    # Setup target pipeline
    target_dp = DataPipeline(
        dataset_name=target_dataset,
        dataset_group=target_group,
        freq=target_freq,
        horizon=H,
        window_size=H * 2,
    )
    target_mp = TestModelPipeline(data_pipeline=target_dp)

    # Train a model on the source data
    source_mp.hyper_tune_and_train(
        max_evals=1,
        mode="out_domain",
        dataset_source=source_dataset,
        dataset_group_source=source_group,
        test_mode=True,
        max_steps=2,
    )
    model_name, model = list(source_mp.models.items())[0]

    # 4. Clean up any pre-existing test files from GCS to ensure a clean slate
    results_folder = get_gcs_path("results/test_results")
    results_file_gcs = (
        f"{results_folder}/{target_dataset}_{target_group}_{model_name}_{H}_"
        f"trained_on_{source_dataset}_{source_group}_finetuning.json"
    )
    gcs_delete_file(results_file_gcs)

    # 5. Run the evaluation pipeline, which now contains the fine-tuning logic
    row_forecast = {}
    evaluation_pipeline_forts_forecast(
        dataset=target_dataset,
        dataset_group=target_group,
        model=model,
        pipeline=target_mp,
        period=target_dp.period,
        horizon=H,
        freq=target_freq,
        row_forecast=row_forecast,
        window_size=H,
        window_size_source=H_TL,
        mode="out_domain",
        dataset_source=source_dataset,
        dataset_group_source=source_group,
        finetune=True,
        test_mode=True,
    )

    # 6. Assert that the correct message (or lack thereof) was printed
    captured = capsys.readouterr()
    if should_skip:
        assert "Skipping fine-tuning" in captured.out
        assert (
            "Forecast SMAPE MEAN (last window) Per Series_out_domain"
            not in row_forecast
        )
    else:
        assert "Skipping fine-tuning" not in captured.out
        # The evaluation should complete, so results should be present
        assert "Forecast SMAPE MEAN (last window) Per Series_out_domain" in row_forecast
