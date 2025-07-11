import os
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.metrics.evaluation_pipeline import \
    evaluation_pipeline_forts_forecast
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
    # 1. Setup source pipeline
    source_dp = DataPipeline(
        dataset_name=source_dataset,
        dataset_group=source_group,
        freq=source_freq,
        horizon=H_TL,
        window_size=H_TL * 2,
    )
    source_mp = TestModelPipeline(data_pipeline=source_dp)

    # 2. Setup target pipeline
    target_dp = DataPipeline(
        dataset_name=target_dataset,
        dataset_group=target_group,
        freq=target_freq,
        horizon=H,
        window_size=H * 2,
    )
    target_mp = TestModelPipeline(data_pipeline=target_dp)

    # 3. Train a model on the source data
    source_mp.hyper_tune_and_train(
        max_evals=1,
        mode="out_domain",
        dataset_source=source_dataset,
        dataset_group_source=source_group,
        test_mode=True,
        max_steps=2,
    )
    model_name, model = list(source_mp.models.items())[0]

    # 4. Replicate the logic from the experiment script
    should_finetune_flag = True  # This simulates passing --finetune
    if should_finetune_flag:
        if H < 2 * H_TL:
            print(
                f"Skipping fine-tuning for {model_name} on {target_dataset}/{target_group} "
                f"from {source_dataset}/{source_group}: "
                f"Target horizon ({H}) is too small for source horizon ({H_TL})."
            )
        else:
            model = target_mp.finetune(
                model_name,
                model,
                dataset_source=target_dataset,
                dataset_group_source=target_group,
                test_mode=True,
                max_steps=2,
            )

    # 5. Assert that the correct message (or lack thereof) was printed
    captured = capsys.readouterr()
    if should_skip:
        assert "Skipping fine-tuning" in captured.out
        assert (
            f"Target horizon ({H}) is too small for source horizon ({H_TL})"
            in captured.out
        )
    else:
        assert "Skipping fine-tuning" not in captured.out

    # 6. Run the evaluation to ensure the pipeline completes without crashing
    row_forecast = {}
    results_file = (
        f"assets/test_results/{target_dataset}_{target_group}_{model_name}_{H}_"
        f"trained_on_{source_dataset}_{source_group}_finetuning.json"
    )
    if os.path.exists(results_file):
        os.remove(results_file)

    evaluation_pipeline_forts_forecast(
        dataset=target_dataset,
        dataset_group=target_group,
        model=model,
        pipeline=target_mp,
        period=target_dp.period,
        horizon=target_dp.h,
        freq=target_dp.freq,
        row_forecast=row_forecast,
        window_size=target_dp.window_size,
        window_size_source=source_mp.h,
        mode="out_domain",
        dataset_source=source_dataset,
        dataset_group_source=source_group,
        finetune=True,  # Replicates the experiment script's behavior
        test_mode=True,
    )

    # 7. Check that some results were produced
    assert "Forecast SMAPE MEAN (last window) Per Series_out_domain" in row_forecast
