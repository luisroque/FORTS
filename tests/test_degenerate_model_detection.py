"""
Test that the model pipeline detects and warns about degenerate models
with loss=0 or that produce NaN predictions.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd


def test_degenerate_model_detection_loss_zero():
    """
    Test that the pipeline detects when the best trial has loss=0,
    which often indicates a degenerate model.
    """
    from neuralforecast.auto import AutoxLSTM

    from forts.data_pipeline.data_pipeline_setup import DataPipeline
    from forts.model_pipeline.model_pipeline import ModelPipeline

    # Setup a minimal pipeline
    dp = DataPipeline(
        dataset_name="Tourism",
        dataset_group="Monthly",
        freq="ME",
        horizon=12,
        window_size=24,
    )
    mp = ModelPipeline(data_pipeline=dp)
    mp.MODEL_LIST = [("AutoxLSTM", AutoxLSTM)]

    # Create a mock results dataframe that simulates the issue
    mock_results_df = pd.DataFrame(
        {
            "loss": [0.0, 5639.79, 6000.0],  # First trial has loss=0 (degenerate!)
            "train_loss": [0.0, 5326.3, 5800.0],
            "trial_id": ["00000", "00001", "00002"],
        }
    )

    # Mock the results attribute on the auto model
    mock_auto_model = MagicMock()
    mock_auto_model.results.get_dataframe.return_value = mock_results_df
    mock_auto_model.__str__ = lambda self: "AutoxLSTM"

    # Mock the NeuralForecast instance
    mock_nf_instance = MagicMock()
    mock_nf_instance.models = [mock_auto_model]

    # Mock predict to return valid predictions for sanity check
    mock_predictions = pd.DataFrame(
        {
            "unique_id": [0, 0, 0],
            "ds": pd.date_range("2020-01-01", periods=3, freq="ME"),
            "AutoxLSTM": [100.0, 110.0, 120.0],  # Valid predictions
        }
    )
    mock_nf_instance.predict.return_value = mock_predictions

    # Test the validation method directly with our mock
    import io
    from contextlib import redirect_stdout

    # Capture the output to check for warnings
    captured_output = io.StringIO()

    with redirect_stdout(captured_output):
        try:
            mp._validate_trained_model(
                mock_nf_instance,
                "AutoxLSTM",
                dp.original_trainval_long_basic_forecast[["unique_id", "ds", "y"]],
            )
        except ValueError as e:
            # It's okay if it raises ValueError for invalid trials
            assert "invalid loss" in str(e).lower() or "degenerate" in str(e).lower()

    # Check that a warning was printed
    output = captured_output.getvalue()
    assert "WARNING" in output and (
        "suspicious" in output.lower() or "valid trials" in output.lower()
    )


def test_results_dataframe_analysis():
    """
    Test that we can correctly identify suspicious loss values in results.
    """
    # Create a results dataframe with various suspicious values
    results_df = pd.DataFrame(
        {
            "loss": [0.0, np.nan, np.inf, -np.inf, 5639.79, 6000.0],
            "trial_id": ["00000", "00001", "00002", "00003", "00004", "00005"],
        }
    )

    # Filter logic from the code
    valid_results = results_df[
        (results_df["loss"] > 0)
        & (results_df["loss"].notna())
        & (~results_df["loss"].isin([np.inf, -np.inf]))
    ]

    # Should only keep the last two trials
    assert len(valid_results) == 2
    assert list(valid_results["trial_id"]) == ["00004", "00005"]

    # Check that suspicious values are filtered out
    assert 0.0 not in valid_results["loss"].values
    assert not valid_results["loss"].isna().any()
    assert not np.isinf(valid_results["loss"]).any()

    print("✓ Successfully filtered out degenerate trials")
    print(f"  Original trials: {len(results_df)}")
    print(f"  Valid trials: {len(valid_results)}")
    print(f"  Filtered: {len(results_df) - len(valid_results)}")
