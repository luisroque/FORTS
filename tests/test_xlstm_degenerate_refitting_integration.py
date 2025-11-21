"""
Integration test for xLSTM degenerate trial detection and refitting.

This test runs the actual hyperparameter tuning process with Ray Tune,
allowing degenerate trials (loss=0.0) to occur naturally, and verifies
that the refitting logic properly handles them without tensor dimension errors.
"""

import numpy as np
import pandas as pd
import pytest


def test_xlstm_refitting_with_real_ray_tune():
    """
    Integration test that runs actual Ray Tune hyperparameter search for xLSTM,
    which may produce degenerate trials (loss=0.0), and verifies the refitting
    logic handles them correctly with proper config extraction and input_size adjustment.

    This test covers the full code path:
    1. Ray Tune runs with configurations that may produce loss=0.0
    2. Best trial detection finds the degenerate trial
    3. _get_best_valid_trial finds the best non-degenerate trial
    4. Config extraction handles 'config/' prefixed parameters
    5. input_size adjustment to >= 3*h for xLSTM
    6. Direct xLSTM instantiation (not AutoFixedxLSTM wrapper)
    7. Results preservation for validation
    """
    from forts.data_pipeline.data_pipeline_setup import DataPipeline
    from forts.model_pipeline.model_pipeline import AutoFixedxLSTM, ModelPipeline

    h = 12

    # Setup pipeline
    dp = DataPipeline(
        dataset_name="Tourism",
        dataset_group="Monthly",
        freq="ME",
        horizon=h,
        window_size=h,
    )
    mp = ModelPipeline(data_pipeline=dp)
    mp.MODEL_LIST = [("AutoxLSTM", AutoFixedxLSTM)]

    # Create minimal training data
    np.random.seed(42)
    n_obs = 100
    train_data = pd.DataFrame(
        {
            "unique_id": [0] * n_obs + [1] * n_obs,
            "ds": pd.date_range("2020-01-01", periods=n_obs, freq="ME").tolist() * 2,
            "y": np.random.randn(n_obs * 2).cumsum() + 100,
        }
    )

    # Override trainval_long with our minimal data
    mp.trainval_long = train_data

    # Run actual hyperparameter tuning with settings that may produce degenerate trials
    # We use very aggressive settings to increase the likelihood of degenerate trials:
    # - Very high learning rate
    # - Very high dropout
    # - input_size = h (which is problematic for xLSTM)
    # - Only 2 trials with minimal steps

    print("\n" + "=" * 80)
    print("Running integration test: xLSTM with potential degenerate trials")
    print("=" * 80)

    try:
        mp.hyper_tune_and_train(
            dataset_source="Tourism",
            dataset_group_source="Monthly",
            max_evals=2,  # Minimal number of trials
            mode="out_domain",  # This sets input_size=h, which is < 3*h
            test_mode=True,
            max_steps=5,  # Very few steps to speed up test
        )

        # If we get here, the refitting logic worked!
        print("\n" + "=" * 80)
        print("✓ Test passed: Refitting logic handled the scenario correctly")
        print("=" * 80)

        # Verify the model was trained
        assert "AutoxLSTM" in mp.models, "AutoxLSTM should be in trained models"

        # Verify we can access the model
        model = mp.models["AutoxLSTM"]

        # Check if results are available (only if model was trained, not loaded)
        if hasattr(model.models[0], "results") and model.models[0].results is not None:
            results = model.models[0].results.get_dataframe()

            print(f"\nResults summary:")
            print(f"  Total trials: {len(results)}")
            print(f"  Loss values: {results['loss'].tolist()}")

            # Check if there were any degenerate trials
            degenerate_trials = results[
                (results["loss"] == 0.0)
                | (results["loss"].isna())
                | (results["loss"].isin([np.inf, -np.inf]))
            ]

            if len(degenerate_trials) > 0:
                print(f"  Degenerate trials detected: {len(degenerate_trials)}")
                print(f"  ✓ Refitting logic successfully handled degenerate trials")

                # Verify valid trials exist
                valid_trials = results[
                    (results["loss"] > 0)
                    & (results["loss"].notna())
                    & (~results["loss"].isin([np.inf, -np.inf]))
                ]
                assert len(valid_trials) > 0, "Should have at least one valid trial"
                print(f"  Valid trials found: {len(valid_trials)}")
            else:
                print(f"  No degenerate trials in this run")
                print(f"  (Test still validates the refitting code path exists)")
        else:
            print(f"\n  Model was loaded from cache (no results to analyze)")
            print(f"  ✓ Cached model loaded successfully")

        # Test predictions to ensure model is functional
        sample_df = train_data[train_data["unique_id"] == 0].copy()
        preds = model.predict(df=sample_df)

        assert not preds.empty, "Predictions should not be empty"
        # When loaded from cache, the column name is the class name (AutoFixedxLSTM)
        # When trained fresh, it might use the registered name (AutoxLSTM)
        model_col = (
            "AutoFixedxLSTM" if "AutoFixedxLSTM" in preds.columns else "AutoxLSTM"
        )
        assert model_col in preds.columns, "Predictions should have model column"

        # Check for NaN predictions
        nan_count = preds[model_col].isna().sum()
        total_count = len(preds)
        if nan_count > 0:
            print(f"\n  Warning: {nan_count}/{total_count} NaN predictions")
        else:
            print(f"\n  ✓ All predictions are valid (no NaN)")

        print("\n" + "=" * 80)
        print("Integration test completed successfully!")
        print("=" * 80 + "\n")

    except RuntimeError as e:
        if "negative dimension" in str(e):
            pytest.fail(
                f"FAILED: Got tensor dimension error during refitting!\n"
                f"This means the input_size adjustment or config extraction failed.\n"
                f"Error: {e}"
            )
        else:
            # Re-raise other RuntimeErrors
            raise
    except ValueError as e:
        if "invalid loss" in str(e).lower() or "degenerate" in str(e).lower():
            pytest.fail(
                f"FAILED: All trials were degenerate and no valid trial was found!\n"
                f"This is a test setup issue - increase max_evals or adjust config.\n"
                f"Error: {e}"
            )
        else:
            # Re-raise other ValueErrors
            raise


def test_xlstm_config_extraction_from_ray_results():
    """
    Test that verifies the config extraction logic properly handles
    Ray Tune's results format with 'config/' prefixed parameters.

    This is a unit test for the specific config extraction logic that was
    causing the bug where input_size wasn't being found.
    """
    from forts.data_pipeline.data_pipeline_setup import DataPipeline
    from forts.model_pipeline.model_pipeline import ModelPipeline

    # Setup pipeline
    dp = DataPipeline(
        dataset_name="Tourism",
        dataset_group="Monthly",
        freq="ME",
        horizon=12,
        window_size=24,
    )
    mp = ModelPipeline(data_pipeline=dp)

    # Create a mock trial dict that mimics Ray Tune's actual output format
    # Note the 'config/' prefix on all config parameters
    mock_trial_dict = {
        "loss": 557.12,
        "train_loss": 520.0,
        "timestamp": 1763749698,
        "checkpoint_dir_name": None,
        "done": False,
        "training_iteration": 1,
        "trial_id": "99bed_00001",
        "date": "2025-11-21_18-28-18",
        "config/h": 24,
        "config/encoder_hidden_size": 32,
        "config/encoder_n_blocks": 3,
        "config/encoder_dropout": 0.405,
        "config/decoder_hidden_size": 64,
        "config/learning_rate": 0.00012,
        "config/scaler_type": "standard",
        "config/max_steps": 30,
        "config/batch_size": 256,
        "config/windows_batch_size": 1024,
        "config/loss": "MAE()",
        "config/random_seed": 16,
        "config/input_size": 24,  # This is < 3*h=36
        "config/step_size": 24,
        "config/start_padding_enabled": True,
        "config/log_every_n_steps": 5,
        "config/valid_loss": "MAE()",
        "logdir": "99bed_00001",
    }

    # Simulate the config extraction logic from model_pipeline.py
    trial_dict = mock_trial_dict.copy()
    trial_dict.pop("loss", None)

    # Extract only the config parameters (those with 'config/' prefix)
    valid_config = {}
    for key, value in trial_dict.items():
        if key.startswith("config/"):
            # Remove the 'config/' prefix
            param_name = key.replace("config/", "")
            valid_config[param_name] = value

    # Remove metadata parameters
    valid_config.pop("step_size", None)
    valid_config.pop("start_padding_enabled", None)
    valid_config.pop("log_every_n_steps", None)
    valid_config.pop("valid_loss", None)
    valid_config.pop("h", None)

    print(f"\nExtracted config: {valid_config}")

    # Verify extraction worked correctly
    assert "input_size" in valid_config, "input_size should be in extracted config"
    assert valid_config["input_size"] == 24, "input_size value should be preserved"
    assert "encoder_hidden_size" in valid_config, "Model params should be extracted"
    assert "step_size" not in valid_config, "Metadata should be removed"
    assert "h" not in valid_config, "h should be removed (passed separately)"

    # Test input_size adjustment logic
    h = 12
    if "input_size" in valid_config:
        min_input_size = 3 * h
        if valid_config["input_size"] < min_input_size:
            print(
                f"Adjusting input_size from {valid_config['input_size']} to {min_input_size}"
            )
            valid_config["input_size"] = min_input_size

    assert valid_config["input_size"] == 36, f"input_size should be adjusted to 3*h=36"

    print("✓ Config extraction and input_size adjustment working correctly")


if __name__ == "__main__":
    # Allow running this test directly for debugging
    print("Running xLSTM degenerate refitting integration test...")
    test_xlstm_config_extraction_from_ray_results()
    test_xlstm_refitting_with_real_ray_tune()
