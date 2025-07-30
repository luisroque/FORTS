import sys

import pandas as pd
from neuralforecast.auto import AutoNHITS

from forts.experiments.run_pipeline import main
from forts.model_pipeline.core.core_extension import CustomNeuralForecast
from forts.model_pipeline.model_pipeline import ModelPipelineCoreset


class TestModelPipelineCoreset(ModelPipelineCoreset):
    MODEL_LIST = [("AutoNHITS", AutoNHITS)]


def test_coreset_leave_one_out(mocker, tmp_path):
    """
    Tests that the coreset pipeline runs without errors by mocking heavy components.
    """
    # Mock the command-line arguments to trigger the coreset logic
    mocker.patch.object(sys, "argv", ["", "--coreset"])

    # --- Mock external and heavy dependencies ---

    # Mock GCS interactions
    mocker.patch(
        "forts.model_pipeline.model_pipeline.get_model_weights_path",
        return_value=tmp_path,
    )
    mocker.patch("forts.model_pipeline.model_pipeline.gcs_write_csv")
    mocker.patch("forts.model_pipeline.core.core_extension.CustomNeuralForecast.save")

    # Mock the evaluation pipeline, we just want to know it gets called
    mock_evaluation = mocker.patch(
        "forts.experiments.run_pipeline.evaluation_pipeline_forts_forecast"
    )

    # Mock the training function to avoid actual training
    def mock_hyper_tune_and_train(self, *args, **kwargs):
        dummy_model = AutoNHITS(h=self.h)
        self.models["AutoNHITS"] = CustomNeuralForecast(
            models=[dummy_model], freq=self.freq
        )

    mock_train = mocker.patch(
        "forts.model_pipeline.model_pipeline.ModelPipelineCoreset.hyper_tune_and_train",
        side_effect=mock_hyper_tune_and_train,
        autospec=True,
    )

    # Mock the dataset list to a small subset to make the test run fast
    small_dataset_group = {
        "Tourism": {
            "Monthly": {"FREQ": "ME", "H": 24},
        },
        "M1": {
            "Monthly": {"FREQ": "ME", "H": 24},
        },
    }
    mocker.patch(
        "forts.experiments.run_pipeline.DATASET_GROUP_FREQ", small_dataset_group
    )

    # Mock the model list to be just one model
    mocker.patch(
        "forts.experiments.helper.get_model_list",
        return_value=[("AutoNHITS", AutoNHITS)],
    )

    # Mock the results check to ensure the pipeline always runs
    mocker.patch(
        "forts.experiments.run_pipeline.check_results_exist",
        return_value=(False, "dummy_path"),
    )

    # Mock the DataPipeline class to avoid loading data
    mock_data_pipeline = mocker.patch(
        "forts.experiments.run_pipeline.DataPipeline", autospec=True
    )
    mock_data_pipeline.return_value.h = 24
    mock_data_pipeline.return_value.period = 12
    mock_data_pipeline.return_value.freq = "ME"
    dummy_df = pd.DataFrame(
        {
            "unique_id": ["series1"],
            "ds": [pd.to_datetime("2023-01-01")],
            "y": [1.0],
        }
    )
    mock_data_pipeline.return_value.original_trainval_long = dummy_df
    mock_data_pipeline.return_value.original_trainval_long_basic_forecast = dummy_df
    mock_data_pipeline.return_value.original_test_long_basic_forecast = dummy_df
    mock_data_pipeline.return_value.original_test_long = dummy_df
    mock_data_pipeline.return_value.dataset_name = "dummy_name"
    mock_data_pipeline.return_value.dataset_group = "dummy_group"
    # --- Run the pipeline ---
    main()

    # --- Assertions ---
    # The leave-one-out loop should run for each dataset in our small group
    assert mock_train.call_count == 2

    # Evaluation should be called for each trained model in each loop iteration
    # one for each item in the small_dataset_group (2) * number of models (1)
    assert mock_evaluation.call_count == 2
    assert mock_evaluation.called
