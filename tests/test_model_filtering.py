import argparse
from unittest.mock import MagicMock, patch

import pandas as pd

from forts.experiments.run_pipeline import main as run_pipeline_main


def test_model_filtering_basic_forecasting():
    """
    Tests if passing the --model argument correctly filters the models to run
    for the basic_forecasting experiment, without running training/inference.
    """
    mock_args = argparse.Namespace(
        use_gpu=False,
        transfer_learning=False,
        coreset=False,
        basic_forecasting=True,
        finetune=False,
        model="AutoTimeMOE",
    )

    DATASET_GROUP_FREQ_MOCK = {"Tourism": {"Monthly": {"FREQ": "ME", "H": 24}}}

    with (
        patch("forts.experiments.run_pipeline.cmd_parser", return_value=mock_args),
        patch("forts.experiments.run_pipeline.get_data_pipeline") as mock_get_dp,
        patch("forts.experiments.run_pipeline.evaluation_pipeline_forts_forecast"),
        patch(
            "forts.experiments.run_pipeline.check_results_exist",
            return_value=(False, "mock_path"),
        ),
        patch.dict(
            "forts.experiments.run_pipeline.DATASET_GROUP_FREQ",
            DATASET_GROUP_FREQ_MOCK,
            clear=True,
        ),
        patch(
            "forts.model_pipeline.model_pipeline.ModelPipeline.hyper_tune_and_train"
        ) as mock_hyper_tune_and_train,
    ):
        mock_dp_instance = MagicMock()
        # Mock the DataFrame attributes that are accessed by ModelPipeline.__init__
        mock_dp_instance.original_trainval_long = pd.DataFrame(
            columns=["unique_id", "ds", "y"]
        )
        mock_dp_instance.original_trainval_long_basic_forecast = pd.DataFrame(
            columns=["unique_id", "ds", "y"]
        )
        mock_dp_instance.original_test_long = pd.DataFrame(
            columns=["unique_id", "ds", "y"]
        )
        mock_dp_instance.original_test_long_basic_forecast = pd.DataFrame(
            columns=["unique_id", "ds", "y"]
        )
        mock_get_dp.return_value = mock_dp_instance

        run_pipeline_main()

        mock_hyper_tune_and_train.assert_called_once()

        kwargs = mock_hyper_tune_and_train.call_args.kwargs

        assert "model_list" in kwargs
        model_list_arg = kwargs["model_list"]
        assert len(model_list_arg) == 1
        assert model_list_arg[0][0] == "AutoTimeMOE"
