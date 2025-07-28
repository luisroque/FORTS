import pytest
from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.gcs_utils import gcs_delete_directory, get_model_weights_path
from forts.model_pipeline.core.core_extension import CustomNeuralForecast
from forts.model_pipeline.model_pipeline import ModelPipeline, _ModelListMixin


class SingleModelMixin(_ModelListMixin):
    """A mixin that overrides the model list to only include AutoNHITS."""

    MODEL_LIST = [("AutoNHITS", AutoNHITS)]


class SingleModelTestPipeline(ModelPipeline, SingleModelMixin):
    """A test pipeline that only uses the AutoNHITS model."""


@pytest.fixture
def pretrained_model_in_gcs():
    """
    Creates a dummy pretrained model in GCS by fitting it on the real
    Tourism dataset for a single step. Yields the parameters for the test,
    and then cleans up the model weights directory.
    """
    params = {
        "dataset_source": "Tourism",
        "dataset_group_source": "Monthly",
        "model_name": "AutoNHITS",
        "h": 24,
        "freq": "ME",
        "max_steps": 1,  # Minimal training for the dummy model
    }

    # 1. Construct the path where the pipeline expects to find the model
    save_dir = f"{get_model_weights_path()}/basic_forecasting/hypertuning"
    model_path = (
        f"{save_dir}/{params['dataset_source']}_{params['dataset_group_source']}_"
        f"{params['model_name']}_neuralforecast"
    )

    # 2. Create and save a dummy model to that path by fitting it on the
    #    actual Tourism dataset for a single step.
    dp = DataPipeline(
        dataset_name=params["dataset_source"],
        dataset_group=params["dataset_group_source"],
        freq=params["freq"],
        horizon=params["h"],
    )
    config = {
        "input_size": 2 * params["h"],
        "max_steps": params["max_steps"],
    }
    dummy_model = AutoNHITS(h=params["h"], config=config, backend="ray")
    dummy_nf = CustomNeuralForecast(models=[dummy_model], freq=params["freq"])

    # Use the real dataset for fitting, but only the required columns
    fit_df = dp.original_trainval_long_basic_forecast[["unique_id", "ds", "y"]]
    dummy_nf.fit(df=fit_df, val_size=params["h"])
    dummy_nf.save(path=model_path, overwrite=True, save_dataset=False)

    yield params, model_path

    # 4. Cleanup: delete the entire model directory from GCS
    gcs_delete_directory(model_path)


@pytest.mark.integration
def test_training_pipeline_skips_if_model_exists(pretrained_model_in_gcs, mocker):
    """
    An integration test that verifies the training pipeline skips training
    if a pretrained model's weights already exist in Google Cloud Storage.
    """
    # --- Arrange ---
    params, model_path = pretrained_model_in_gcs

    # 1. Instantiate a real DataPipeline and a test version of the ModelPipeline
    dp = DataPipeline(
        dataset_name=params["dataset_source"],
        dataset_group=params["dataset_group_source"],
        freq=params["freq"],
        horizon=params["h"],
    )
    # Use the single-model version to speed up the test
    mp = SingleModelTestPipeline(data_pipeline=dp)

    # 2. Spy on the `fit` method of the forecast object to see if it's called
    spy = mocker.spy(CustomNeuralForecast, "fit")

    # --- Act ---
    # This should find the pretrained model on GCS and load it instead of training.
    mp.hyper_tune_and_train(
        dataset_source=params["dataset_source"],
        dataset_group_source=params["dataset_group_source"],
        mode="basic_forecasting",
        max_steps=1,
    )

    # --- Assert ---
    # 1. The `fit` method should NOT have been called, as training was skipped.
    spy.assert_not_called()

    # 2. The pipeline's `models` dictionary should contain the loaded model.
    assert params["model_name"] in mp.models
    assert isinstance(mp.models[params["model_name"]], CustomNeuralForecast)
