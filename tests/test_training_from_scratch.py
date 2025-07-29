from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.model_pipeline.core.core_extension import CustomNeuralForecast
from forts.model_pipeline.model_pipeline import ModelPipeline, _ModelListMixin


class SingleModelMixin(_ModelListMixin):
    """A mixin that overrides the model list to only include AutoNHITS."""

    MODEL_LIST = [("AutoNHITS", AutoNHITS)]


class SingleModelTestPipeline(ModelPipeline, SingleModelMixin):
    """A test pipeline that only uses the AutoNHITS model."""


def test_training_from_scratch_with_autonhits(mocker, tmp_path):
    """
    A unit test that verifies the training pipeline trains a new AutoNHITS model
    from scratch on the Tourism dataset.
    """
    # --- Arrange ---
    # 1. Mock all GCS interactions to keep this a fast unit test.
    #    Use the pytest tmp_path fixture for a secure temporary directory.
    mocker.patch(
        "forts.model_pipeline.model_pipeline.get_model_weights_path",
        return_value=str(tmp_path),
    )
    mocker.patch("forts.model_pipeline.model_pipeline.gcs_write_csv")
    mocker.patch("forts.model_pipeline.core.core_extension.CustomNeuralForecast.save")

    # 2. Set up parameters for a minimal run.
    params = {
        "dataset_source": "Tourism",
        "dataset_group_source": "Monthly",
        "h": 24,
        "freq": "ME",
        "max_steps": 1,
        "max_evals": 1,
    }

    # 3. Instantiate the real DataPipeline and our test ModelPipeline.
    dp = DataPipeline(
        dataset_name=params["dataset_source"],
        dataset_group=params["dataset_group_source"],
        freq=params["freq"],
        horizon=params["h"],
    )
    mp = SingleModelTestPipeline(data_pipeline=dp)

    # 4. Spy on the `fit` method to verify that it gets called.
    spy = mocker.spy(CustomNeuralForecast, "fit")

    # --- Act ---
    # This call should find no model and proceed to train one.
    mp.hyper_tune_and_train(
        dataset_source=params["dataset_source"],
        dataset_group_source=params["dataset_group_source"],
        mode="basic_forecasting",
        test_mode=True,
        max_steps=params["max_steps"],
        max_evals=params["max_evals"],
    )

    # --- Assert ---
    # 1. The `fit` method should have been called exactly once.
    spy.assert_called_once()

    # 2. The pipeline's `models` dictionary should contain the newly trained model.
    assert "AutoNHITS" in mp.models
    assert isinstance(mp.models["AutoNHITS"], CustomNeuralForecast)
