import pytest
from neuralforecast.auto import AutoNHITS

from forts.data_pipeline.data_pipeline_setup import DataPipeline
from forts.gcs_utils import (
    gcs_delete_file,
    gcs_path_exists,
    gcs_write_json,
    get_gcs_path,
)
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.core.core_extension import CustomNeuralForecast
from forts.model_pipeline.model_pipeline import ModelPipeline


@pytest.fixture
def tourism_monthly_results_file():
    """
    Creates a dummy results file in GCS for the Tourism/Monthly/AutoNHITS
    experiment, yields the parameters for the test, and then cleans up the
    file.
    """
    # 1. Define the parameters
    params = {
        "dataset": "Tourism",
        "dataset_group": "Monthly",
        "model_name": "AutoNHITS",
        "horizon": 24,
        "mode": "basic_forecasting",
        "freq": "ME",
        "period": 12,
        "test_mode": False,  # We are testing a real results path
        "finetune": False,
    }

    # 2. Construct the GCS path exactly as the pipeline would
    results_folder = get_gcs_path(f"results_forecast_{params['mode']}")
    results_file_path = (
        f"{results_folder}/{params['dataset']}_{params['dataset_group']}_"
        f"{params['model_name']}_{params['horizon']}.json"
    )

    # 3. Create and upload the dummy results file to GCS
    dummy_results = {
        "Dataset": "Tourism",
        "Group": "Monthly",
        "Method": "AutoNHITS",
        "Forecast SMAPE MEAN (last window) Per Series_basic_forecasting": 18.9948,
        "source": "integration_test_fixture",
    }
    gcs_write_json(dummy_results, results_file_path)

    # Sanity check to ensure the file is there before the test runs
    assert gcs_path_exists(results_file_path)

    # Yield the parameters to the test function
    yield params

    # 4. Cleanup: delete the file from GCS after the test is done
    gcs_delete_file(results_file_path)


@pytest.mark.integration
def test_evaluation_pipeline_skips_for_tourism_monthly_with_real_objects(
    tourism_monthly_results_file, mocker
):
    """
    An integration test verifying the pipeline is skipped when the specific
    Tourism/Monthly/AutoNHITS results file exists in Google Cloud Storage.
    This test uses real pipeline and model objects.
    """
    # --- Arrange ---
    # The fixture gives us the parameters used to create the file on GCS.
    params = tourism_monthly_results_file

    # 1. Instantiate real objects for the pipeline.
    dp = DataPipeline(
        dataset_name=params["dataset"],
        dataset_group=params["dataset_group"],
        freq=params["freq"],
        horizon=params["horizon"],
    )
    mp = ModelPipeline(data_pipeline=dp)

    # 2. Mimic the result of the `hyper_tune_and_train` step by creating
    #    the model and placing it inside the pipeline's `models` dictionary.
    #    This ensures our test setup faithfully replicates the app's state.
    nf_model = AutoNHITS(h=params["horizon"], backend="ray")
    model_to_test = CustomNeuralForecast(models=[nf_model], freq=dp.freq)
    mp.models[params["model_name"]] = model_to_test

    # 3. Use mocker.spy to watch the expensive method without replacing it.
    spy = mocker.spy(mp, "predict_from_last_window_one_pass")

    row_forecast = {}  # This dictionary will be populated by the function.

    # The function doesn't accept 'model_name', so we remove it.
    model_name_to_pass = params.pop("model_name")

    # --- Act ---
    # Call the evaluation pipeline, passing the model from the pipeline instance.
    evaluation_pipeline_forts_forecast(
        pipeline=mp,
        model=mp.models[model_name_to_pass],
        row_forecast=row_forecast,
        **params,
    )

    # --- Assert ---
    # 1. The most important check: The spied forecasting function was NOT called.
    spy.assert_not_called()

    # 2. The `row_forecast` dictionary should have been updated with the
    #    contents of the file we uploaded to GCS.
    assert (
        row_forecast["Forecast SMAPE MEAN (last window) Per Series_basic_forecasting"]
        == 18.9948
    )
