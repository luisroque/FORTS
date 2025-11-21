import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models.xlstm import xLSTM


def make_ts_data(n_series, n_obs, start_id=0):
    """
    Generate a simple time series dataframe.
    """
    data = {
        "unique_id": np.repeat(np.arange(start_id, start_id + n_series), n_obs),
        "ds": np.tile(
            pd.to_datetime(pd.date_range(start="2000-01-01", periods=n_obs, freq="D")),
            n_series,
        ),
        "y": np.random.randn(n_series * n_obs),
    }
    return pd.DataFrame(data)


def test_xlstm_train_10_predict_3():
    """
    Test that xLSTM trained on 10 series can predict on 3 NEW series.
    This verifies that the model is truly univariate and not bound to the training series count/IDs.
    """
    h = 12
    input_size = 24

    # 1. Train on 10 series
    n_train_series = 10
    n_obs = 100
    train_df = make_ts_data(n_train_series, n_obs, start_id=0)

    models = [
        xLSTM(
            h=h,
            input_size=input_size,
            encoder_hidden_size=32,
            max_steps=10,
            batch_size=8,
            start_padding_enabled=True,
            scaler_type="standard",
        )
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=train_df)

    # Predict on 3 NEW series (different IDs)
    n_test_series = 3
    # start_id=100 ensures unique_ids are different from training
    test_df = make_ts_data(n_test_series, n_obs, start_id=100)

    predictions = nf.predict(df=test_df)

    assert predictions is not None

    # check shape: should have h * n_test_series rows
    assert len(predictions) == h * n_test_series

    # The column name will be xLSTM unless we set alias
    assert "xLSTM" in predictions.columns

    # check that there are NO NaN values
    nan_count = predictions["xLSTM"].isnull().sum()
    assert (
        nan_count == 0
    ), f"Found {nan_count} NaN predictions out of {len(predictions)} total predictions. xLSTM should not produce NaN values."
