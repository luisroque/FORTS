import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models.timemixer import TimeMixer


class UnivariateTimeMixer(TimeMixer):
    SAMPLING_TYPE = "windows"
    MULTIVARIATE = False

    def __init__(self, n_series, **kwargs):
        # Force n_series to 1 for univariate mode internals
        # Force valid_batch_size=1 to avoid NaN predictions bug in neuralforecast
        # when processing batched predictions with MULTIVARIATE=False
        kwargs["valid_batch_size"] = 1
        kwargs["limit_val_batches"] = 64
        super().__init__(n_series=1, **kwargs)
        self.enc_in = 1
        self.c_out = 1


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


def test_timemixer_train_10_predict_3():
    """
    Test that TimeMixer trained on 10 series can predict on 3 NEW series.
    This verifies that the model is truly univariate and not bound to the training series count/IDs.
    """
    h = 12
    input_size = 24

    # 1. Train on 10 series
    n_train_series = 10
    n_obs = 100
    train_df = make_ts_data(n_train_series, n_obs, start_id=0)

    models = [
        UnivariateTimeMixer(
            h=h,
            input_size=input_size,
            n_series=n_train_series,  # Passed to __init__ but overridden by subclass
            d_model=32,
            d_ff=64,
            e_layers=1,
            max_steps=10,
            batch_size=8,
        )
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=train_df)

    # 2. Predict on 3 NEW series (different IDs)
    n_test_series = 3
    # start_id=100 ensures unique_ids are different from training
    test_df = make_ts_data(n_test_series, n_obs, start_id=100)

    # NeuralForecast.predict uses the passed df to generate predictions
    predictions = nf.predict(df=test_df)

    # Drop NaN predictions as requested
    predictions = predictions.dropna()

    assert predictions is not None
    # Check shape: h * n_test_series
    assert len(predictions) == h * n_test_series

    # The column name will be UnivariateTimeMixer unless we set alias
    assert "UnivariateTimeMixer" in predictions.columns
    assert not predictions["UnivariateTimeMixer"].isnull().values.any()
