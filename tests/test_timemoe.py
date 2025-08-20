import numpy as np
import pandas as pd
import pytest
from neuralforecast import NeuralForecast

from forts.model_pipeline.timemoe import TimeMOE


@pytest.fixture
def simple_ts_data():
    """
    Generate a simple time series dataframe for testing.
    2 series, 100 observations each.
    """
    n_series = 2
    n_obs = 100
    data = {
        "unique_id": np.repeat(np.arange(n_series), n_obs),
        "ds": np.tile(
            pd.to_datetime(pd.date_range(start="2000-01-01", periods=n_obs, freq="D")),
            n_series,
        ),
        "y": np.random.randn(n_series * n_obs),
    }
    return pd.DataFrame(data)


def test_timemoe_runs(simple_ts_data):
    """
    Test that the TimeMOE model can be instantiated, trained, and produce predictions.
    """
    h = 12
    input_size = 24
    models = [
        TimeMOE(
            h=h,
            input_size=input_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_steps=10,  # Keep it short for a test
            batch_size=8,
            windows_batch_size=32,
        )
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=simple_ts_data)
    predictions = nf.predict()

    assert predictions is not None
    assert len(predictions) == h * len(simple_ts_data["unique_id"].unique())
    assert "TimeMOE" in predictions.columns
