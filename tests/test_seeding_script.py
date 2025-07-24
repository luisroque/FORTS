from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.seed_gcs_datasets import seed_m1_dataset


def test_seed_m1_dataset(mocker):
    """
    Tests the M1 dataset seeding function.
    - Mocks GCS to use an in-memory filesystem.
    - Mocks the gluonts dataset downloader.
    - Verifies that the function correctly processes the dummy data
      and "uploads" it to the mock GCS.
    """
    mocker.patch.dict("os.environ", {"GCS_BUCKET": "forts-ml-research-466308"})
    # 1. Mock GCS using an in-memory filesystem
    mocker.patch("scripts.seed_gcs_datasets.GCS_FS", new=MagicMock())
    mock_gcs_fs = MagicMock()
    mocker.patch("scripts.seed_gcs_datasets.GCS_FS", new=mock_gcs_fs)
    mocker.patch("scripts.seed_gcs_datasets.gcs_path_exists", return_value=False)

    # 2. Mock the gluonts downloader
    dummy_series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        name="target",
    )
    dummy_start = pd.Period("2023-01-01", "D")
    mock_gluon_dataset = MagicMock()
    mock_gluon_dataset.train = [{"target": dummy_series.values, "start": dummy_start}]
    mocker.patch(
        "scripts.seed_gcs_datasets.get_dataset", return_value=mock_gluon_dataset
    )

    # 3. Mock tempfile directory
    with patch("scripts.seed_gcs_datasets.tempfile.TemporaryDirectory") as mock_tmp:
        mock_tmp.return_value.__enter__.return_value = "."

        # 4. Run the function to be tested
        seed_m1_dataset("Monthly", ".")

    # 5. Assert that the GCS upload function was called with the correct path
    expected_gcs_path = (
        "gs://forts-ml-research-466308/forts-experiments/datasets/m1/monthly.parquet"
    )
    mock_gcs_fs.put.assert_called_once()
    call_args = mock_gcs_fs.put.call_args[0]
    # In a mocked temp dir, the local path will be relative
    assert call_args[0].endswith("m1_monthly.parquet")
    assert call_args[1] == expected_gcs_path
