import json
import os
from unittest.mock import patch

import pandas as pd
import pytest

from forts.gcs_utils import gcs_write_csv, gcs_write_json


@pytest.fixture(autouse=True)
def cleanup_fallback_dir():
    """
    Fixture to automatically clean up the gcs_fallback directory before and after each test.
    """
    import shutil

    fallback_dir = "gcs_fallback_test"
    if os.path.exists(fallback_dir):
        shutil.rmtree(fallback_dir)
    yield
    if os.path.exists(fallback_dir):
        shutil.rmtree(fallback_dir)


def test_gcs_write_csv_fallback(caplog):
    """
    Tests that gcs_write_csv saves a file locally when GCS is unavailable.
    """
    with patch("forts.gcs_utils.get_gcs_fs", side_effect=Exception("GCS unavailable")):
        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        gcs_path = "gs://test-bucket/test.csv"
        gcs_write_csv(test_df, gcs_path)

        local_path = "gcs_fallback/test-bucket/test.csv"
        assert os.path.exists(local_path)
        assert "GCS not available. Saving to local fallback" in caplog.text

        # Verify the contents of the local file
        local_df = pd.read_csv(local_path)
        pd.testing.assert_frame_equal(test_df, local_df)


def test_gcs_write_json_fallback(caplog):
    """
    Tests that gcs_write_json saves a file locally when GCS is unavailable.
    """
    with patch("forts.gcs_utils.get_gcs_fs", side_effect=Exception("GCS unavailable")):
        test_data = {"key": "value"}
        gcs_path = "gs://test-bucket/test.json"
        gcs_write_json(test_data, gcs_path)

        local_path = "gcs_fallback/test-bucket/test.json"
        assert os.path.exists(local_path)
        assert "GCS not available. Saving to local fallback" in caplog.text

        # Verify the contents of the local file
        with open(local_path, "r") as f:
            local_data = json.load(f)
        assert test_data == local_data
