import logging
import os
from unittest.mock import patch

import pytest

import forts.config
from forts import gcs_utils


@pytest.fixture(autouse=True)
def reset_gcs_fs_singleton():
    """
    Fixture to automatically reset the GCSFileSystem singleton before each test.
    This prevents state from leaking between tests.
    """
    gcs_utils._gcs_fs = None


def test_gcs_fs_with_service_account_env_set(caplog):
    """
    Tests that the correct log message is emitted when
    GOOGLE_APPLICATION_CREDENTIALS is set.
    """
    test_creds_file = "gcp-key.json"
    if not os.path.exists(test_creds_file):
        pytest.skip(f"{test_creds_file} not found, skipping unit test.")

    with (
        patch("gcsfs.GCSFileSystem"),
        patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": test_creds_file}),
    ):
        with caplog.at_level(logging.INFO):
            gcs_utils.get_gcs_fs()
            assert (
                f"Using service account credentials from {test_creds_file}"
                in caplog.text
            )


def test_gcs_fs_without_service_account_env(caplog):
    """
    Tests that a warning is logged when GOOGLE_APPLICATION_CREDENTIALS is not set.
    """
    with patch("gcsfs.GCSFileSystem"), patch.dict(os.environ, {}, clear=True):
        with caplog.at_level(logging.WARNING):
            gcs_utils.get_gcs_fs()
            assert (
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
                in caplog.text
            )


@pytest.mark.integration
def test_gcs_authentication_with_key_file():
    """
    An integration test that uses a service account key file to authenticate and checks for bucket access
    by calling a function in gcs_utils.
    This test is skipped if gcp-key.json is not found or the GCS bucket is not configured.
    """
    key_file = "gcp-key.json"
    gcs_bucket = os.getenv(forts.config.GCS_BUCKET_NAME)

    if not os.path.exists(key_file):
        pytest.skip("gcp-key.json not found, skipping integration test.")

    if not gcs_bucket:
        pytest.skip(
            f"{forts.config.GCS_BUCKET_NAME} env var not set, skipping integration test."
        )

    with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": key_file}):
        # Reset the singleton to force re-authentication
        gcs_utils._gcs_fs = None

        bucket_path = f"gs://{gcs_bucket}"
        try:
            # Check if the bucket exists by trying to access it via a gcs_utils function
            assert gcs_utils.gcs_path_exists(
                bucket_path
            ), f"Bucket {bucket_path} does not exist or is not accessible."
        except Exception as e:
            pytest.fail(
                f"GCS authentication/access failed with {key_file} for bucket {bucket_path}: {e}"
            )
