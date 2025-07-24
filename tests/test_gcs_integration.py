import os
import uuid

import pandas as pd
import pytest

from forts.gcs_utils import gcs_delete_file, gcs_read_csv, gcs_write_csv, get_gcs_path


@pytest.mark.skipif(
    not os.environ.get("GCS_BUCKET"),
    reason="GCS_BUCKET environment variable not set. Skipping integration test.",
)
def test_gcs_write_and_read_integration():
    """
    A true integration test that writes a dummy CSV to the real GCS bucket
    and reads it back, verifying the content.
    This test requires the GCS_BUCKET environment variable to be set.
    """
    test_df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})

    # Use a unique filename to prevent collisions if tests run in parallel
    test_filename = f"integration_test_{uuid.uuid4()}.csv"
    test_path = f"{get_gcs_path('test-artifacts')}/{test_filename}"

    try:
        # 1. Write the DataFrame to GCS
        print(f"Writing test file to: {test_path}")
        gcs_write_csv(test_df, test_path)

        # 2. Read it back from GCS
        print(f"Reading test file from: {test_path}")
        read_df = gcs_read_csv(test_path)

        # 3. Verify the content
        pd.testing.assert_frame_equal(test_df, read_df)
        print("Data verification successful.")

    finally:
        # 4. Clean up the test file from GCS
        print(f"Cleaning up test file: {test_path}")
        gcs_delete_file(test_path)
