import pandas as pd
import pytest

from forts.gcs_utils import gcs_delete_file, gcs_read_csv, gcs_write_csv, get_gcs_path


@pytest.fixture
def gcs_test_path():
    """Fixture for a test GCS path that will be cleaned up."""
    path = get_gcs_path("test_data.csv")
    yield path
    # Cleanup: delete the file after the test
    gcs_delete_file(path)


def test_gcs_write_and_read_csv(gcs_test_path):
    """Test writing to and reading from GCS."""
    test_df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})

    # 1. Write the DataFrame to GCS
    gcs_write_csv(test_df, gcs_test_path)

    # 2. Read it back from GCS
    read_df = gcs_read_csv(gcs_test_path)

    # 3. Verify the content
    pd.testing.assert_frame_equal(test_df, read_df)
