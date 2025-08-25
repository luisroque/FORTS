import os
import shutil

from forts.experiments import process_results as pr
from forts.gcs_utils import gcs_delete_file, gcs_path_exists, gcs_read_csv, get_gcs_path


def test_process_results_integration_small():
    """
    Integration test that hits real GCS but limits the number of files per directory
    to keep runtime short
    """
    # Test the complete main() function with limited files and test output directory
    test_summary_dir = get_gcs_path("results_summary_test")

    print("Running process_results.main() with limited files...")
    pr.main(max_files=3, output_path=test_summary_dir)

    # Validate output exists and has expected structure
    out_csv = f"{test_summary_dir}/latex_summary.csv"
    assert gcs_path_exists(
        out_csv
    ), "Expected summary CSV was not created in GCS or fallback."

    summary_df = gcs_read_csv(out_csv)
    assert "Method" in summary_df.columns, "Summary must include a Method column."
    assert any(
        col.endswith("_MASE") for col in summary_df.columns
    ), "Summary must include MASE columns."
    assert any(
        col.endswith("_Rank") for col in summary_df.columns
    ), "Summary must include Rank columns."

    # Cleanup: remove generated file from GCS (ignore if deletion fails)
    try:
        gcs_delete_file(out_csv)
    except IOError as e:
        print(f"Cleanup failed for {out_csv}: {e}")

    print("Integration test completed successfully!")


def test_process_results_integration_with_cache():
    """
    Integration test for process_results that verifies local caching.
    It hits real GCS but limits the number of files per directory for speed.
    """
    test_summary_dir = get_gcs_path("results_summary_test")
    cache_dir = pr.LOCAL_CACHE_DIR

    # Ensure cache is clean before starting
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)

    print("Running process_results.main() with caching enabled...")
    pr.main(max_files=3, output_path=test_summary_dir)

    # --- Verification ---
    # 1. Check for GCS output
    out_csv = f"{test_summary_dir}/latex_summary.csv"
    assert gcs_path_exists(out_csv), "Expected summary CSV was not created in GCS."

    summary_df = gcs_read_csv(out_csv)
    assert "Method" in summary_df.columns, "Summary must include a Method column."
    assert any(
        col.endswith("_MASE") for col in summary_df.columns
    ), "Summary must include MASE columns."

    # 2. Check that local cache has been populated
    cached_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(cache_dir)
        for f in files
        if f.endswith(".json")
    ]
    assert len(cached_files) > 0, "Local cache directory should contain JSON files."

    # --- Cleanup ---
    try:
        gcs_delete_file(out_csv)
    except IOError as e:
        print(f"GCS cleanup failed for {out_csv}: {e}")
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    print("Integration test with caching completed successfully!")
