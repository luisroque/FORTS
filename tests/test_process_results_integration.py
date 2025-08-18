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
