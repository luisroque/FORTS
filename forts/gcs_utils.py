import logging
import os

import pandas as pd
from dotenv import load_dotenv
from gcsfs import GCSFileSystem

import forts.config

# Load environment variables from .env file
load_dotenv()


# --- GCS Configuration ---

GCS_BUCKET = os.getenv(forts.config.GCS_BUCKET_NAME)
if not GCS_BUCKET:
    print("Warning: FORTS_GCS_BUCKET_NAME environment variable not set.")


# The base path within the bucket for all experiment outputs.
# This can be overridden by setting the GCS_EXPERIMENT_ROOT environment variable.
GCS_EXPERIMENT_ROOT = os.environ.get("GCS_EXPERIMENT_ROOT", "forts-experiments")

# --- GCS File System Singleton ---

# Use a singleton pattern to ensure we only have one GCSFileSystem instance.
_gcs_fs = None


def get_gcs_fs():
    """
    Returns a singleton instance of the GCSFileSystem.
    Authentication is handled automatically by the gcsfs library. For server-based
    authentication, ensure the `GOOGLE_APPLICATION_CREDENTIALS` environment
    variable is set to the path of your service account key file.
    """
    global _gcs_fs
    if _gcs_fs is None:
        service_account_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if service_account_file:
            logging.info(
                f"Using service account credentials from {service_account_file}"
            )
        else:
            logging.warning(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
                "Falling back to default authentication (e.g., gcloud ADC)."
            )
        _gcs_fs = GCSFileSystem()
    return _gcs_fs


# --- Path Management ---


def get_gcs_base_path():
    """
    Returns the full GCS path for the experiment root.
    """
    return f"gs://{GCS_BUCKET}/{GCS_EXPERIMENT_ROOT}"


def get_gcs_path(subfolder: str):
    """
    Constructs a full GCS path for a given subfolder.
    """
    return f"{get_gcs_base_path()}/{subfolder}"


def get_model_weights_path():
    """
    Returns the GCS path for storing model weights.
    """
    return get_gcs_path("model_weights")


def get_datasets_path():
    """
    Returns the GCS path for datasets.
    """
    return get_gcs_path("datasets")


# --- File I/O Operations ---


def gcs_path_exists(gcs_path: str) -> bool:
    """
    Checks if a path exists in GCS or in the local fallback.
    """
    local_path = _get_local_fallback_path(gcs_path)
    if os.path.exists(local_path):
        return True

    try:
        fs = get_gcs_fs()
        return fs.exists(gcs_path)
    except Exception as e:
        logging.warning(
            f"Failed to check GCS path {gcs_path} existence, returning False. Error: {e}"
        )
        return False


def gcs_list_files(gcs_path: str, extension: str = ".json") -> list:
    """
    Lists all files in a GCS directory with a given extension.
    """
    fs = get_gcs_fs()
    if not fs.exists(gcs_path):
        return []
    return [f"gs://{f}" for f in fs.glob(f"{gcs_path}/*{extension}")]


def _get_local_fallback_path(gcs_path: str) -> str:
    """
    Converts a GCS path to its corresponding local fallback path.
    """
    fallback_dir = os.environ.get("FORTS_FALLBACK_DIR", "gcs_fallback")
    return os.path.join(fallback_dir, gcs_path.replace("gs://", ""))


def _read_from_local_fallback(local_path: str, reader_func):
    """
    Reads data from the local fallback path using the provided reader function.
    """
    logging.info(f"Reading from local fallback: {local_path}")
    return reader_func(local_path)


def _gcs_read_with_fallback(gcs_path: str, gcs_reader_func, local_reader_func):
    """
    Tries to read from the local fallback first, then from GCS.
    """
    local_path = _get_local_fallback_path(gcs_path)
    if os.path.exists(local_path):
        try:
            return _read_from_local_fallback(local_path, local_reader_func)
        except Exception as e:
            logging.warning(
                f"Local fallback file found at {local_path}, but failed to read. "
                f"Attempting to read from GCS. Error: {e}"
            )

    try:
        return gcs_reader_func(gcs_path)
    except Exception as e:
        logging.error(f"Failed to read from GCS at {gcs_path}: {e}")
        raise


def gcs_read_csv(gcs_path: str) -> pd.DataFrame:
    """
    Reads a CSV file from GCS into a pandas DataFrame, with a local fallback.
    """

    def gcs_reader(path):
        with get_gcs_fs().open(path, "r") as f:
            return pd.read_csv(f)

    def local_reader(path):
        return pd.read_csv(path)

    return _gcs_read_with_fallback(gcs_path, gcs_reader, local_reader)


def gcs_read_parquet(gcs_path: str) -> pd.DataFrame:
    """
    Reads a Parquet file from GCS into a pandas DataFrame, with a local fallback.
    """

    def gcs_reader(path):
        with get_gcs_fs().open(path, "rb") as f:
            return pd.read_parquet(f)

    def local_reader(path):
        return pd.read_parquet(path)

    return _gcs_read_with_fallback(gcs_path, gcs_reader, local_reader)


def gcs_open_with_fallback(gcs_path: str, mode: str = "rb"):
    """
    Opens a file from GCS or the local fallback, returning a file-like object.
    It's the caller's responsibility to close the file handle.
    """
    local_path = _get_local_fallback_path(gcs_path)
    if os.path.exists(local_path):
        logging.info(f"Opening local fallback file: {local_path}")
        return open(local_path, mode)

    try:
        logging.info(f"Opening GCS file: {gcs_path}")
        fs = get_gcs_fs()
        return fs.open(gcs_path, mode)
    except Exception as e:
        logging.error(f"Failed to open GCS file at {gcs_path}: {e}")
        raise


def _save_to_local_fallback(gcs_path: str, data, writer_func):
    """
    Saves data to a local fallback directory when GCS is unavailable.
    """
    fallback_dir = os.environ.get("FORTS_FALLBACK_DIR", "gcs_fallback")
    local_path = os.path.join(fallback_dir, gcs_path.replace("gs://", ""))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    logging.warning(f"GCS not available. Saving to local fallback: {local_path}")
    writer_func(data, local_path)


def gcs_write_csv(df: pd.DataFrame, gcs_path: str):
    """
    Writes a pandas DataFrame to a CSV file in GCS.
    If GCS is unavailable, it saves the file to a local fallback directory.
    """
    try:
        with get_gcs_fs().open(gcs_path, "w") as f:
            df.to_csv(f, index=False)
        logging.info(f"Successfully wrote to GCS: {gcs_path}")
    except Exception as e:
        logging.error(f"Failed to write to GCS at {gcs_path}: {e}")
        _save_to_local_fallback(gcs_path, df, lambda d, p: d.to_csv(p, index=False))


def gcs_read_json(gcs_path: str) -> dict:
    """
    Reads a JSON file from GCS, with a local fallback.
    """
    import json

    def gcs_reader(path):
        with get_gcs_fs().open(path, "r") as f:
            return json.load(f)

    def local_reader(path):
        with open(path, "r") as f:
            return json.load(f)

    return _gcs_read_with_fallback(gcs_path, gcs_reader, local_reader)


def gcs_write_json(data: dict, gcs_path: str):
    """
    Writes a dictionary to a JSON file in GCS.
    If GCS is unavailable, it saves the file to a local fallback directory.
    """
    import json

    try:
        with get_gcs_fs().open(gcs_path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"Successfully wrote to GCS: {gcs_path}")
    except Exception as e:
        logging.error(f"Failed to write to GCS at {gcs_path}: {e}")
        _save_to_local_fallback(
            gcs_path, data, lambda d, p: json.dump(d, open(p, "w"), indent=4)
        )


def gcs_delete_directory(gcs_path: str):
    """
    Deletes a directory and its contents from GCS if it exists.
    """
    fs = get_gcs_fs()
    if fs.exists(gcs_path):
        fs.rm(gcs_path, recursive=True)


def gcs_delete_file(gcs_path: str):
    """
    Deletes a file from GCS if it exists.
    """
    fs = get_gcs_fs()
    if fs.exists(gcs_path):
        fs.rm(gcs_path)
