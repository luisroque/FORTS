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
    """
    global _gcs_fs
    if _gcs_fs is None:
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
    Checks if a path exists in GCS.
    """
    fs = get_gcs_fs()
    return fs.exists(gcs_path)


def gcs_list_files(gcs_path: str, extension: str = ".json") -> list:
    """
    Lists all files in a GCS directory with a given extension.
    """
    fs = get_gcs_fs()
    if not fs.exists(gcs_path):
        return []
    return [f"gs://{f}" for f in fs.glob(f"{gcs_path}/*{extension}")]


def gcs_read_csv(gcs_path: str) -> pd.DataFrame:
    """
    Reads a CSV file from GCS into a pandas DataFrame.
    """
    with get_gcs_fs().open(gcs_path, "r") as f:
        return pd.read_csv(f)


def gcs_write_csv(df: pd.DataFrame, gcs_path: str):
    """
    Writes a pandas DataFrame to a CSV file in GCS.
    """
    with get_gcs_fs().open(gcs_path, "w") as f:
        df.to_csv(f, index=False)


def gcs_read_json(gcs_path: str) -> dict:
    """
    Reads a JSON file from GCS.
    """
    import json

    with get_gcs_fs().open(gcs_path, "r") as f:
        return json.load(f)


def gcs_write_json(data: dict, gcs_path: str):
    """
    Writes a dictionary to a JSON file in GCS.
    """
    import json

    with get_gcs_fs().open(gcs_path, "w") as f:
        json.dump(data, f, indent=4)


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
