import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.api_core.exceptions import ClientError
from google.cloud import secretmanager

# --- Environment Variable and Secret Configuration ---

# GCP Secret Manager names (also used as environment variable keys)
GCP_PROJECT_ID = "FORTS_GCP_PROJECT_ID"
GCP_REGION = "FORTS_GCP_REGION"
AR_REPO_NAME = "FORTS_AR_REPO_NAME"
DOCKER_IMAGE_NAME = "FORTS_DOCKER_IMAGE_NAME"
GCS_BUCKET_NAME = "FORTS_GCS_BUCKET_NAME"

# List of all configuration keys
ALL_CONFIG_KEYS = [
    GCP_PROJECT_ID,
    GCP_REGION,
    AR_REPO_NAME,
    DOCKER_IMAGE_NAME,
    GCS_BUCKET_NAME,
]

# --- Helper Functions ---


def get_gcp_secret(secret_id: str, project_id: str) -> Optional[str]:
    """
    Retrieves a secret from Google Cloud Secret Manager.
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except ClientError as e:
        print(f"Could not access secret {secret_id}: {e}")
        return None


# --- Main Configuration Loading ---


def load_config():
    """
    Loads configuration from a .env file or GCP Secret Manager.
    """
    # Try loading from .env file first (for local development)
    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("Loaded configuration from .env file.")
        return

    # If no .env file, try loading from GCP Secret Manager (for cloud runs)
    gcp_project_id = os.getenv(GCP_PROJECT_ID)
    if not gcp_project_id:
        # Also check the a different name for the project id
        gcp_project_id = os.getenv("GCP_PROJECT_ID")

    if gcp_project_id:
        print(
            f"Loading configuration from GCP Secret Manager for project {gcp_project_id}..."
        )
        for key in ALL_CONFIG_KEYS:
            secret_value = get_gcp_secret(key, gcp_project_id)
            if secret_value:
                os.environ[key] = secret_value
    else:
        print("Warning: GCP_PROJECT_ID not set. Unable to load secrets.")


# Load configuration when this module is imported
load_config()
