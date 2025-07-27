#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# You can change the experiment name to test different pipelines.
EXPERIMENT_NAME="transfer_finetune"
IMAGE_NAME="forts-local-test"

# --- Script Logic ---
echo "--- Starting Local Docker Test ---"

# 1. Source environment variables from .env file
if [ -f .env ]; then
    echo "Sourcing environment variables from .env file..."
    source .env
else
    echo "Error: .env file not found. Please create one from .env.example."
    exit 1
fi

# 2. Check if GCS_BUCKET is set
if [ -z "$GCS_BUCKET" ]; then
    echo "Error: GCS_BUCKET environment variable is not set in the .env file."
    exit 1
fi

# 3. Build the Docker image for amd64 platform
echo "Building Docker image for amd64: $IMAGE_NAME..."
docker build --platform linux/amd64 -t "$IMAGE_NAME" .

# 4. Run the container with the environment variable and credentials
echo "Running Docker container with experiment: $EXPERIMENT_NAME..."
docker run \
  -e "GCS_BUCKET=${GCS_BUCKET}" \
  -v "${HOME}/.config/gcloud:/root/.config/gcloud" \
  "$IMAGE_NAME" \
  "$EXPERIMENT_NAME"

echo "--- ✅ Local test completed successfully! ---"
