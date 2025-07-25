#!/bin/bash

# A script to build, push, and run an experiment on GCP Vertex AI.
#
# Usage:
#   ./run_gcp_job.sh [experiment_name] [machine_type]
#
# Arguments:
#   experiment_name: The experiment to run from run_all_experiments.sh
#                    (e.g., basic, coreset, transfer_finetune).
#                    Defaults to 'all'.
#   machine_type:    The machine profile to use for the job.
#                    Options: cpu-medium, cpu-fast, gpu-medium, gpu-fast.
#                    Defaults to 'cpu-medium'.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Load environment variables from .env file
if [ -f load_env.sh ]; then
    source load_env.sh
else
    echo "Error: load_env.sh not found. Please ensure you are in the project root."
    exit 1
fi

# Check for required variables
if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCP_REGION" ] || [ -z "$AR_REPO_NAME" ] || [ -z "$DOCKER_IMAGE_NAME" ]; then
    echo "Error: Required environment variables are not set. Please check your .env file."
    exit 1
fi

# Experiment to run (defaults to 'all')
EXPERIMENT_NAME=${1:-all}
MACHINE_TYPE=${2:-cpu-medium} # Default to cpu-medium
JOB_DISPLAY_NAME="forts-${EXPERIMENT_NAME}-${MACHINE_TYPE}-$(date +%Y%m%d-%H%M%S)"
DOCKER_IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO_NAME}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG:-latest}"

# --- Machine Type Configuration ---
case "$MACHINE_TYPE" in
    cpu-medium)
        WORKER_POOL_SPEC="machine-type=n1-standard-8"
        ;;
    cpu-fast)
        WORKER_POOL_SPEC="machine-type=n1-standard-16"
        ;;
    gpu-medium)
        WORKER_POOL_SPEC="machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1"
        ;;
    gpu-fast)
        WORKER_POOL_SPEC="machine-type=n1-standard-16,accelerator-type=NVIDIA_L4,accelerator-count=1"
        ;;
    *)
        echo "Error: Invalid machine type '$MACHINE_TYPE'."
        echo "Available types: cpu-medium, cpu-fast, gpu-medium, gpu-fast"
        exit 1
        ;;
esac


# --- Main Script Logic ---
echo "--- Starting GCP Vertex AI Job Submission ---"
echo "Project ID:        ${GCP_PROJECT_ID}"
echo "Region:            ${GCP_REGION}"
echo "Docker Image URI:    ${DOCKER_IMAGE_URI}"
echo "Experiment to run:   ${EXPERIMENT_NAME}"
echo "Machine Type:      ${MACHINE_TYPE} (${WORKER_POOL_SPEC})"
echo "Job Display Name:  ${JOB_DISPLAY_NAME}"
echo ""

# 1. Authenticate Docker with Artifact Registry using a Service Account Key
echo "--> Authenticating Docker with Artifact Registry using Service Account..."
cat gcp-key.json | docker login -u _json_key --password-stdin https://${GCP_REGION}-docker.pkg.dev

# 2. Build the Docker Image (will be fast due to caching)
echo "--> Building the Docker image..."
docker build -t ${DOCKER_IMAGE_URI} .

# 3. Push the Docker Image
echo "--> Pushing the Docker image to Artifact Registry..."
docker push ${DOCKER_IMAGE_URI}

# 4. Submit the Vertex AI Training Job
echo "--> Submitting the Vertex AI Custom Job..."
gcloud ai custom-jobs create \
  --project=${GCP_PROJECT_ID} \
  --region=${GCP_REGION} \
  --display-name=${JOB_DISPLAY_NAME} \
  --worker-pool-spec="${WORKER_POOL_SPEC},replica-count=1,container-image-uri=${DOCKER_IMAGE_URI}" \
  --args="${EXPERIMENT_NAME}" \
  --set-env-vars="GCS_BUCKET=$GCS_BUCKET"

echo ""
echo "--- ✅ Job submitted successfully! ---"
echo "You can monitor its progress in the GCP Console under Vertex AI > Jobs."
