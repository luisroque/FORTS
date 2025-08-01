#!/bin/bash

# A script to build, push, and run an experiment on GCP Vertex AI.
#
# Usage:
#   ./run_gcp_job.sh [experiment_name] [machine_type] [additional_args...]
#
# Arguments:
#   experiment_name: The experiment to run from run_all_experiments.sh
#                    (e.g., basic, coreset, transfer_finetune). Defaults to 'all'.
#   machine_type:    The machine profile to use for the job.
#                    Options: cpu-medium, cpu-fast, gpu-medium, gpu-fast. Defaults to 'cpu-medium'.
#   additional_args: Any additional arguments to pass to the experiment script.
#                    For example, `--model AutoNHITS`.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Load environment variables. Assumes the correct conda env is already active.
eval "$(python -m forts.config)"

# Check for required variables
if [ -z "$FORTS_GCP_PROJECT_ID" ] || [ -z "$FORTS_GCP_REGION" ] || [ -z "$FORTS_DOCKER_IMAGE_NAME" ]; then
    echo "Error: Required environment variables are not set. Please check your .env file or GCP secrets."
    exit 1
fi

# --- Authenticate with GCP Service Account ---
KEY_FILE="gcp-key.json"
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Service account key file '$KEY_FILE' not found in the root directory."
    exit 1
fi
echo "--> Authenticating with GCP using service account from $KEY_FILE..."
gcloud auth activate-service-account --key-file=$KEY_FILE

echo "--> Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker "${FORTS_GCP_REGION}-docker.pkg.dev" --quiet

# --- Arguments ---
EXPERIMENT_NAME=${1:-all}
MACHINE_TYPE_KEY=${2:-cpu-medium} # Default to cpu-medium
shift 2
ADDITIONAL_ARGS=("$@")

# --- Machine Type Mapping ---
GPU_FLAG=""
case "$MACHINE_TYPE_KEY" in
    cpu-medium)
        MACHINE_SPEC="machine-type=n1-standard-8"
        ;;
    cpu-fast)
        MACHINE_SPEC="machine-type=n1-standard-16"
        ;;
    gpu-medium)
        MACHINE_SPEC="machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1"
        GPU_FLAG="--use-gpu"
        ;;
    gpu-fast)
        MACHINE_SPEC="machine-type=n1-standard-16,accelerator-type=NVIDIA_L4,accelerator-count=1"
        GPU_FLAG="--use-gpu"
        ;;
    *)
        echo "Error: Invalid machine type '$MACHINE_TYPE_KEY'."
        echo "Available types: cpu-medium, cpu-fast, gpu-medium, gpu-fast"
        exit 1
        ;;
esac

# --- Job Setup ---
MODEL_NAME_PART=""
if [[ " ${ADDITIONAL_ARGS[*]} " =~ " --model " ]]; then
    # Extract model name for a more descriptive job name
    MODEL_NAME=$(echo "${ADDITIONAL_ARGS[@]}" | awk -F'--model ' '{print $2}' | awk '{print $1}')
    if [ -n "$MODEL_NAME" ]; then
        MODEL_NAME_PART="-${MODEL_NAME}"
    fi
fi
JOB_DISPLAY_NAME="forts-${EXPERIMENT_NAME}${MODEL_NAME_PART}-${MACHINE_TYPE_KEY}-$(date +%Y%m%d-%H%M%S)"
DOCKER_IMAGE_URI="${FORTS_GCP_REGION}-docker.pkg.dev/${FORTS_GCP_PROJECT_ID}/${FORTS_AR_REPO_NAME}/${FORTS_DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG:-latest}"
SERVICE_ACCOUNT_EMAIL="vertex-ai-runner@${FORTS_GCP_PROJECT_ID}.iam.gserviceaccount.com"

# --- Main Script Logic ---
echo "--- Starting GCP Vertex AI Job Submission ---"
echo "Project ID:        ${FORTS_GCP_PROJECT_ID}"
echo "Region:            ${FORTS_GCP_REGION}"
echo "Docker Image URI:  ${DOCKER_IMAGE_URI}"
echo "Experiment to run: ${EXPERIMENT_NAME}"
echo "Machine Type:      ${MACHINE_TYPE_KEY} (${MACHINE_SPEC})"
echo "Job Display Name:  ${JOB_DISPLAY_NAME}"
echo ""

# 1. Build and Push the Docker Image
echo "--> Building and pushing the Docker image for amd64 platform..."
docker build --platform linux/amd64 \
  --build-arg FORTS_GCP_PROJECT_ID=${FORTS_GCP_PROJECT_ID} \
  -t ${DOCKER_IMAGE_URI} .
docker push ${DOCKER_IMAGE_URI}

# --- Format arguments for Vertex AI ---
ALL_JOB_ARGS=("${EXPERIMENT_NAME}")
if [[ -n "$GPU_FLAG" ]]; then
    ALL_JOB_ARGS+=("${GPU_FLAG}")
fi
if [[ ${#ADDITIONAL_ARGS[@]} -gt 0 ]]; then
    ALL_JOB_ARGS+=("${ADDITIONAL_ARGS[@]}")
fi

# Format as a comma-separated list of quoted strings for the --args flag
GCLOUD_ARGS=$(printf '"%s",' "${ALL_JOB_ARGS[@]}")
GCLOUD_ARGS=${GCLOUD_ARGS%,} # Remove trailing comma

# 2. Submit the Vertex AI Training Job
echo "--> Submitting the Vertex AI Custom Job..."
gcloud ai custom-jobs create \
  --project=${FORTS_GCP_PROJECT_ID} \
  --region=${FORTS_GCP_REGION} \
  --display-name=${JOB_DISPLAY_NAME} \
  --worker-pool-spec="${MACHINE_SPEC},replica-count=1,container-image-uri=${DOCKER_IMAGE_URI}" \
  --args=${GCLOUD_ARGS} \
  --service-account=${SERVICE_ACCOUNT_EMAIL}

echo ""
echo "--- ✅ Job submitted successfully! ---"
echo "You can monitor its progress in the GCP Console under Vertex AI > Jobs."
