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
# Load environment variables. Assumes the correct conda env is already active.
eval "$(python -m forts.config)"

# Check for required variables
if [ -z "$FORTS_GCP_PROJECT_ID" ] || [ -z "$FORTS_GCP_REGION" ] || [ -z "$FORTS_DOCKER_IMAGE_NAME" ]; then
    echo "Error: Required environment variables are not set. Please check your .env file or GCP secrets."
    exit 1
fi

# --- Arguments ---
EXPERIMENT_NAME=${1:-all}
MACHINE_TYPE_KEY=${2:-cpu-medium} # Default to cpu-medium

# --- Machine Type Mapping ---
case "$MACHINE_TYPE_KEY" in
    cpu-medium)
        MACHINE_TYPE="n1-standard-8"
        ;;
    cpu-fast)
        MACHINE_TYPE="n1-standard-16"
        ;;
    gpu-medium)
        MACHINE_TYPE="n1-standard-8"
        ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
        ACCELERATOR_COUNT=1
        ;;
    gpu-fast)
        MACHINE_TYPE="n1-standard-16"
        ACCELERATOR_TYPE="NVIDIA_L4"
        ACCELERATOR_COUNT=1
        ;;
    *)
        echo "Error: Invalid machine type '$MACHINE_TYPE_KEY'."
        echo "Available types: cpu-medium, cpu-fast, gpu-medium, gpu-fast"
        exit 1
        ;;
esac

# --- Job Setup ---
JOB_DISPLAY_NAME="forts-${EXPERIMENT_NAME}-${MACHINE_TYPE_KEY}-$(date +%Y%m%d-%H%M%S)"
DOCKER_IMAGE_URI="${FORTS_GCP_REGION}-docker.pkg.dev/${FORTS_GCP_PROJECT_ID}/${FORTS_AR_REPO_NAME}/${FORTS_DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG:-latest}"
TEMP_JOB_CONFIG="gcp_job.yaml"

# --- Main Script Logic ---
echo "--- Starting GCP Vertex AI Job Submission ---"
echo "Project ID:        ${FORTS_GCP_PROJECT_ID}"
echo "Region:            ${FORTS_GCP_REGION}"
echo "Docker Image URI:  ${DOCKER_IMAGE_URI}"
echo "Experiment to run: ${EXPERIMENT_NAME}"
echo "Machine Type:      ${MACHINE_TYPE_KEY} (${MACHINE_TYPE})"
echo "Job Display Name:  ${JOB_DISPLAY_NAME}"
echo ""

# 1. Build and Push the Docker Image
echo "--> Building and pushing the Docker image for amd64 platform..."
docker build --platform linux/amd64 -t ${DOCKER_IMAGE_URI} .
docker push ${DOCKER_IMAGE_URI}

# 2. Create the Job Configuration File from the Template
echo "--> Creating job configuration file from template..."
cp gcp_job_template.yaml ${TEMP_JOB_CONFIG}

sed -i.bak "s|MACHINE_TYPE|${MACHINE_TYPE}|g" ${TEMP_JOB_CONFIG}
sed -i.bak "s|DOCKER_IMAGE_URI|${DOCKER_IMAGE_URI}|g" ${TEMP_JOB_CONFIG}
sed -i.bak "s|EXPERIMENT_NAME|${EXPERIMENT_NAME}|g" ${TEMP_JOB_CONFIG}
sed -i.bak "s|GCP_PROJECT_ID|${FORTS_GCP_PROJECT_ID}|g" ${TEMP_JOB_CONFIG}

# Add accelerator config if specified, otherwise remove the placeholder
if [ -n "$ACCELERATOR_TYPE" ]; then
    ACCELERATOR_YAML="    acceleratorType: ${ACCELERATOR_TYPE}\n    acceleratorCount: ${ACCELERATOR_COUNT}"
    # Use awk for safer multiline replacement
    awk -v var="$ACCELERATOR_YAML" '{gsub(/# ACCELERATOR_CONFIG_PLACEHOLDER/, var)}1' ${TEMP_JOB_CONFIG} > tmp && mv tmp ${TEMP_JOB_CONFIG}
else
    # Remove the placeholder line if no accelerator is specified
    sed -i.bak "/# ACCELERATOR_CONFIG_PLACEHOLDER/d" ${TEMP_JOB_CONFIG}
fi

# 3. Submit the Vertex AI Training Job
echo "--> Submitting the Vertex AI Custom Job from ${TEMP_JOB_CONFIG}..."
gcloud ai custom-jobs create \
  --project=${FORTS_GCP_PROJECT_ID} \
  --region=${FORTS_GCP_REGION} \
  --display-name=${JOB_DISPLAY_NAME} \
  --config=${TEMP_JOB_CONFIG}

# 4. Cleanup
rm ${TEMP_JOB_CONFIG}
rm ${TEMP_JOB_CONFIG}.bak

echo ""
echo "--- ✅ Job submitted successfully! ---"
echo "You can monitor its progress in the GCP Console under Vertex AI > Jobs."
