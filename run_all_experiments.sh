#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Environment variables are expected to be loaded by the calling script (run.sh)

# --- GCP Authentication ---
# When running on Vertex AI, authentication is handled automatically by the
# environment's service account. No manual authentication is needed.
# --- Function Definitions for Each Experiment ---

run_basic_forecasting() {
    echo "Running basic forecasting..."
    python -m forts.experiments.run_pipeline --basic-forecasting "$@"
    echo "Basic forecasting complete."
}

run_transfer_learning() {
    echo "Running transfer learning..."
    python -m forts.experiments.run_pipeline --transfer-learning "$@"
    echo "Transfer learning complete."
}

run_transfer_learning_finetune() {
    echo "Running transfer learning with fine-tuning..."
    python -m forts.experiments.run_pipeline --transfer-learning --finetune "$@"
    echo "Transfer learning with fine-tuning complete."
}

run_coreset() {
    echo "Running coreset..."
    python -m forts.experiments.run_pipeline --coreset "$@"
    echo "Coreset complete."
}

run_coreset_finetune() {
    echo "Running coreset with fine-tuning..."
    python -m forts.experiments.run_pipeline --coreset --finetune "$@"
    echo "Coreset with fine-tuning complete."
}

run_process_results() {
    echo "Processing results..."
    python -m forts.experiments.process_results
    echo "Results processing complete."
}

run_all() {
    run_basic_forecasting
    run_transfer_learning
    run_transfer_learning_finetune
    run_coreset
    run_coreset_finetune
    run_process_results
    echo "All experiments complete."
}


# --- Main Script Logic ---

usage() {
    echo "Usage: $0 [experiment_name] [--use-gpu] [--model MODEL_NAME]"
    echo "Available experiments: basic, transfer, transfer_finetune, coreset, coreset_finetune, process_results, all"
    echo "If 'all' or no experiment is specified, all experiments will be run."
    exit 1
}

# Default to running all experiments if no argument is provided.
if [ -z "$1" ]; then
    run_all
    exit 0
fi

experiment="$1"
shift  # Remove the experiment name from the argument list

# Run the specified experiment.
case "$experiment" in
    basic)
        run_basic_forecasting "$@"
        ;;
    transfer)
        run_transfer_learning "$@"
        ;;
    transfer_finetune)
        run_transfer_learning_finetune "$@"
        ;;
    coreset)
        run_coreset "$@"
        ;;
    coreset_finetune)
        run_coreset_finetune "$@"
        ;;
    all)
        run_all "$@"
        ;;
    process_results)
        run_process_results "$@"
        ;;
    *)
        echo "Error: Invalid experiment name '$experiment'."
        usage
        ;;
esac
