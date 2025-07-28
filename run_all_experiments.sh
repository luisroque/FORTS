#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Load environment variables. Assumes the correct conda env is already active.
eval "$(python -m forts.config)"

# --- Function Definitions for Each Experiment ---

run_basic_forecasting() {
    echo "Running basic forecasting..."
    python -m forts.experiments.run_pipeline --basic-forecasting
    echo "Basic forecasting complete."
}

run_transfer_learning() {
    echo "Running transfer learning..."
    python -m forts.experiments.run_pipeline --transfer-learning
    echo "Transfer learning complete."
}

run_transfer_learning_finetune() {
    echo "Running transfer learning with fine-tuning..."
    python -m forts.experiments.run_pipeline --transfer-learning --finetune
    echo "Transfer learning with fine-tuning complete."
}

run_coreset() {
    echo "Running coreset..."
    python -m forts.experiments.run_pipeline --coreset
    echo "Coreset complete."
}

run_coreset_finetune() {
    echo "Running coreset with fine-tuning..."
    python -m forts.experiments.run_pipeline --coreset --finetune
    echo "Coreset with fine-tuning complete."
}

run_all() {
    run_basic_forecasting
    run_transfer_learning
    run_transfer_learning_finetune
    run_coreset
    run_coreset_finetune
    echo "All experiments complete."
}

# --- Main Script Logic ---

usage() {
    echo "Usage: $0 [experiment_name]"
    echo "Available experiments: basic, transfer, transfer_finetune, coreset, coreset_finetune, all"
    echo "If 'all' or no experiment is specified, all experiments will be run."
    exit 1
}

# Default to running all experiments if no argument is provided.
if [ -z "$1" ]; then
    run_all
    exit 0
fi

# Run the specified experiment.
case "$1" in
    basic)
        run_basic_forecasting
        ;;
    transfer)
        run_transfer_learning
        ;;
    transfer_finetune)
        run_transfer_learning_finetune
        ;;
    coreset)
        run_coreset
        ;;
    coreset_finetune)
        run_coreset_finetune
        ;;
    all)
        run_all
        ;;
    *)
        echo "Error: Invalid experiment name '$1'."
        usage
        ;;
esac
