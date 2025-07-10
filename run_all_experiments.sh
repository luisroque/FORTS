#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Basic Forecasting ---
echo "Running basic forecasting..."
python -m forts.experiments.run_pipeline --basic-forecasting
echo "Basic forecasting complete."

# --- Transfer Learning ---
echo "Running transfer learning..."
python -m forts.experiments.run_pipeline --transfer-learning
echo "Transfer learning complete."

# --- Transfer Learning with Fine-tuning ---
echo "Running transfer learning with fine-tuning..."
python -m forts.experiments.run_pipeline --transfer-learning --finetune
echo "Transfer learning with fine-tuning complete."

# --- Coreset ---
echo "Running coreset..."
python -m forts.experiments.run_pipeline --coreset
echo "Coreset complete."

# --- Coreset with Fine-tuning ---
echo "Running coreset with fine-tuning..."
python -m forts.experiments.run_pipeline --coreset --finetune
echo "Coreset with fine-tuning complete."

echo "All experiments complete." 