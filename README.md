# FORTS

> **FORTS: An Evaluation Framework for Time Series Foundation Models**

## Overview

We introduce FORTS, a novel evaluation framework designed to assess the generalization and transfer capabilities of time series foundation models (TSFMs). While recent advances in TSFMs demonstrate strong in-domain forecasting performance, consistent and rigorous evaluation under distributional shift remains lacking. To ground our evaluation, we introduce a taxonomy that separates TSFMs into three components (algorithm, data, and model) and incorporates a domain-aware axis that distinguishes between training and testing distributions. Building on this formulation, FORTS defines four complementary evaluation regimes: (i) full-shot, (ii) in-domain zero-shot, (iii) single-source out-of-domain zero-shot, and (iv) multi-source out-of-domain zero-shot. These regimes are instantiated on a diverse suite of ten benchmark datasets covering over 60 million observations across domains, frequencies, and temporal dynamics. We demonstrate how FORTS surfaces model robustness and inductive bias properties that are obscured by traditional evaluations. Our experiments across eight models highlight discrepancies between generalization and accuracy, revealing that several recent TSFMs fail to maintain performance under cross-domain transfer. We release all code, datasets, and configurations to promote transparency and standardization in TSFM evaluation.

## Project Structure

```
FORTS/
├── assets/                # Plots, data splits, and model weights
├── lightning_logs/        # Logs from PyTorch Lightning
├── forts/             # Core package
│   ├── data_pipeline/       # Data loading, splitting, and feature engineering
│   ├── experiments/         # Scripts for running experiments
│   ├── metrics/             # Evaluation metrics and pipelines
│   ├── model_pipeline/      # FORTS and baseline model implementations
│   ├── visualization/       # Plotting utilities
│   └── ...
├── requirements.txt       # Project dependencies
└── README.md
```

## Quickstart

### Installation

To set up the environment, please install the required packages:

```bash
pip install -r requirements.txt
```

### Reproducing Main Results

Below is the main table from the paper, comparing FORTS to other baselines across four evaluation regimes:

| **Model**       | **In-Domain** | **Out-of-Domain** | **Coreset (Leave-One-Out)** | **Basic Forecasting** |
|:----------------|:-------------:|:-----------------:|:---------------------------:|:---------------------:|
| **FORTS**     | 2.7         | 1.0       | 1.33             | 4.6  | 1.41             | 5.8  | **1.93**             | **3.8**  | **1.55**             | **2.4**  |
| PatchTST        | **0.8**     | 1.4       | 2.78             | 3.7  | 1.76             | 3.9  | 2.21                 | 4.6  | 1.70                 | 3.4  |
| NHITS           | 1.2         | **0.5**   | **1.09**         | **2.3**  | **1.11**         | **2.3**  | _2.01_               | _4.2_  | _1.61_               | _2.7_  |
| TFT             | _1.1_       | _0.6_     | _1.27_           | _2.4_  | _1.18_           | _2.4_  | 2.18                 | 5.8  | 1.71                 | 3.0  |
| TSMixer         | 2.1         | 1.7       | 1.89             | 4.9  | 1.83             | 5.0  | 3.25                 | 6.9  | 2.15                 | 4.1  |
| iTransformer    | 1.7         | 1.6       | 1.85             | 5.0  | 1.86             | 5.2  | 2.93                 | 7.3  | 2.01                 | 4.3  |
| KAN             | 2.2         | 1.8       | 2.01             | 5.4  | 1.95             | 5.7  | 3.45                 | 7.8  | 2.30                 | 4.7  |

*The table shows the average rank of each model across all datasets for a given evaluation regime.*

### Datasets

FORTS is evaluated on 10 publicly available datasets including:

- **M-Competitions**: M1, M3, M4, M5
- **Long-Horizon**: Tourism, Traffic, Weather
- **Other Benchmarks**: Labour, Wiki2, ECL

### Running Experiments

> To reproduce key experiments with **FORTS**, use the appropriate command below based on the selected evaluation regime.

#### Basic Forecasting

This regime evaluates models on their ability to forecast future values based on past data from the same time series.

```bash
python forts/experiments/run_pipeline.py --use-gpu --basic-forecasting
```

#### Coreset (Leave-One-Out)

This setup tests a model's ability to generalize to entirely new datasets by training on a "coreset" of multiple source datasets and evaluating on a held-out target dataset.

```bash
python forts/experiments/run_pipeline.py --use-gpu --coreset
```

#### In-Domain and Out-of-Domain (Transfer Learning)

These regimes assess how well a model trained on one dataset (source) performs on another (target).

```bash
python forts/experiments/run_pipeline.py --use-gpu --transfer-learning
```