# FORTS: A Framework to Evaluate Domain Transferability for Time Series Forecasting

We introduce FORTS, a novel evaluation framework designed to assess the generalization and transfer capabilities of time series forecasting algorithms. Recent interest in time series foundation models (TSFMs) has highlighted the need for systematic evaluation under distributional shift, but these challenges apply equally to a broader class of forecasting algorithms. Motivated by this gap, FORTS defines four complementary evaluation regimes: (i) full-shot, (ii) in-domain zero-shot, (iii) single-source out-of-domain zero-shot, and (iv) multi-source out-of-domain zero-shot. We simulate these regimes using a diverse suite of ten benchmark datasets covering over 60 million observations across domains, frequencies, and temporal dynamics. FORTS enables rigorous assessment of algorithm robustness and inductive bias properties that traditional evaluation methods miss. Our experiments across eight forecasting methods highlight discrepancies between generalization and accuracy, revealing that many algorithms fail to maintain performance under cross-domain transfer. We release all code, datasets, and configurations to promote transparency and standardization in time series forecasting evaluation.

## FORTS Framework

`FORTS` formalizes the above in four evaluation settings that systematically vary the domain overlap between training and test sets. This allows a more systematic and controlled evaluation of how a TSFM performs in terms of *domain transferability*.

#### Full-shot evaluation
This regime reflects the standard forecasting setup (single domain): each time series $Y^{(i)} \in \mathcal{D}^{(j)}$ is split temporally into training and test segments. Specifically, for a fixed cutoff time $t^*$, we define:
```math
\mathcal{Y}_{\text{train}} \subset \mathcal{D}^{(j)}, \quad \mathcal{Y}_{\text{test}} \subset \mathcal{D}^{(j)}
```
for all $i \in \{1, \dots, n_j\}$. The model is trained and evaluated on different segments of the same series. This setup tests the ability of the model to extrapolate in time but does not evaluate its capacity to generalize across unseen series and across domains.

#### In-domain transfer learning
In this regime, the model is trained and evaluated on disjoint subsets of time series from the same domain $\mathcal{S}_j$. Specifically, let the dataset $\mathcal{D}^{(j)}$ be partitioned into non-overlapping subsets:
```math
\mathcal{Y}_{\text{train}} \subset \mathcal{D}^{(j)}, \quad \mathcal{Y}_{\text{test}} \subset \mathcal{D}^{(j)} \setminus \mathcal{Y}_{\text{train}}
```
with no overlap in series. This setup assesses the ability of the model to generalize from a subset of time series to unseen series within the same domain. It simulates settings such as forecasting for new products, clients, or sensors not seen during training.

#### Single-source out-of-domain transfer learning
This setting evaluates domain transfer from a single source domain $\mathcal{S}_j$ to a distinct target domain $\mathcal{S}_k$. The model is trained on all time series from dataset $\mathcal{D}^{(j)}$ and tested on all time series from $\mathcal{D}^{(k)}$, with $j \neq k$:
```math
\mathcal{Y}_{\text{train}} = \mathcal{D}^{(j)}, \quad \mathcal{Y}_{\text{test}} = \mathcal{D}^{(k)}
```
This regime reflects real-world deployment scenarios in which a forecasting model must generalize to entirely new environments without access to domain-specific retraining data.

#### Multiple-source out-of-domain transfer learning
The most challenging regime mirrors the goal of TSFMs: train once on diverse datasets and generalize zero-shot to unseen domains. Given a collection of datasets $\{ \mathcal{D}^{(1)}, \dots, \mathcal{D}^{(M)} \}$, we perform leave-one-domain-out evaluation. For a held-out domain $\mathcal{S}_k$, we define:
```math
\mathcal{Y}_{\text{train}} = \bigcup_{j \neq k} \mathcal{D}^{(j)}, \quad
\mathcal{Y}_{\text{test}} = \mathcal{D}^{(k)}
```
This setting tests whether pre-training on multiple diverse domains results in robust representations that transfer with or without fine-tuning.

Each regime in `FORTS` is designed to isolate a specific axis of generalization. *Full-shot* focuses on temporal extrapolation. *In-domain transfer learning* evaluates cross-series generalization within structurally similar environments. *Single-source out-of-domain transfer learning* tests adaptation under a strong domain shift. *Multi-source out-of-domain transfer learning* demonstrates foundation-level generalization.

## Key Research Questions

We focus on four key research questions that guide the evaluation:

- **Q1**: How well do they generalize to unseen time series within the same domain?
- **Q2**: Can they transfer effectively to new domains when trained on a single source dataset?
- **Q3**: Can they generalize across domains when trained on multiple source datasets, a common requirement for TSFMs?

## Project Structure

```
FORTS/
├── assets/                # Plots, data splits, and model weights
├── forts/                 # Core package
│   ├── data_pipeline/     # Data loading, splitting, and feature engineering
│   ├── experiments/       # Scripts for running experiments
│   ├── load_data/         # Data loaders for different datasets
│   ├── metrics/           # Evaluation metrics and pipelines
│   ├── model_pipeline/    # FORTS and baseline model implementations
│   └── visualization/     # Plotting utilities
├── tests/                 # Unit and integration tests
├── scripts/               # Utility scripts
├── terraform/             # Terraform configurations for cloud infrastructure
├── requirements.txt
└── README.md
```

## Installation

To set up the environment, please install the required packages:

```bash
pip install -r requirements.txt
```

### Datasets

| Dataset | Frequency | # time series | # observations | Horizon ($H$) |
|:---|:---|:---|:---|:---|
| Tourism | Monthly | 366 | 109,661 | 24 |
| M1 | Monthly | 617 | 44,884 | 24 |
| | Quarterly | 203 | 8,323 | 8 |
| M3 | Monthly | 1,428 | 167,475 | 24 |
| | Quarterly | 756 | 37,026 | 8 |
| | Yearly | 645 | 18,333 | 4 |
| M4 | Monthly | 48,000 | 11,246,400 | 24 |
| | Quarterly | 23,999 | 2,398,978 | 8 |
| Traffic | Daily | 207 | 75,762 | 30 |
| M5 | Daily | 30,490 | 47,624,269 | 30 |
| **Total** | | **106,711** | **61,777,111** | -- |

## Running Experiments

> To reproduce key experiments with **FORTS**, use the appropriate command below based on the selected evaluation regime.

#### Full-shot evaluation

This regime evaluates models on their ability to forecast future values based on past data from the same time series.

```bash
./run_all_experiments.sh basic --use-gpu
```

#### In-domain & Single-source out-of-domain transfer learning

This setup assesses the ability of the model to generalize from a subset of time series to unseen series, both within the same domain and to a different target domain.

```bash
./run_all_experiments.sh transfer --use-gpu
```

#### Multiple-source out-of-domain transfer learning

This setup tests a model's ability to generalize to entirely new datasets by training on a "coreset" of multiple source datasets and evaluating on a held-out target dataset.

```bash
./run_all_experiments.sh coreset --use-gpu
```

You can also run all experiments sequentially:

```bash
./run_all_experiments.sh all --use-gpu
```

## Citation

It will be added in the future.
```
