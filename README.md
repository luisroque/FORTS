# FORTS: An Evaluation Framework for Transfer Learning in Time Series Forecasting

`FORTS` is a novel evaluation framework for assessing the generalization and transfer capabilities of time series forecasting architectures. Recent time series models achieve strong in-domain performance, but rigorous evaluation of their generalization capabilities remains limited. `FORTS` evaluates forecasting models under four complementary regimes that progressively test within-series forecasting, in-domain generalization, and cross-domain transfer to unseen distributions, enabling richer comparisons across architectures and practical guidance for model selection in real-world applications.

We illustrate the framework by evaluating **ten SOTA methods** on **ten benchmark datasets** encompassing more than 100,000 time series and over 60 million observations drawn from domains such as retail, tourism, finance, and traffic. `FORTS` reveals unexpected patterns across evaluation regimes:

- **Residual MLP-based models** (especially **N-BEATS** and **NHITS**) achieve the strongest full-shot and in-domain performance, reflecting strong within-distribution accuracy.
- In transfer settings, **PatchTST** is the strongest method for single-source out-of-domain transfer, while **xLSTM** and **KAN** perform best in the multi-source regime.
- Several larger, more computationally intensive models — including **TimeMOE**, **iTransformer**, and **TSMixer** — do not consistently improve generalization despite substantially higher training costs (over 20× slower to train than N-BEATS).

These results challenge the assumption that larger Transformer architectures are inherently superior and highlight the importance of evaluating forecasting models across diverse generalization settings. To ensure reproducibility, we release all code, datasets, and experimental configurations.

## FORTS Framework

`FORTS` defines two core testing scenarios. In the **in-domain** case, the test set is drawn from the same domain as the training set. In the **out-of-domain** case, the model is evaluated on an unseen target domain $\mathcal{S}_k$, with $\mathcal{D}^{(k)}$ excluded from the training corpus. Based on these scenarios, `FORTS` formalizes four evaluation regimes that systematically vary the domain *and* series overlap between training and test sets. This allows a more systematic evaluation of how a forecasting method performs in terms of *domain transferability*.

#### Full-shot evaluation
This regime reflects the most common forecasting setup. Each time series $\mathbf{y}^{(i)} \in \mathcal{D}^{(j)}$ is split temporally into training and test segments. For a fixed cutoff time $t^*$:
```math
\mathbf{y}^{(i)}_{\mathrm{train}} = \{ y_t^{(i)} \mid t \leq t^* \}, \qquad \mathbf{y}^{(i)}_{\mathrm{test}} = \{ y_t^{(i)} \mid t > t^* \}.
```
The model is trained and evaluated on different segments of the same series. This setup tests the ability of the model to extrapolate in known time series but does not evaluate its capacity to generalize across unseen series or domains.

#### In-domain generalization
The model is trained and evaluated on disjoint subsets of time series from the same domain $\mathcal{S}_j$. The dataset $\mathcal{D}^{(j)}$ is partitioned into non-overlapping subsets:
```math
\mathcal{D}^{(j)}_{\mathrm{train}} \subset \mathcal{D}^{(j)}, \qquad \mathcal{D}^{(j)}_{\mathrm{test}} \subset \mathcal{D}^{(j)} \setminus \mathcal{D}^{(j)}_{\mathrm{train}}.
```
This assesses the ability of the model to generalize from a subset of series to unseen series within the same domain. For example, a model trained on historical sales from a subset of stores is tested on unseen stores within the same retail chain, simulating demand forecasting for newly opened locations or products not seen during training.

#### Single-source out-of-domain transfer
This setting evaluates domain transfer from a single source domain $\mathcal{S}_j$ to a distinct target domain $\mathcal{S}_k$. The model is trained on all time series from $\mathcal{D}^{(j)}$ and tested on all time series from $\mathcal{D}^{(k)}$, with $j \neq k$:
```math
\mathcal{D}_{\mathrm{train}} = \mathcal{D}^{(j)}, \qquad \mathcal{D}_{\mathrm{test}} = \mathcal{D}^{(k)}.
```
This reflects scenarios in which a forecasting model must generalize to entirely new domains without access to data from those domains for retraining. For example, a model trained on sales data from a fashion retailer is evaluated on a grocery retailer, testing whether temporal and seasonal patterns learned in one domain transfer to a different product mix and customer behavior.

#### Multi-source out-of-domain transfer
This setting evaluates domain transfer from multiple source domains $\{ \mathcal{S}_j \}_{j \neq k}$ to a distinct target domain $\mathcal{S}_k$. The model is trained on all time series from several source datasets and evaluated on a separate held-out target dataset:
```math
\mathcal{D}_{\mathrm{train}} = \bigcup_{j \neq k} \mathcal{D}^{(j)}, \qquad \mathcal{D}_{\mathrm{test}} = \mathcal{D}^{(k)}.
```
This mirrors the cross-domain transfer challenge targeted by foundation models, where data from multiple source domains is leveraged to build a more generalizable model. We adopt a leave-one-domain-out procedure for clarity and reproducibility, though the regime is also compatible with alternative resampling strategies such as domain-wise $k$-fold cross-validation or bootstrapped domain sampling.

## Research Questions

`FORTS` is designed around six research questions, each targeting a specific axis of generalization:

- **RQ1**: How well do SOTA methods generalize to future time points within the same time series?
- **RQ2**: How effectively do they generalize to unseen time series within the same domain?
- **RQ3**: To what extent can algorithms trained on a single source dataset transfer to a new domain?
- **RQ4**: Can algorithms trained on multiple source datasets generalize to entirely new domains?
- **RQ5**: What is the relative performance of different algorithms in varying transferability conditions?
- **RQ6**: How do the different architectures compare in terms of computational efficiency?

## Methods

We evaluate ten methods, selected to represent a broad spectrum of modern architectural approaches:

| Method | Family | Description |
|:---|:---|:---|
| [`NBEATS`](https://arxiv.org/abs/1905.10437) | Residual MLP | Deep residual MLP based on backward/forward basis expansions, with interpretable decompositions. |
| [`NHITS`](https://arxiv.org/abs/2201.12886) | Hierarchical MLP | Combines residual forecasting with multi-rate input sampling for long-horizon tasks. |
| [`KAN`](https://arxiv.org/abs/2404.19756) | Kolmogorov–Arnold | Replaces fixed activations with learnable univariate transformations for greater expressiveness. |
| [`PatchTST`](https://arxiv.org/abs/2211.14730) | Transformer | Segments inputs into patches and applies self-attention over patches for efficient long-sequence forecasting. |
| [`iTransformer`](https://arxiv.org/abs/2310.06625) | Transformer | Inverts the attention structure with instance normalization and token mixing for long-context tasks. |
| [`TSMixer`](https://arxiv.org/abs/2303.06053) | MLP | Lightweight architecture alternating mixing operations over time and feature dimensions. |
| [`TFT`](https://arxiv.org/abs/1912.09363) | Transformer | Integrates attention, gating, and variable selection for interpretable multi-step forecasts. |
| [`TimeMOE`](https://arxiv.org/abs/2409.16040) | Mixture-of-Experts | Routes specialized experts to different inputs through a gating network. |
| [`TimeMixer`](https://arxiv.org/abs/2405.14616) | MLP | Models time series at multiple temporal resolutions, mixing fine- and coarse-grained representations. |
| [`xLSTM`](https://arxiv.org/abs/2405.04517) | Recurrent | Enhanced LSTM for long-range dependency modeling and in-context learning; backbone of TiRex. |

All methods are implemented and tuned with the [`neuralforecast`](https://github.com/Nixtla/neuralforecast) library from Nixtla. Architectures not available in `neuralforecast` (e.g. `TimeMOE`) were implemented by adapting the official code released by the authors. For `TimeMOE`, the hyperparameter search space is based on the [50M-parameter pretrained version](https://huggingface.co/Maple728/TimeMoE-50M).

For every method we perform hyperparameter optimization via random search, sampling and evaluating **20 configurations** (`MAX_EVALS` in [`forts/config.py`](forts/config.py)) on a validation set. The optimization covers both model-specific parameters and basic preprocessing choices (e.g. normalization strategy). The best-performing configuration is then used to retrain the model on the complete training data.

## Datasets

The evaluation uses a diverse benchmark suite of ten datasets (across six benchmark families) spanning multiple domains, sampling frequencies (monthly, quarterly, yearly, and daily), and forecast horizons. In total, the benchmark comprises over 100,000 time series and more than 60 million observations.

| Dataset | Frequency | # time series | # observations | Horizon ($H$) |
|:---|:---|---:|---:|---:|
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
| **Total** | | **106,711** | **61,731,111** | -- |

Datasets are accessed through Nixtla's [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast) library (a project dependency) and cover the Tourism, M1, M3, M4, M5, and Traffic benchmark families.

## Results

We report forecasting accuracy with the **Mean Absolute Scaled Error (MASE)**, which normalizes the forecast error by the average in-sample one-step seasonal naive error, making it interpretable and comparable across datasets. The table below reports mean MASE $\pm$ sample standard deviation and average rank (lower is better) across datasets and regimes. The `Time` column reports normalized training time relative to NBEATS, the fastest evaluated method. **Best** mean values per column are in bold; *second-best* are in italics.

| Method | Time (×NBEATS) | Full-shot MASE | Full-shot Rank | In-domain MASE | In-domain Rank | Single-source MASE | Single-source Rank | Multi-source MASE | Multi-source Rank |
|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| KAN | 1.588 | 1.376 ± 0.464 | 4.0 | 1.442 ± 0.605 | 4.4 | 8.060 ± 14.026 | 5.1 | *1.627* ± 0.516 | *4.2* |
| NBEATS | **1.000** | **1.315** ± 0.436 | **2.7** | **1.245** ± 0.416 | **2.2** | 2.287 ± 0.838 | 4.9 | 2.001 ± 1.227 | *4.2* |
| NHITS | 2.683 | *1.319* ± 0.483 | *2.9* | 1.423 ± 0.629 | *3.6* | 86.653 ± 256.823 | 6.5 | 1.783 ± 0.556 | 5.5 |
| PatchTST | 4.005 | 1.432 ± 0.431 | 5.5 | 1.438 ± 0.511 | 5.5 | **2.128** ± 0.989 | **2.1** | 1.681 ± 0.572 | 5.0 |
| TFT | 3.141 | 1.439 ± 0.517 | 5.3 | 1.474 ± 0.478 | 5.7 | *2.280* ± 1.074 | *4.1* | 1.714 ± 0.613 | 4.6 |
| TSMixer | 23.751 | 1.810 ± 0.853 | 8.8 | 1.675 ± 0.724 | 8.3 | 2.298 ± 1.091 | 4.7 | 1.816 ± 0.580 | 7.3 |
| TimeMOE | 21.486 | 1.445 ± 0.454 | 5.8 | *1.386* ± 0.486 | 5.6 | 2.437 ± 1.234 | 6.3 | 1.837 ± 0.835 | 6.3 |
| iTransformer | 20.474 | 1.667 ± 0.545 | 8.6 | 1.667 ± 0.574 | 8.2 | 2.552 ± 1.203 | 7.5 | 2.221 ± 1.922 | 6.1 |
| xLSTM | 2.158 | 1.493 ± 0.575 | 5.4 | 1.905 ± 1.757 | 6.4 | 4.524 ± 1.897 | 8.3 | **1.595** ± 0.522 | **3.8** |
| TimeMixer | *1.032* | 1.442 ± 0.469 | 6.2 | 1.423 ± 0.551 | 5.2 | 2.296 ± 1.106 | 5.5 | 1.888 ± 0.631 | 8.0 |

**Key findings:**

- **Full-shot & in-domain (RQ1–RQ2):** Residual MLP-based models dominate. NBEATS achieves the best MASE and rank in both regimes, with NHITS close behind, while heavier models such as iTransformer and TSMixer lag.
- **Single-source transfer (RQ3):** Domain shift reshuffles the ranking. NHITS and KAN suffer severe failures on specific source domains (note their very large standard deviations), while PatchTST degrades least and emerges as the most consistent method.
- **Multi-source transfer (RQ4):** xLSTM delivers the lowest MASE, with KAN close behind — recurrent and KAN-based architectures rival or surpass Transformer variants when exposed to multiple source domains. The compact Transformers (PatchTST, TFT) remain the most stable across regimes.
- **Across regimes (RQ5):** No single architecture dominates — the best model changes from NBEATS (full-shot, in-domain) to PatchTST (single-source) to xLSTM/KAN (multi-source). Strong in-domain performance is not sufficient evidence of transferability.
- **Efficiency (RQ6):** TimeMixer (~1.03×), KAN (~1.59×), and xLSTM (~2.16×) stay lightweight relative to NBEATS, whereas TimeMOE, iTransformer, and TSMixer require roughly 20–24× the training time without reliably improving generalization.

## Project Structure

```
FORTS/
├── assets/                # Plots, data splits, and model weights
├── forts/                 # Core package
│   ├── config.py          # Global configuration (e.g. MAX_EVALS = 20)
│   ├── data_pipeline/     # Data loading, splitting, and feature engineering
│   ├── experiments/       # Pipeline entry points and results processing
│   ├── load_data/         # Dataset loaders (Tourism, M1, M3, M4, M5, Traffic, ...)
│   ├── metrics/           # Evaluation metrics (MASE) and pipelines
│   ├── model_pipeline/    # Model implementations (Auto* wrappers, TimeMOE)
│   └── visualization/     # Plotting utilities
├── tests/                 # Unit and integration tests
├── scripts/               # Utility scripts
├── terraform/             # Terraform configurations for cloud (GCP) infrastructure
├── run_all_experiments.sh # Experiment runner
├── requirements.txt
└── README.md
```

## Installation

`FORTS` targets Python 3.11. Install the required packages into your environment:

```bash
pip install -r requirements.txt
```

## Running Experiments

Reproduce the four evaluation regimes with `run_all_experiments.sh`. Each command maps to a regime; add `--use-gpu` to enable GPU acceleration and `--model <AutoModelName>` (e.g. `AutoNHITS`) to run a single method.

#### Full-shot evaluation
Forecast future values from past data within the same time series.
```bash
./run_all_experiments.sh basic --use-gpu
```

#### In-domain & single-source out-of-domain transfer
Generalize from a subset of series to unseen series, both within the same domain and to a different target domain.
```bash
./run_all_experiments.sh transfer --use-gpu
```

#### Multi-source out-of-domain transfer
Train on a "coreset" of multiple source datasets and evaluate on a held-out target dataset (leave-one-domain-out).
```bash
./run_all_experiments.sh coreset --use-gpu
```

#### Other commands
The runner also supports fine-tuning variants and results processing:

```bash
./run_all_experiments.sh transfer_finetune --use-gpu   # transfer + fine-tuning on the target
./run_all_experiments.sh coreset_finetune --use-gpu    # multi-source + fine-tuning on the target
./run_all_experiments.sh process_efficiency            # aggregate training-time / efficiency metrics
./run_all_experiments.sh process_results               # aggregate MASE / rank results
./run_all_experiments.sh all --use-gpu                 # run every experiment sequentially
```
