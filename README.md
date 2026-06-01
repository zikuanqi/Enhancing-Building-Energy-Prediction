[**English**](README.md) | [**中文**](README.zh-CN.md)

# KAN-Transformer for Building Energy Consumption Prediction

[![CI](https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/actions/workflows/ci.yml/badge.svg)](https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/branch/main/graph/badge.svg)](https://codecov.io/gh/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Official code reproduction of **"A KAN-based Transformer learning network for building energy consumption prediction"** — Zikuan Qi, University of Sydney.

This repository provides a complete, reproducible PyTorch implementation of the KAN-Transformer architecture proposed in the paper. The network predicts multi-step hourly energy consumption for **four building categories** (office / residential / commercial / industrial) using multivariate time series — energy loads plus five exogenous features (temperature, humidity, precipitation, wind speed, occupancy).

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture Overview](#architecture-overview)
- [Mathematical Formulation](#mathematical-formulation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Quick Start](#quick-start)
- [Configuration System](#configuration-system)
- [Training Details](#training-details)
- [Evaluation & Metrics](#evaluation--metrics)
- [Comparative Methods (Table 1)](#comparative-methods-table-1)
- [Module Implementation Details](#module-implementation-details)
- [Visualization](#visualization)
- [Testing](#testing)
- [CI / CD & Coverage](#ci--cd--coverage)
- [Citation](#citation)
- [License](#license)

---

## Highlights

| Feature | Description |
|---|---|
| **Multi-scale temporal decomposition** | Hierarchical moving-average decomposition into trend / seasonal / weekly / daily / short-term streams (Section 2.2) |
| **Dynamic Transformer (DyT)** | Learned gated residual `Y = αX + βF(X)` replacing Add & Norm, with sigmoid gates over LayerNorm (Eq. 4–6) |
| **MatMul-free dense layer** | Ternary weights `{-1, 0, +1}` with adaptive thresholds and straight-through estimator (Eq. 7–11) |
| **Hierarchical KAN feed-forward** | B-spline activations, 128 → 256 → 128, 8 splines × 16 basis functions (Eq. 12–13) |
| **Personalized heads** | Optional per-building residual heads on a shared trunk for rapid adaptation |
| **10 comparative methods** | Full ablation study + CNN-LSTM, LSTM-Attention baselines (Table 1 reproduction) |
| **Cross-OS CI** | Linux / Windows / macOS × Python 3.10 / 3.11 / 3.12 = 9 parallel jobs |
| **96% test coverage** | 59 unit tests covering every module, with branch coverage enabled |

---

## Architecture Overview

```
                    Input: (B, T, features)
                              │
                    ┌─────────▼──────────┐
                    │  Multi-Scale       │
                    │  Temporal          │
                    │  Decomposition     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Component         │
                    │  Embedding +       │
                    │  Cross-Scale Attn  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Sinusoidal        │
                    │  Positional        │
                    │  Encoding          │
                    └─────────┬──────────┘
                              │
               ┌──────────────▼──────────────┐
               │   KAN-Transformer Block ×N  │
               │  ┌────────────────────────┐ │
               │  │ DyT( Multi-Head        │ │
               │  │      Self-Attention )  │ │
               │  └───────────┬────────────┘ │
               │  ┌───────────▼────────────┐ │
               │  │ DyT( Hierarchical      │ │
               │  │      KAN FFN )         │ │
               │  └───────────┬────────────┘ │
               └──────────────┬──────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Mean Pooling      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  MatMul-Free Dense │
                    │  Projection Head   │
                    └─────────┬──────────┘
                              │
              Output: (B, horizon, num_targets)
```

**Data flow:**

1. **Multi-scale temporal decomposition** — the raw input sequence is decomposed into five temporal components (trend, seasonal, weekly, daily, short-term) via hierarchical centred moving averages.
2. **Component embedding** — each component is linearly projected to `d_model` dimensions, then fused via cross-component attention that learns to weight the importance of each time scale.
3. **Positional encoding** — standard sinusoidal encoding to inject temporal order information.
4. **N × KAN-Transformer blocks** — each block applies DyT-wrapped multi-head self-attention followed by DyT-wrapped hierarchical KAN feed-forward.
5. **MatMul-free projection head** — temporal mean pooling → ternary-weight dense layer producing `(horizon × num_targets)` predictions.

---

## Mathematical Formulation

### Objective Function (Eq. 1)

The total loss is MSE with L2 regularization:

```
L = (1/N) Σ ||y_pred - y_true||² + λ Σ ||θ||²
```

where λ is the weight decay coefficient (default 0.0001), implemented via `AdamW`.

### Data Preprocessing (Eq. 2–3)

**Linear interpolation** for missing values (Eq. 2):

```
x_t = x_{t-k} + (t - (t-k)) / ((t+m) - (t-k)) · (x_{t+m} - x_{t-k})
```

**Min-max normalization** (Eq. 3):

```
x_norm = (x - x_min) / (x_max - x_min) ∈ [0, 1]
```

Normalization ranges are fitted on the training set only and reused for validation/test to prevent data leakage.

### Dynamic Transformer Residual (Eq. 4–6)

Replaces conventional Add & Norm with learned gates:

```
Y_t = α_t · X_t + β_t · F_t(X_t)                    (4)
α_t = σ(W_α · LayerNorm(X_t) + b_α)                  (5)
β_t = σ(W_β · LayerNorm(F_t(X_t)) + b_β)             (6)
```

where σ is the sigmoid function and F_t is the wrapped sub-layer (self-attention or KAN-FFN).

### MatMul-Free Dense Layer (Eq. 7–11)

Ternary quantization of continuous weights:

```
Ω_{j,i} ∈ {-1, 0, +1}
Q(M) = +1 if M > τ₊ ;  -1 if M < τ₋ ;  0 otherwise   (11)
```

where `τ± = ± α · std(M)` are adaptive thresholds. The forward pass uses the additive accumulator (Eq. 10):

```
Γ_i = Σ_{j: Ω=+1} v_j  -  Σ_{j: Ω=-1} v_j           (10)
```

Gradients flow through a straight-through estimator (clipped to [-1, 1]).

### KAN Feed-Forward (Eq. 12–13)

Each KAN unit applies `k` learned B-spline activations to `k` linear projections:

```
z_i = Σ_{j=1..k} g_{i,j}(w_{i,j}^T x + b_{i,j})      (12)
g_{i,j}(s) = Σ_{l=1..L} c_{i,j,l} · B_l(s)            (13)
```

where `B_l(s)` are cubic B-spline basis functions on a uniform knot grid. A SiLU residual connection aids optimisation. The hierarchical layout is 128 → 256 → 128, with k=8 splines and L=16 basis functions per spline.

---

## Project Structure

```
KAN-Transformer-BECP/
├── .github/
│   └── workflows/
│       ├── ci.yml                # Cross-OS CI: ruff + pytest + codecov
│       └── weekly.yml            # Weekly pip-audit + regression check
├── configs/
│   ├── default.yaml              # Full training config (30 epochs, d_model=128)
│   ├── quick.yaml                # Smoke-run config (3 epochs, d_model=64)
│   └── transformer.yaml          # Plain Transformer baseline
├── models/
│   ├── __init__.py               # Exports all model classes
│   ├── dyt.py                    # Dynamic Transformer (DyT) layer — Eq. 4–6
│   ├── matmul_free.py            # Ternary MatMul-free dense — Eq. 7–11
│   ├── kan.py                    # KAN layer + HierarchicalKAN — Eq. 12–13
│   ├── kan_transformer.py        # Full proposed network
│   └── baselines.py              # 9 comparative methods (Table 1)
├── utils/
│   ├── preprocessing.py          # Missing values, anomaly detection, normalization,
│   │                             #   temporal encoding, rolling stats, HDD/CDD
│   ├── decomposition.py          # Multi-scale temporal decomposition
│   ├── data.py                   # EnergyDataset, DataLoader, synthetic generator
│   ├── metrics.py                # MAPE / RMSE / MAE / R²
│   ├── visualize.py              # Figure 2 (predicted vs true) / Figure 3 (comparison)
│   ├── callbacks.py              # Early stopping, structured logger, seed helper
│   └── config.py                 # YAML loader with deep-merge
├── experiments/
│   ├── compare.py                # Reproduces Table 1 — trains all 10 methods
│   └── plot_figures.py           # Renders Figure 2/3 + per-metric bar charts
├── notebooks/
│   └── quickstart.ipynb          # End-to-end Jupyter walkthrough
├── tests/                        # 59 unit tests (96% coverage)
│   ├── test_smoke.py             # Forward + backward on every model variant
│   ├── test_dyt.py               # DyT gate bounds, output shapes
│   ├── test_matmul_free.py       # Ternary quantization, Eq. 10 equivalence
│   ├── test_kan.py               # B-spline partition-of-unity, KAN shapes
│   ├── test_metrics.py           # Metric correctness against known values
│   ├── test_preprocessing.py     # Interpolation, normalization, feature math
│   ├── test_decomposition.py     # Decomposition reconstruction identity
│   ├── test_callbacks.py         # Early stopping, logger, seed determinism
│   ├── test_config.py            # YAML load, deep merge, dotted get
│   ├── test_data.py              # Dataset shapes, DataLoader, synthetic gen
│   └── test_visualize.py         # Plotting smoke tests (Agg backend)
├── train.py                      # Single-model training entry point
├── evaluate.py                   # Evaluate a saved checkpoint
├── pyproject.toml                # Build, lint, test, coverage config
├── requirements.txt              # Pinned pip dependencies
├── LICENSE                       # MIT
├── CHANGELOG.md
└── CONTRIBUTING.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- pip (or conda)

### Basic install

```bash
git clone https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach.git
cd Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach

# Editable install (recommended for development)
pip install -e .
```

### With extras

```bash
# Development tools (pytest, pytest-cov, ruff)
pip install -e ".[dev]"

# Jupyter notebook support (jupyter, seaborn)
pip install -e ".[notebook]"

# Both
pip install -e ".[dev,notebook]"
```

### GPU support

The default `pip install torch` gives CPU-only on some platforms. For CUDA:

```bash
# Example: CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Console scripts

After installation, three CLI commands become available:

| Command | Equivalent |
|---|---|
| `becp-train` | `python train.py` |
| `becp-evaluate` | `python evaluate.py` |
| `becp-compare` | `python experiments/compare.py` |

---

## Data Format

### Synthetic data (default)

If no CSV file is found at the configured path, the system automatically generates a **synthetic dataset** that mirrors the paper's data structure:

- **Duration:** 2 years of hourly data (17,520 timesteps)
- **4 load columns:** `office`, `residential`, `commercial`, `industrial`
- **5 exogenous features:** `temperature`, `humidity`, `precipitation`, `wind_speed`, `occupancy`
- Realistic daily/weekly/seasonal patterns with controlled noise

This allows you to run all experiments immediately without sourcing real data.

### Real data

Place your CSV at `data/energy.csv` (or set `--data <path>`). Required format:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Hourly timestamps (parsed as `DatetimeIndex`) |
| `office` | float | Office building energy consumption (kWh) |
| `residential` | float | Residential building energy consumption |
| `commercial` | float | Commercial building energy consumption |
| `industrial` | float | Industrial building energy consumption |
| `temperature` | float | Outdoor temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `precipitation` | float | Precipitation (mm) |
| `wind_speed` | float | Wind speed (m/s) |
| `occupancy` | float | Occupancy rate (0–1) |

The preprocessing pipeline automatically handles:
- Missing value interpolation (Eq. 2)
- Anomaly detection via robust z-score (MAD-based)
- Min-max normalization fitted on training split only (Eq. 3)
- Feature engineering: temporal sin/cos encoding, rolling statistics (24h & 168h windows), heating/cooling degree days, occupancy-weighted demand
- Temporal 70/10/20 train/val/test split

---

## Quick Start

### 1. Smoke run (~2 min on CPU)

```bash
python train.py --config configs/quick.yaml
```

Uses d_model=64 and 3 epochs — good for verifying the setup works.

### 2. Full proposed model

```bash
python train.py --config configs/default.yaml
```

Trains the KAN-Transformer with default hyperparameters (d_model=128, 30 epochs, cosine LR schedule).

### 3. Reproduce Table 1

```bash
python experiments/compare.py --epochs 20
```

Trains all 10 methods sequentially on the same dataset, computes MAPE / RMSE / MAE / R² for each, and saves predictions to `predictions.npz`.

### 4. Generate Figures

```bash
python experiments/plot_figures.py
```

Reads `predictions.npz` and renders:
- **Figure 2**: Predicted vs. true energy for each building category (proposed method)
- **Figure 3**: All methods comparison on a shared axis
- **Per-metric bar charts**: Side-by-side metric comparison across methods

### 5. Jupyter notebook

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/quickstart.ipynb
```

Interactive end-to-end walkthrough: synthetic data → preprocessing → training → evaluation → visualization.

---

## Configuration System

The project uses a **layered YAML configuration** system with deep-merge semantics:

```
configs/default.yaml  ←  your override YAML  ←  CLI flags
      (base)               (deep-merged)         (highest priority)
```

### Default configuration

```yaml
experiment:
  name: kan_transformer_default
  seed: 42
  output_dir: checkpoints
  log_level: INFO

data:
  csv_path: data/energy.csv       # falls back to synthetic if missing
  window: 168                     # 1 week of hourly history
  horizon: 24                     # predict next 24 hours
  batch_size: 64
  num_workers: 0
  rolling_windows: [24, 168]
  train_frac: 0.7
  val_frac: 0.1

model:
  name: proposed
  d_model: 128
  n_heads: 8
  num_layers: 3
  dropout: 0.1
  kan_hidden: [128, 256, 128]
  num_splines: 8
  num_basis: 16
  num_buildings: 1

training:
  epochs: 30
  lr: 0.001
  weight_decay: 0.0001            # λ in Eq. (1)
  grad_clip: 5.0
  early_stopping:
    enabled: true
    patience: 8
    min_delta: 0.0001
  scheduler:
    name: cosine
    warmup_epochs: 2
```

### Creating a custom config

Only specify the keys you want to override — everything else inherits from `default.yaml`:

```yaml
# configs/my_experiment.yaml
model:
  d_model: 256
  n_heads: 16
training:
  lr: 0.0005
  epochs: 50
```

```bash
python train.py --config configs/my_experiment.yaml
```

### CLI overrides

CLI flags take highest priority:

```bash
python train.py --config configs/default.yaml --epochs 50 --lr 5e-4 --batch_size 128
python train.py --resume checkpoints/proposed/checkpoint.pt
python experiments/compare.py --methods proposed transformer cnn_lstm
```

### Programmatic access

```python
from utils.config import load_config, get

cfg = load_config("configs/default.yaml")
lr = get(cfg, "training.lr")                    # 0.001
patience = get(cfg, "training.early_stopping.patience")  # 8
missing = get(cfg, "some.nonexistent.key", default=42)   # 42
```

---

## Training Details

### Optimizer

- **AdamW** with weight decay λ = 0.0001 (Eq. 1)
- Gradient clipping at max norm 5.0

### Learning rate schedule

- **Cosine annealing** (`CosineAnnealingLR`) over the full training budget
- 2 warmup epochs (configurable)

### Early stopping

- Monitors validation loss
- Patience: 8 epochs (configurable)
- Minimum delta: 0.0001 — an improvement smaller than this is counted as stagnation

### Checkpointing

- **Best model** saved to `checkpoints/<model>/best.pt` (lowest validation loss)
- **Latest model** saved to `checkpoints/<model>/checkpoint.pt` (every epoch)
- Full state: model weights + optimizer state + scheduler state + epoch + best_val

### Resume training

```bash
python train.py --resume checkpoints/proposed/checkpoint.pt
```

Loads all state and continues from the saved epoch.

### Structured logging

Every training run produces both console output and a structured `train.log` in the output directory.

### Deterministic seeding

`set_seed(42)` sets seeds for Python's `random`, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

---

## Evaluation & Metrics

### Run evaluation

```bash
python evaluate.py --model proposed --checkpoint checkpoints/proposed/best.pt
```

### Metrics

All metrics are computed on the flattened prediction array (all building categories × all horizons):

| Metric | Formula | Description |
|---|---|---|
| **MAPE** | `mean(|y_true - y_pred| / |y_true|)` | Mean Absolute Percentage Error |
| **RMSE** | `sqrt(mean((y_true - y_pred)²))` | Root Mean Squared Error |
| **MAE** | `mean(|y_true - y_pred|)` | Mean Absolute Error |
| **R²** | `1 - SS_res / SS_tot` | Coefficient of Determination |

---

## Comparative Methods (Table 1)

The repository implements all 10 methods from the paper's comparative study:

| # | Method | KAN | DyT | MatMul-free | Module |
|---|---|---|---|---|---|
| 1 | **Proposed (KAN-Transformer)** | ✅ | ✅ | ✅ | `KANTransformer` |
| 2 | Transformer | ❌ | ❌ | ❌ | `PlainTransformer` |
| 3 | Transformer-KAN | ✅ | ❌ | ❌ | `TransformerKAN` |
| 4 | Transformer-DyT | ❌ | ✅ | ❌ | `TransformerDyT` |
| 5 | Transformer-MatMul-free | ❌ | ❌ | ✅ | `TransformerMatMulFree` |
| 6 | Transformer-KAN-MatMul-free | ✅ | ❌ | ✅ | `TransformerKANMatMulFree` |
| 7 | Transformer-KAN-DyT | ✅ | ✅ | ❌ | `TransformerKANDyT` |
| 8 | Transformer-DyT-MatMul-free | ❌ | ✅ | ✅ | `TransformerDyTMatMulFree` |
| 9 | CNN-LSTM | — | — | — | `CNNLSTM` |
| 10 | LSTM-Attention | — | — | — | `LSTMAttention` |

The Transformer variants share a `_ConfigurableTransformer` backbone with boolean flags (`use_kan`, `use_dyt`, `use_matmul_free`), making each component's contribution directly attributable.

### Running the full comparison

```bash
# All 10 methods
python experiments/compare.py --epochs 20

# Selected methods only
python experiments/compare.py --methods proposed transformer cnn_lstm --epochs 20
```

Results are saved to `metrics.json` and predictions to `predictions.npz`.

---

## Module Implementation Details

### `models/dyt.py` — Dynamic Transformer Layer

```python
class DyTLayer(nn.Module):
    """Wraps any sub-layer with a dynamic gated residual (Eq. 4–6)."""
    # α = sigmoid(W_α · LayerNorm(X) + b_α)
    # β = sigmoid(W_β · LayerNorm(F(X)) + b_β)
    # output = dropout(α * X + β * F(X))
```

- Gates α and β are produced **per token and per channel** (d_model-dimensional)
- Both gates are bounded in (0, 1) via sigmoid
- Dropout is applied after gating

### `models/matmul_free.py` — Ternary Dense Layer

```python
class MatMulFreeDense(nn.Module):
    """y = scale ⊙ (x @ Q(W).T) + bias"""
    # Q(W) quantizes continuous weights to {-1, 0, +1}
    # Adaptive thresholds: τ± = ± α · mean(|W|)
    # Backward: straight-through estimator with [-1, 1] clipping
```

- `ternary_accumulate()` method implements Eq. 10 explicitly for verification
- `quantized_weight()` exports the deployable ternary matrix
- Learnable per-output scale compensates for quantization magnitude loss

### `models/kan.py` — KAN Layer

```python
class KANLayer(nn.Module):
    """z_i = Σ_j g_{i,j}(w^T x + b) where g is a learned B-spline"""
    # k=8 spline functions per output unit
    # L=16 cubic B-spline basis functions on a uniform knot grid
    # SiLU residual connection for optimization stability
```

- `_b_spline_basis()` — Cox-de Boor recursive evaluation of B-spline basis
- `HierarchicalKAN` — three KANLayer stack (128 → 256 → 128) with dropout

### `models/kan_transformer.py` — Full Proposed Network

- `MultiScaleDecomposition` → `ComponentEmbedding` (with cross-scale attention) → `SinusoidalPositionalEncoding` → N × `KANTransformerBlock` → `MatMulFreeDense` head
- Optional personalized per-building heads via `building_id` argument
- Supports both integer and batched tensor building IDs

### `utils/preprocessing.py` — Data Pipeline

- `linear_interpolate_missing()` — Eq. 2 via pandas
- `detect_and_clean_anomalies()` — robust z-score (MAD), consecutive runs → interpolate, isolated → neighbour mean
- `minmax_normalize()` — Eq. 3, with reusable train-fitted ranges
- `temporal_encoding()` — sin/cos pairs for hour, day-of-week, month, season
- `rolling_statistics()` — rolling mean/var/max over 24h and 168h windows
- `degree_days_and_demand()` — HDD, CDD, occupancy-weighted demand
- `preprocess_energy_dataframe()` — full pipeline with 70/10/20 temporal split

### `utils/decomposition.py` — Multi-Scale Decomposition

Differentiable PyTorch module using `F.conv1d` with edge-replicated padding:
- Hierarchical: strip trend → seasonal → weekly → daily → short-term residual
- Configurable periods (default: daily=24, weekly=168, seasonal=720, trend=4320)

---

## Visualization

### Figure 2 — Predicted vs. True

```python
from utils.visualize import plot_predictions_vs_true
plot_predictions_vs_true(y_true, y_pred, building_names=["office", "residential", ...])
```

### Figure 3 — Method Comparison

```python
from utils.visualize import plot_method_comparison
plot_method_comparison(results_dict, building="office")
```

### Per-Metric Bar Chart

```python
from utils.visualize import plot_metric_bar
plot_metric_bar(metrics_dict, metric="MAPE")
```

Or generate all figures at once:

```bash
python experiments/plot_figures.py
```

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run with coverage

```bash
pytest tests/ -v --cov=models --cov=utils --cov=experiments --cov-report=term
```

### Test suite overview (59 tests)

| File | Tests | What is verified |
|---|---|---|
| `test_smoke.py` | 3 | Forward + backward on proposed, baseline, and recurrent models |
| `test_dyt.py` | 3 | Gate values bounded in (0,1), output shape, dropout effect |
| `test_matmul_free.py` | 4 | Ternary quantization, Eq. 10 ≡ matmul equivalence, gradient flow |
| `test_kan.py` | 4 | B-spline partition-of-unity, KAN output shape, hierarchical KAN |
| `test_metrics.py` | 4 | MAPE/RMSE/MAE/R² against known values |
| `test_preprocessing.py` | 5 | Interpolation correctness, normalization ranges, feature engineering |
| `test_decomposition.py` | 3 | Components sum to original, output shapes, short sequences |
| `test_callbacks.py` | 8 | Early stopping triggers/resets, seed determinism, logger file output |
| `test_config.py` | 10 | YAML deep merge, dotted get, missing file error, empty override |
| `test_data.py` | 9 | Dataset shapes, DataLoader batches, CSV load/fallback, determinism |
| `test_visualize.py` | 6 | Plotting smoke tests (all use matplotlib Agg backend for headless CI) |

### Coverage

Current coverage: **96%** (branch coverage enabled). The only excluded files are the CLI entry-point scripts (`experiments/compare.py`, `experiments/plot_figures.py`) which are exercised via end-to-end runs rather than unit tests.

---

## CI / CD & Coverage

### Continuous Integration

Every push and pull request to `main` triggers the **CI pipeline** (`.github/workflows/ci.yml`):

1. **Matrix build**: 3 OS (Linux, Windows, macOS) × 3 Python versions (3.10, 3.11, 3.12) = **9 parallel jobs**
2. **Lint**: `ruff check .` with rules E, F, I, B, UP, SIM
3. **Test**: `pytest tests/ -v` with coverage collection
4. **Coverage upload**: Codecov (ubuntu + Python 3.12 only)

### Weekly audit

A cron job (`.github/workflows/weekly.yml`) runs every Monday at 06:00 UTC:
- `pip-audit` for known vulnerabilities
- `pip list --outdated` for dependency staleness
- Full test suite against latest dependency versions
- Automatically opens a GitHub issue if anything fails

### Local linting

```bash
ruff check .
ruff format .  # auto-format
```

---

## Citation

```bibtex
@inproceedings{qi2025kan,
  title     = {A KAN-based Transformer learning network for building
               energy consumption prediction},
  author    = {Qi, Zikuan},
  year      = {2025},
  note      = {International Conference paper, University of Sydney}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE) — free for academic and commercial use, with attribution.
