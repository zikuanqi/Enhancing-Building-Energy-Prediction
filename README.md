# KAN-Transformer for Building Energy Consumption Prediction

[![CI](https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/actions/workflows/ci.yml/badge.svg)](https://github.com/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach/branch/main/graph/badge.svg)](https://codecov.io/gh/zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reproduction of **"A KAN-based Transformer learning network for building
energy consumption prediction"** by Zikuan Qi (University of Sydney).

The network predicts multi-step energy consumption for four building
categories (office / residential / commercial / industrial) using
multivariate hourly time series — energy loads plus five exogenous features
(temperature, humidity, precipitation, wind speed, occupancy).

## Highlights

- **Multi-scale temporal decomposition** — daily / weekly / seasonal /
  short-term / long-term trend streams (Section 2.2).
- **Dynamic Transformer (DyT)** — learned gated residual replacing Add &
  Norm: `Y = αX + βF(X)` with sigmoid gates over LayerNorm (Eq. 4–6).
- **MatMul-free dense layer** — ternary weights `{-1, 0, +1}` with adaptive
  thresholds and a straight-through estimator (Eq. 7–11).
- **Hierarchical KAN feed-forward** — B-spline activations, 128 → 256 → 128,
  8 splines × 16 basis (Eq. 12–13).
- **Personalized heads** — optional per-building residual heads on top of a
  shared trunk for rapid adaptation to new buildings.

## Project layout

```
KAN-Transformer-BECP/
├── configs/                  # YAML configs (default.yaml, quick.yaml, transformer.yaml)
├── models/
│   ├── dyt.py                # Dynamic Transformer (DyT)
│   ├── matmul_free.py        # Ternary MatMul-free dense layer
│   ├── kan.py                # KAN layer + Hierarchical KAN
│   ├── kan_transformer.py    # The proposed network
│   └── baselines.py          # Plain Transformer + all comparative variants
├── utils/
│   ├── preprocessing.py      # Missing-value / anomaly / norm / feature engineering
│   ├── decomposition.py      # Multi-scale temporal decomposition
│   ├── data.py               # Dataset, DataLoader, synthetic generator
│   ├── metrics.py            # MAPE / RMSE / MAE / R²
│   ├── visualize.py          # Figure 2 / Figure 3 reproductions
│   ├── callbacks.py          # Early stopping, logger, deterministic seed
│   └── config.py             # YAML loader with deep-merge
├── experiments/
│   ├── compare.py            # Reproduces Table 1 + saves predictions.npz
│   └── plot_figures.py       # Renders Figure 2/3 + per-metric bar charts
├── notebooks/quickstart.ipynb
├── tests/                    # 30 unit + smoke tests
├── train.py                  # Single-model training entry point
├── evaluate.py               # Evaluate a trained checkpoint
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
pip install -e .
# with dev dependencies (pytest, ruff)
pip install -e ".[dev]"
# with notebook dependencies (jupyter, seaborn)
pip install -e ".[notebook]"
```

## Quick start

```bash
# 1) Smoke run on the small config (~2 minutes on CPU)
python train.py --config configs/quick.yaml

# 2) Full proposed model
python train.py --config configs/default.yaml

# 3) Reproduce Table 1 — every comparative method on one dataset
python experiments/compare.py --epochs 20

# 4) Render Figure 2 and Figure 3 from the saved predictions
python experiments/plot_figures.py
```

If `data/energy.csv` is not present, the code automatically falls back to a
synthetic dataset that mirrors the paper's variables (4 categories × hourly
× 5 exogenous features).

### Real data format

If you supply `data/energy.csv`, it must have a `timestamp` column plus the
columns:

```
office, residential, commercial, industrial,
temperature, humidity, precipitation, wind_speed, occupancy
```

at hourly resolution.

## CLI configuration

Every script merges your YAML on top of `configs/default.yaml`, and CLI
flags override everything. A few useful overrides:

```bash
python train.py --config configs/default.yaml --epochs 50 --lr 5e-4
python train.py --resume checkpoints/proposed/checkpoint.pt
python experiments/compare.py --methods proposed transformer cnn_lstm
```

## Tests

```bash
pytest tests/ -v
```

The suite covers DyT gate bounds, MatMul-free numerical equivalence of
Eq. (10) vs. matmul, KAN B-spline partition-of-unity, multi-scale
decomposition reconstruction, preprocessing math, early-stopping logic, and
end-to-end forward/backward on every model variant.

## Reference

Qi, Z. (2025) *A KAN-based Transformer learning network for building
energy consumption prediction.* International Conference paper.

## License

[MIT](LICENSE)
