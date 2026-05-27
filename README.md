# KAN-Transformer for Building Energy Consumption Prediction

Reproduction of the paper **"A KAN-based Transformer learning network for building energy consumption prediction"** by Zikuan Qi (University of Sydney).

The network predicts multi-step energy consumption for four building categories using multivariate time series (energy + 5 exogenous features: temperature, humidity, precipitation, wind speed, occupancy). It combines:

- **Multi-scale temporal decomposition** — daily / weekly / seasonal / short-term / long-term components.
- **Dynamic Transformer (DyT) layer** — replaces Add & Norm with learned weighted residual `Y = αX + βF(X)`.
- **MatMul-free dense layer** — ternary weights `{-1, 0, +1}` with straight-through estimator.
- **KAN feed-forward** — Kolmogorov–Arnold Network with learned B-spline activations (hierarchical: 128 → 256 → 128, 8 splines × 16 basis).

## Project structure

```
KAN-Transformer-BECP/
├── models/
│   ├── dyt.py                # Dynamic Transformer (DyT) layer
│   ├── matmul_free.py        # Ternary MatMul-free dense layer
│   ├── kan.py                # KAN layer + B-spline basis
│   ├── kan_transformer.py    # Full KAN-Transformer model
│   └── baselines.py          # Transformer / CNN-LSTM / LSTM-Attention variants
├── utils/
│   ├── preprocessing.py      # Missing-value, anomaly, normalization, feature eng.
│   ├── decomposition.py      # Multi-scale temporal decomposition
│   ├── metrics.py            # MAPE / RMSE / MAE / R²
│   └── data.py               # Dataset / DataLoader / synthetic generator
├── experiments/
│   └── compare.py            # Table 1 reproduction — all comparative methods
├── train.py                  # Single-model training entry point
├── evaluate.py               # Evaluation entry point
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# Train the proposed network on synthetic / supplied data
python train.py --model proposed --epochs 50

# Reproduce Table 1 — average MAPE / RMSE / MAE / R² across all comparative methods
python experiments/compare.py
```

If `data/energy.csv` is not provided the code falls back to a synthetic dataset that
mirrors the paper's variables (4 categories × hourly resolution × 5 exogenous features).

## Reference

Qi, Z. (2025) *A KAN-based Transformer learning network for building energy
consumption prediction.* International Conference paper.
