# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LICENSE (MIT), CHANGELOG, CONTRIBUTING.

## [0.1.0] — 2025-05-27

### Added
- Initial reproduction of "A KAN-based Transformer learning network for
  building energy consumption prediction" (Qi, 2025).
- Core architecture modules:
  - `models/dyt.py` — Dynamic Transformer residual (Eq. 4–6).
  - `models/matmul_free.py` — Ternary MatMul-free dense layer (Eq. 7–11).
  - `models/kan.py` — Kolmogorov-Arnold layer with B-spline basis (Eq. 12–13).
  - `models/kan_transformer.py` — Full proposed network.
  - `models/baselines.py` — All nine comparative methods from Table 1.
- Data pipeline:
  - `utils/preprocessing.py` — Missing value handling, anomaly detection,
    Min-Max normalization, temporal / rolling / degree-day features.
  - `utils/decomposition.py` — Multi-scale temporal decomposition.
  - `utils/data.py` — `EnergyDataset`, `build_dataloaders`, synthetic generator.
- Training / evaluation entry points: `train.py`, `evaluate.py`,
  `experiments/compare.py` (reproduces Table 1).
- Smoke test suite covering every model variant.
