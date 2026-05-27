# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LICENSE (MIT), CHANGELOG, CONTRIBUTING.
- 29 additional unit tests bringing the suite to **59 passing**:
  `test_visualize.py` (6), `test_config.py` (10), `test_data.py` (9),
  4 new `test_callbacks.py` cases.
- Codecov coverage reporting + README badge.
- Cross-OS CI matrix (Linux / Windows / macOS × Python 3.10 / 3.11 / 3.12
  = 9 parallel jobs).
- Weekly cron workflow (`weekly.yml`) running pip-audit + tests against
  latest dependency versions; auto-opens an issue on regression.

### Changed
- `pyproject.toml`: configure coverage to omit CLI scripts
  (`experiments/compare.py`, `experiments/plot_figures.py`) and tests.
  Library coverage now **96 %** (branch coverage on), up from 72 %.
- `pyproject.toml` with packaging metadata, optional `dev` / `notebook`
  extras, and console-script entry points (`becp-train`, `becp-evaluate`,
  `becp-compare`).
- YAML configuration system (`configs/`, `utils/config.py`) with deep-merge
  over a default config and CLI override.
- Visualization module (`utils/visualize.py`) and
  `experiments/plot_figures.py` reproducing Figure 2 (proposed vs. true)
  and Figure 3 (all methods comparison) plus per-metric bar charts.
- Early stopping, structured file + console logger, deterministic seeding
  helper, and checkpoint resume in `train.py` (`utils/callbacks.py`).
- GitHub Actions CI (`.github/workflows/ci.yml`) — ruff lint + pytest on
  Python 3.10/3.11/3.12.
- 28 per-module unit tests in `tests/` covering DyT gates, MatMul-free
  Eq. (10) equivalence, KAN B-spline partition-of-unity, decomposition
  reconstruction, preprocessing math, early-stopping logic, and metrics.
- Quickstart Jupyter notebook (`notebooks/quickstart.ipynb`).

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
