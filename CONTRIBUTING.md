# Contributing

Thanks for your interest in improving this reproduction of the KAN-Transformer
building energy prediction work.

## Getting set up

```bash
git clone git@github.com:zikuanqi/Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach.git
cd Enhancing-Building-Energy-Prediction-via-KAN-Transformer-A-Multi-scale-Temporal-Learning-Approach
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

The smoke test in [tests/test_smoke.py](tests/test_smoke.py) builds every
model and verifies a forward + backward pass. Per-module unit tests live
alongside in `tests/`.

## Code style

- Python 3.10+ syntax (`from __future__ import annotations`, PEP 604 unions).
- Run `ruff check .` and `ruff format .` before committing.
- Type hints on public functions; brief one-line docstrings.

## Reporting issues

Please include:
- The command you ran and full error output.
- Output of `python -c "import torch; print(torch.__version__)"`.
- Your OS and Python version.

## Pull requests

1. Open an issue first for non-trivial changes so we can discuss the approach.
2. Keep each PR focused — one logical change per PR.
3. Add or update tests when changing model code.
4. Update `CHANGELOG.md` under `[Unreleased]`.
