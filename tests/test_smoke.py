"""Smoke tests — build every model, run one forward + backward pass on
synthetic data, and confirm output shapes.

Run with::

    python -m tests.test_smoke

or via pytest::

    pytest tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import (  # noqa: E402
    CNNLSTM,
    KANTransformer,
    LSTMAttention,
    PlainTransformer,
    TransformerDyT,
    TransformerDyTMatMulFree,
    TransformerKAN,
    TransformerKANDyT,
    TransformerKANMatMulFree,
    TransformerMatMulFree,
)
from utils.data import build_dataloaders, synthetic_energy_dataframe  # noqa: E402
from utils.metrics import all_metrics  # noqa: E402
from utils.preprocessing import preprocess_energy_dataframe  # noqa: E402

B, T, F_IN = 4, 64, 9  # batch, window, features (4 loads + 5 exog)
HORIZON, N_TARGETS = 12, 4


def _check_model(cls, *, input_dim=F_IN, num_targets=N_TARGETS, horizon=HORIZON):
    model = cls(input_dim=input_dim, num_targets=num_targets, horizon=horizon)
    x = torch.randn(B, T, input_dim)
    y = torch.randn(B, horizon, num_targets)
    y_hat = model(x)
    assert y_hat.shape == y.shape, f"{cls.__name__}: got {y_hat.shape}, want {y.shape}"
    loss = nn.MSELoss()(y_hat, y)
    loss.backward()
    print(f"  ok  {cls.__name__:32s} → output {tuple(y_hat.shape)}, loss {loss.item():.4f}")


def test_all_models():
    for cls in [
        PlainTransformer,
        TransformerKAN,
        TransformerDyT,
        TransformerMatMulFree,
        TransformerKANMatMulFree,
        TransformerKANDyT,
        TransformerDyTMatMulFree,
        CNNLSTM,
        LSTMAttention,
        KANTransformer,
    ]:
        _check_model(cls)


def test_pipeline_synthetic():
    df = synthetic_energy_dataframe(periods=24 * 90)
    splits = preprocess_energy_dataframe(df)
    train_loader, val_loader, test_loader = build_dataloaders(
        splits, window=48, horizon=12, batch_size=16,
    )
    x, y = next(iter(train_loader))
    assert x.dim() == 3 and y.dim() == 3
    print(f"  ok  pipeline → train batch X {tuple(x.shape)}, y {tuple(y.shape)}")

    # Smoke train: 1 step of the proposed model
    model = KANTransformer(
        input_dim=x.shape[-1], num_targets=y.shape[-1], horizon=y.shape[1]
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    y_hat = model(x)
    loss = nn.MSELoss()(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"  ok  one training step on proposed model, loss {loss.item():.4f}")


def test_metrics():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((32, 12, 4))
    m = all_metrics(y, y + 0.1)
    for k in ("MAPE", "RMSE", "MAE", "R2"):
        assert k in m
    print(f"  ok  metrics: {m}")


if __name__ == "__main__":
    print("Running smoke tests...")
    test_metrics()
    test_pipeline_synthetic()
    test_all_models()
    print("\nAll smoke tests passed.")
