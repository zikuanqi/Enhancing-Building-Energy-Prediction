"""Train a single model on the building energy prediction task.

Usage:
    python train.py --model proposed --epochs 50
    python train.py --model transformer_kan_dyt --epochs 30

The L2 regularization term λ·||θ||² in Eq. (1) is realised by ``weight_decay``
in the AdamW optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
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
from utils.data import prepare
from utils.metrics import all_metrics


MODEL_REGISTRY = {
    "proposed": KANTransformer,
    "transformer": PlainTransformer,
    "transformer_kan": TransformerKAN,
    "transformer_dyt": TransformerDyT,
    "transformer_matmul_free": TransformerMatMulFree,
    "transformer_kan_matmul_free": TransformerKANMatMulFree,
    "transformer_kan_dyt": TransformerKANDyT,
    "transformer_dyt_matmul_free": TransformerDyTMatMulFree,
    "cnn_lstm": CNNLSTM,
    "lstm_attention": LSTMAttention,
}


def build_model(name: str, input_dim: int, num_targets: int, horizon: int) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    return cls(input_dim=input_dim, num_targets=num_targets, horizon=horizon)


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n_samples = 0
    preds_chunks, trues_chunks = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            y_hat = model(x)
            loss = criterion(y_hat, y)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
        preds_chunks.append(y_hat.detach().cpu().numpy())
        trues_chunks.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_chunks, axis=0)
    trues = np.concatenate(trues_chunks, axis=0)
    return total_loss / max(n_samples, 1), preds, trues


def train(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, splits = prepare(
        csv_path=args.data,
        window=args.window,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )
    input_dim = len(splits.load_cols) + len(splits.feature_cols)
    model = build_model(args.model, input_dim, len(splits.load_cols), args.horizon).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    pbar = tqdm(range(args.epochs), desc=f"Training {args.model}")
    for ep in pbar:
        tr_loss, _, _ = epoch_pass(model, train_loader, optim, device, criterion)
        val_loss, _, _ = epoch_pass(model, val_loader, None, device, criterion)
        sched.step()
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss})
        pbar.set_postfix(train=f"{tr_loss:.4f}", val=f"{val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_pred, test_true = epoch_pass(model, test_loader, None, device, criterion)
    metrics = all_metrics(test_true, test_pred)

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"model": args.model, "metrics": metrics, "history": history}, f, indent=2)

    print(f"\n{args.model} test metrics: {metrics}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="proposed", choices=list(MODEL_REGISTRY))
    p.add_argument("--data", default="data/energy.csv", help="CSV path; falls back to synthetic if missing")
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4, help="λ in Eq. (1)")
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
