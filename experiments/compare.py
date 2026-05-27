"""Reproduce Table 1 of the paper — train every method and report the
average MAPE / RMSE / MAE / R² on the held-out test set.

This is intentionally a single self-contained script so a reader can run
``python experiments/compare.py`` and reproduce the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Allow running from repo root: ``python experiments/compare.py``
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import MODEL_REGISTRY, build_model, epoch_pass  # noqa: E402
from utils.data import prepare  # noqa: E402
from utils.metrics import all_metrics  # noqa: E402


METHODS = [
    "transformer",
    "transformer_kan",
    "transformer_dyt",
    "transformer_matmul_free",
    "transformer_kan_matmul_free",
    "transformer_kan_dyt",
    "transformer_dyt_matmul_free",
    "cnn_lstm",
    "lstm_attention",
    "proposed",
]


def run_one(name, splits, train_loader, val_loader, test_loader, device, epochs, lr, wd):
    input_dim = len(splits.load_cols) + len(splits.feature_cols)
    model = build_model(name, input_dim, len(splits.load_cols), test_loader.dataset.horizon).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for _ in range(epochs):
        epoch_pass(model, train_loader, optim, device, criterion)
        val_loss, _, _ = epoch_pass(model, val_loader, None, device, criterion)
        sched.step()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    _, pred, true = epoch_pass(model, test_loader, None, device, criterion)
    return all_metrics(true, pred), pred, true


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/energy.csv")
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--output", default="experiments/results.json")
    p.add_argument("--save_predictions", default="experiments/predictions.npz",
                   help="npz with y_true + each method's preds (set to '' to skip)")
    p.add_argument("--methods", nargs="*", default=METHODS)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, splits = prepare(
        csv_path=args.data, window=args.window, horizon=args.horizon, batch_size=args.batch_size,
    )

    results = {}
    saved_preds: dict[str, np.ndarray] = {}
    saved_true: np.ndarray | None = None
    for name in tqdm(args.methods, desc="Methods"):
        try:
            metrics, pred, true = run_one(
                name, splits, train_loader, val_loader, test_loader,
                device, args.epochs, args.lr, args.weight_decay,
            )
            saved_preds[name] = pred
            saved_true = true
        except Exception as e:  # noqa: BLE001
            metrics = {"error": str(e)}
        results[name] = metrics
        print(f"{name:36s} → {metrics}")

    if args.save_predictions and saved_true is not None:
        out = Path(args.save_predictions)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, y_true=saved_true, **saved_preds)
        print(f"\nSaved predictions → {out}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Table 1 ===")
    print(f"{'Method':36s}  {'MAPE':>8s} {'RMSE':>8s} {'MAE':>8s} {'R^2':>8s}")
    for name, m in results.items():
        if "error" in m:
            print(f"{name:36s}  ERROR: {m['error']}")
            continue
        print(
            f"{name:36s}  "
            f"{m['MAPE']:>8.3f} {m['RMSE']:>8.3f} {m['MAE']:>8.3f} {m['R2']:>8.3f}"
        )


if __name__ == "__main__":
    main()
