"""Evaluate a trained checkpoint on the held-out test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train import MODEL_REGISTRY, build_model, epoch_pass
from utils.data import prepare
from utils.metrics import all_metrics


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, splits = prepare(
        csv_path=args.data,
        window=args.window,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )
    input_dim = len(splits.load_cols) + len(splits.feature_cols)
    model = build_model(args.model, input_dim, len(splits.load_cols), args.horizon).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    _, pred, true = epoch_pass(model, test_loader, None, device, nn.MSELoss())
    metrics = all_metrics(true, pred)
    print(json.dumps({"model": args.model, "metrics": metrics}, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/energy.csv")
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
