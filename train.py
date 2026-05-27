"""Train a single model on the building energy prediction task.

Features:
    - YAML config (configs/) with CLI overrides
    - Early stopping with configurable patience / min-delta
    - Structured logging (console + file)
    - Deterministic seeding
    - Resume training from a checkpoint
    - L2 weight decay (λ in Eq. (1)) via AdamW

Usage::

    python train.py --config configs/default.yaml
    python train.py --config configs/quick.yaml --epochs 5
    python train.py --resume checkpoints/proposed/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import time
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
from utils.callbacks import EarlyStopping, set_seed, setup_logger
from utils.config import load_config
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
    return MODEL_REGISTRY[name](input_dim=input_dim, num_targets=num_targets, horizon=horizon)


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
    grad_clip: float = 5.0,
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
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
        preds_chunks.append(y_hat.detach().cpu().numpy())
        trues_chunks.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_chunks, axis=0)
    trues = np.concatenate(trues_chunks, axis=0)
    return total_loss / max(n_samples, 1), preds, trues


def save_checkpoint(path: Path, *, model, optimizer, scheduler, epoch, best_val) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, *, model, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler is not None and ckpt.get("sched_state") is not None:
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt


def train(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("becp.train", log_dir=out_dir, level=args.cfg["experiment"].get("log_level", "INFO"))
    set_seed(args.cfg["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Model:  %s", args.model)
    logger.info("Config: %s", json.dumps(args.cfg, indent=2))

    train_loader, val_loader, test_loader, splits = prepare(
        csv_path=args.data, window=args.window, horizon=args.horizon, batch_size=args.batch_size,
    )
    input_dim = len(splits.load_cols) + len(splits.feature_cols)
    model = build_model(args.model, input_dim, len(splits.load_cols), args.horizon).to(device)
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.MSELoss()
    es_cfg = args.cfg["training"].get("early_stopping", {})
    early = EarlyStopping(
        patience=es_cfg.get("patience", 8),
        min_delta=es_cfg.get("min_delta", 1e-4),
    ) if es_cfg.get("enabled", True) else None

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model=model, optimizer=optim, scheduler=sched)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        logger.info("Resumed from %s at epoch %d (best_val=%.4f)", args.resume, start_epoch, best_val)

    history = []
    grad_clip = args.cfg["training"].get("grad_clip", 5.0)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "checkpoint.pt"

    t0 = time.time()
    pbar = tqdm(range(start_epoch, args.epochs), desc=f"Training {args.model}")
    for ep in pbar:
        tr_loss, _, _ = epoch_pass(model, train_loader, optim, device, criterion, grad_clip)
        val_loss, _, _ = epoch_pass(model, val_loader, None, device, criterion)
        sched.step()
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss})
        pbar.set_postfix(train=f"{tr_loss:.4f}", val=f"{val_loss:.4f}")
        logger.info("epoch=%d train=%.4f val=%.4f", ep, tr_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model=model, optimizer=optim, scheduler=sched, epoch=ep, best_val=best_val)
        save_checkpoint(last_path, model=model, optimizer=optim, scheduler=sched, epoch=ep, best_val=best_val)

        if early is not None and early.step(val_loss):
            logger.info("Early stopping at epoch %d (best_val=%.4f)", ep, early.best_loss)
            break

    logger.info("Training took %.1f s", time.time() - t0)

    # Load best for test evaluation
    if best_path.exists():
        load_checkpoint(best_path, model=model)
    _, test_pred, test_true = epoch_pass(model, test_loader, None, device, criterion)
    metrics = all_metrics(test_true, test_pred)
    logger.info("Test metrics: %s", metrics)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"model": args.model, "metrics": metrics, "history": history}, f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML config; merged onto configs/default.yaml")
    p.add_argument("--model", default=None, choices=list(MODEL_REGISTRY))
    p.add_argument("--data", default=None, help="CSV path; falls back to synthetic if missing")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None, help="λ in Eq. (1)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume", default=None, help="Path to a checkpoint.pt to resume from")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.model is None: args.model = cfg["model"]["name"]
    if args.data is None: args.data = cfg["data"].get("csv_path", "data/energy.csv")
    if args.window is None: args.window = cfg["data"]["window"]
    if args.horizon is None: args.horizon = cfg["data"]["horizon"]
    if args.batch_size is None: args.batch_size = cfg["data"]["batch_size"]
    if args.epochs is None: args.epochs = cfg["training"]["epochs"]
    if args.lr is None: args.lr = cfg["training"]["lr"]
    if args.weight_decay is None: args.weight_decay = cfg["training"]["weight_decay"]
    if args.output_dir is None: args.output_dir = cfg["experiment"].get("output_dir", "checkpoints")
    args.cfg = cfg
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
