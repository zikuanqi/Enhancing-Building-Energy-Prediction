"""Training callbacks: early stopping and structured logging."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EarlyStopping:
    """Stop training when validation loss plateaus.

    A relative improvement smaller than ``min_delta`` over ``patience``
    consecutive epochs triggers a stop. The best loss seen so far is kept
    in ``best_loss`` for logging.
    """

    patience: int = 8
    min_delta: float = 1e-4
    best_loss: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop *after* this epoch."""
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped


def setup_logger(
    name: str = "becp",
    log_dir: str | Path | None = None,
    level: str | int = "INFO",
) -> logging.Logger:
    """Console + optional file logger with a clean format."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    """Best-effort deterministic seeding for reproducibility."""
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
