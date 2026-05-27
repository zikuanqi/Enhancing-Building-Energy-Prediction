"""Unit tests for utils.callbacks (early stopping, seed, logger)."""

from __future__ import annotations

import logging

import numpy as np
import torch

from utils.callbacks import EarlyStopping, set_seed, setup_logger


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=3, min_delta=0.0)
    assert not es.step(1.0)  # improvement
    assert not es.step(1.0)  # plateau 1
    assert not es.step(1.0)  # plateau 2
    assert es.step(1.0)      # plateau 3 → stop


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, min_delta=0.0)
    es.step(2.0)
    es.step(2.0)   # bad 1
    es.step(1.0)   # improvement → reset
    assert es.bad_epochs == 0
    assert not es.stopped


def test_set_seed_is_deterministic():
    set_seed(123)
    a = (np.random.rand(5), torch.rand(5))
    set_seed(123)
    b = (np.random.rand(5), torch.rand(5))
    assert np.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])


def test_setup_logger_returns_logger():
    log = setup_logger("becp.test", level="INFO")
    assert isinstance(log, logging.Logger)
    assert log.handlers, "logger should attach at least one handler"


def test_setup_logger_writes_to_file(tmp_path):
    log = setup_logger("becp.test_file", log_dir=tmp_path, level="DEBUG")
    log.info("hello world")
    # close the file handler so Windows allows the temp dir to be cleaned up
    for h in list(log.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        log.removeHandler(h)
    log_file = tmp_path / "train.log"
    assert log_file.exists()
    assert "hello world" in log_file.read_text(encoding="utf-8")


def test_setup_logger_reuses_existing_handlers():
    """Calling setup_logger twice with the same name must not duplicate handlers."""
    a = setup_logger("becp.test_reuse")
    n = len(a.handlers)
    b = setup_logger("becp.test_reuse")
    assert a is b
    assert len(b.handlers) == n


def test_early_stopping_first_step_records_loss():
    es = EarlyStopping(patience=3, min_delta=0.0)
    assert es.best_loss == float("inf")
    es.step(0.5)
    assert es.best_loss == 0.5
    assert es.bad_epochs == 0


def test_early_stopping_min_delta_must_be_strict_improvement():
    """A drop smaller than min_delta should NOT reset the counter."""
    es = EarlyStopping(patience=10, min_delta=0.1)
    es.step(1.0)
    es.step(0.95)  # improvement only 0.05 < min_delta → counted as plateau
    assert es.bad_epochs == 1
