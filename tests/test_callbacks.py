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
