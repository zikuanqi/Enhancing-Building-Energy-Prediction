"""YAML config loader with deep merge over a default config.

Usage::

    cfg = load_config("configs/default.yaml")
    cfg = load_config("configs/quick.yaml")          # merged with default
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it onto ``configs/default.yaml``.

    Returns ``configs/default.yaml`` verbatim when ``path`` is ``None`` or
    equal to the default path.
    """
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    if path is None or Path(path).resolve() == DEFAULT_CONFIG_PATH.resolve():
        return cfg
    with open(path) as f:
        override = yaml.safe_load(f) or {}
    return _deep_merge(cfg, override)


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value via dotted path: e.g. ``"training.lr"``."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
