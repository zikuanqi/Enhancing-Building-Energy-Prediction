"""Unit tests for utils.config — YAML loader, deep merge, dotted get."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import DEFAULT_CONFIG_PATH, _deep_merge, get, load_config


def test_default_config_has_required_top_level_keys():
    cfg = load_config()
    for k in ("experiment", "data", "model", "training"):
        assert k in cfg, f"missing top-level key: {k}"


def test_load_with_explicit_default_path_is_equivalent_to_none():
    a = load_config()
    b = load_config(DEFAULT_CONFIG_PATH)
    assert a == b


def test_load_with_override_yaml_merges_deeply(tmp_path: Path):
    override = tmp_path / "override.yaml"
    override.write_text(
        "model:\n  d_model: 256\n  n_heads: 16\n"
        "training:\n  lr: 0.0005\n"
    )
    cfg = load_config(override)
    # Overridden keys take new values.
    assert cfg["model"]["d_model"] == 256
    assert cfg["model"]["n_heads"] == 16
    assert cfg["training"]["lr"] == 0.0005
    # Untouched keys keep defaults from configs/default.yaml.
    assert cfg["model"]["num_splines"] == 8
    assert cfg["training"]["weight_decay"] == 0.0001
    assert "data" in cfg


def test_empty_override_file_keeps_defaults(tmp_path: Path):
    override = tmp_path / "empty.yaml"
    override.write_text("")  # yaml.safe_load returns None
    cfg = load_config(override)
    # Should not crash; should equal the default.
    assert cfg == load_config()


def test_deep_merge_handles_nested_dicts():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 99, "z": 3}, "c": 5}
    out = _deep_merge(base, override)
    assert out == {"a": 1, "b": {"x": 1, "y": 99, "z": 3}, "c": 5}
    # Original dict is not mutated.
    assert base == {"a": 1, "b": {"x": 1, "y": 2}}


def test_deep_merge_override_replaces_non_dict():
    base = {"a": {"x": 1}}
    override = {"a": 99}
    assert _deep_merge(base, override) == {"a": 99}


def test_get_dotted_path():
    cfg = {"training": {"lr": 0.001, "scheduler": {"name": "cosine"}}}
    assert get(cfg, "training.lr") == 0.001
    assert get(cfg, "training.scheduler.name") == "cosine"


def test_get_returns_default_for_missing_key():
    cfg = {"a": {"b": 1}}
    assert get(cfg, "a.b") == 1
    assert get(cfg, "a.missing", default=42) == 42
    assert get(cfg, "x.y.z") is None


def test_get_returns_default_when_traversing_non_dict():
    cfg = {"a": 5}
    # 'a' is an int, can't descend further.
    assert get(cfg, "a.b.c", default="fallback") == "fallback"


def test_load_config_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")
