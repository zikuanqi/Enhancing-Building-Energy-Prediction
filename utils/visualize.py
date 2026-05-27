"""Plotting utilities reproducing Figure 2 and Figure 3 of the paper.

Figure 2: proposed model's predictions overlaid on the ground truth for
each of the four building categories.

Figure 3: proposed model vs. all comparative methods on a shared time
window — the proposed (black dashed) line should track the true (red) line
closer than the baselines do.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_NAMES = ("Office", "Residential", "Commercial", "Industrial")


def plot_predictions_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path = "experiments/figure2.png",
    sample_idx: int = 0,
    category_names: tuple[str, ...] = CATEGORY_NAMES,
) -> Path:
    """Figure 2 reproduction: 2x2 grid, one panel per building category.

    y_true, y_pred: (N, horizon, num_targets)
    sample_idx selects which test sample (sliding window) to plot.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    horizon = y_true.shape[1]
    n_cat = y_true.shape[2]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    axes = axes.ravel()
    t = np.arange(horizon)
    for c in range(min(n_cat, 4)):
        ax = axes[c]
        ax.plot(t, y_true[sample_idx, :, c], label="True", color="tab:red", linewidth=1.8)
        ax.plot(
            t, y_pred[sample_idx, :, c],
            label="Predicted", color="black", linestyle="--", linewidth=1.4,
        )
        ax.set_title(category_names[c] if c < len(category_names) else f"Category {c}")
        ax.set_xlabel("Forecast step (h)")
        ax.set_ylabel("Normalized load")
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(loc="upper right")
    fig.suptitle("Figure 2 — Predicted vs. true building energy consumption")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_method_comparison(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_path: str | Path = "experiments/figure3.png",
    sample_idx: int = 0,
    category: int = 0,
    category_names: tuple[str, ...] = CATEGORY_NAMES,
    proposed_key: str = "proposed",
) -> Path:
    """Figure 3 reproduction: overlay every method's prediction on one
    category's true trajectory. The "proposed" line is drawn bold black
    dashed to match the paper's figure.
    """
    y_true = np.asarray(y_true)
    horizon = y_true.shape[1]
    t = np.arange(horizon)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        t, y_true[sample_idx, :, category],
        label="True value", color="tab:red", linewidth=2.2,
    )
    cmap = plt.get_cmap("tab10")
    others = [k for k in predictions if k != proposed_key]
    for i, name in enumerate(others):
        ax.plot(
            t, predictions[name][sample_idx, :, category],
            label=name, color=cmap(i % 10), alpha=0.75, linewidth=1.1,
        )
    if proposed_key in predictions:
        ax.plot(
            t, predictions[proposed_key][sample_idx, :, category],
            label="Proposed", color="black", linestyle="--", linewidth=2.0,
        )
    cat_label = category_names[category] if category < len(category_names) else f"Category {category}"
    ax.set_title(f"Figure 3 — {cat_label}: proposed vs. comparative methods")
    ax.set_xlabel("Forecast step (h)")
    ax.set_ylabel("Normalized load")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_metric_bar(
    results: dict[str, dict[str, float]],
    metric: str = "RMSE",
    out_path: str | Path = "experiments/metric_bar.png",
) -> Path:
    """Bar chart comparing one metric across all methods (handy companion to
    the Table 1 reproduction)."""
    names = [n for n, m in results.items() if metric in m]
    vals = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["tab:green" if n == "proposed" else "tab:blue" for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparative methods — {metric}")
    plt.xticks(rotation=30, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
