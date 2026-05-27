"""Generate Figure 2 and Figure 3 from one or more trained model checkpoints.

Usage::

    # After running experiments/compare.py to populate per-model predictions:
    python experiments/plot_figures.py --predictions experiments/predictions.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.visualize import (  # noqa: E402
    plot_method_comparison,
    plot_metric_bar,
    plot_predictions_vs_true,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="experiments/predictions.npz",
                   help="npz with keys: y_true, <method_name>=preds, ...")
    p.add_argument("--results", default="experiments/results.json",
                   help="JSON of metrics per method (from compare.py)")
    p.add_argument("--out_dir", default="experiments")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--category", type=int, default=0)
    args = p.parse_args()

    pred_file = Path(args.predictions)
    if not pred_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_file}. "
            "Run experiments/compare.py with --save_predictions first."
        )
    data = np.load(pred_file)
    y_true = data["y_true"]
    methods = [k for k in data.files if k != "y_true"]
    preds = {k: data[k] for k in methods}

    out_dir = Path(args.out_dir)
    if "proposed" in preds:
        f2 = plot_predictions_vs_true(
            y_true, preds["proposed"],
            out_path=out_dir / "figure2.png",
            sample_idx=args.sample_idx,
        )
        print(f"Figure 2 → {f2}")

    f3 = plot_method_comparison(
        y_true, preds,
        out_path=out_dir / "figure3.png",
        sample_idx=args.sample_idx,
        category=args.category,
    )
    print(f"Figure 3 → {f3}")

    results_path = Path(args.results)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        for metric in ("MAPE", "RMSE", "MAE", "R2"):
            out = plot_metric_bar(results, metric=metric,
                                   out_path=out_dir / f"metric_{metric.lower()}.png")
            print(f"Bar chart ({metric}) → {out}")


if __name__ == "__main__":
    main()
