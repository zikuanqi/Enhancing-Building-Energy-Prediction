"""Multi-scale temporal decomposition for building energy time series.

Separates raw load into:
  - long-term trend (low-frequency moving average)
  - seasonal component (yearly cycle, removed by long-window centred mean)
  - weekly cycle (period = 168 hours)
  - daily fluctuation (period = 24 hours)
  - short-term residual

Implemented as a differentiable PyTorch module so it can sit inside the model
graph, but it can also be applied off-line on numpy arrays.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _moving_average_1d(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Centred moving average along the time axis (dim=1).

    x: (B, T, C). Pads with edge-replication so output length matches T.
    """
    if kernel <= 1:
        return x
    pad = kernel // 2
    # Conv1d wants (B, C, T)
    xt = x.transpose(1, 2)
    xt = F.pad(xt, (pad, kernel - 1 - pad), mode="replicate")
    weight = torch.full((xt.size(1), 1, kernel), 1.0 / kernel, device=x.device, dtype=x.dtype)
    out = F.conv1d(xt, weight, groups=xt.size(1))
    return out.transpose(1, 2)


class MultiScaleDecomposition(nn.Module):
    """Decompose a sequence into multi-scale components.

    Returns a dict of ``trend``, ``seasonal``, ``weekly``, ``daily``,
    ``short_term`` — each the same shape as ``x``.

    The decomposition is hierarchical: longer scales are stripped first so
    each remaining component is the residual against coarser scales.
    """

    def __init__(
        self,
        daily_period: int = 24,
        weekly_period: int = 168,
        seasonal_period: int = 24 * 30,
        trend_period: int = 24 * 30 * 6,
    ) -> None:
        super().__init__()
        self.daily_period = daily_period
        self.weekly_period = weekly_period
        self.seasonal_period = seasonal_period
        self.trend_period = trend_period

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        T = x.size(1)
        trend = _moving_average_1d(x, min(self.trend_period, max(T, 1)))
        no_trend = x - trend

        seasonal = _moving_average_1d(no_trend, min(self.seasonal_period, max(T, 1)))
        no_season = no_trend - seasonal

        weekly = _moving_average_1d(no_season, min(self.weekly_period, max(T, 1)))
        no_week = no_season - weekly

        daily = _moving_average_1d(no_week, min(self.daily_period, max(T, 1)))
        short_term = no_week - daily

        return {
            "trend": trend,
            "seasonal": seasonal,
            "weekly": weekly,
            "daily": daily,
            "short_term": short_term,
        }
