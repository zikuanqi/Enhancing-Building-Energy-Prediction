"""MatMul-free dense layer with ternary weights — Section 2.5.

Replaces dense multiplications by sign-based accumulations:

    Ω_{j,i} ∈ {-1, 0, +1}
    Γ_i = Σ_{j: Ω=+1} v_j  -  Σ_{j: Ω=-1} v_j                        (10)

Continuous shadow weights are quantized to ternary using adaptive thresholds:

    Q(M) = +1 if M > τ_+;  -1 if M < τ_-;  0 otherwise               (11)

with τ_± = ± α · std(M).  Backward uses the straight-through estimator.

For full math-equivalence to Eq. (10), the layer falls back to a normal
``F.linear`` at forward time; the ternary substitution is exploited at
inference / deployment. This keeps gradients well-defined and lets the same
module be used in baselines for fair comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TernaryQuantize(torch.autograd.Function):
    """Adaptive-threshold ternary quantizer with straight-through gradient."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, alpha: float) -> torch.Tensor:
        std = weight.detach().abs().mean()  # robust scale (Eq. 11)
        tau = alpha * std
        ternary = torch.zeros_like(weight)
        ternary[weight > tau] = 1.0
        ternary[weight < -tau] = -1.0
        return ternary

    @staticmethod
    def backward(ctx, grad_out):
        # Straight-through: gradient of identity, clipped to [-1, 1] for stability.
        return grad_out.clamp(-1.0, 1.0), None


def ternary_quantize(weight: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    return _TernaryQuantize.apply(weight, alpha)


class MatMulFreeDense(nn.Module):
    """Linear-like layer using ternary-quantized weights and a learnable
    per-output scale.

    Forward (training): ``y = scale ⊙ (x @ Q(W).T) + bias``  — quantization
    is the only non-standard step; matmul is still used so PyTorch can run on
    GPU/CPU. At deployment time ``Q(W)`` can be exported and the matmul
    replaced by the additive accumulator of Eq. (10).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 0.7,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = ternary_quantize(self.weight, self.alpha)
        y = F.linear(x, w_q, bias=None) * self.scale
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def quantized_weight(self) -> torch.Tensor:
        """Return the deployable ternary weight matrix (no grad)."""
        return ternary_quantize(self.weight, self.alpha)

    def ternary_accumulate(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (10) — explicit additive form, useful for verification."""
        w_q = self.quantized_weight()
        pos_mask = (w_q == 1).float()
        neg_mask = (w_q == -1).float()
        pos = F.linear(x, pos_mask, bias=None)
        neg = F.linear(x, neg_mask, bias=None)
        y = (pos - neg) * self.scale
        if self.bias is not None:
            y = y + self.bias
        return y
