"""Confidence-weighted full-to-masked consistency loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ConfidenceConsistencyLossConfig:
    lambda_cons: float = 0.1
    gamma: float = 1.0
    enabled: bool = True


def confidence_weights_from_probs(p_full: torch.Tensor, *, gamma: float = 1.0) -> torch.Tensor:
    """Return detached confidence weights in [0, 1] from full-view probabilities."""
    weights = (2.0 * torch.abs(p_full.detach() - 0.5)).clamp(min=0.0, max=1.0)
    return weights.pow(float(gamma)).detach()


class ConfidenceWeightedConsistencyLoss(nn.Module):
    """Soft-target BCE from full-view teacher probabilities to masked-view logits."""

    def __init__(self, config: ConfidenceConsistencyLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConfidenceConsistencyLossConfig()

    def forward(self, logits_super_full: torch.Tensor, logits_super_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits_super_full.shape != logits_super_mask.shape:
            raise ValueError("Full and masked superclass logits must have the same shape")
        if logits_super_full.ndim != 2:
            raise ValueError("Superclass logits must have shape (batch, classes)")

        if not self.config.enabled or float(self.config.lambda_cons) == 0.0:
            zero = logits_super_mask.sum() * 0.0
            weights = torch.zeros_like(logits_super_mask)
            p_full = torch.sigmoid(logits_super_full.detach())
            p_mask = torch.sigmoid(logits_super_mask.detach())
            return {
                "cw_consistency_loss": zero,
                "mean_consistency_weight": weights.mean().detach(),
                "min_consistency_weight": weights.min().detach(),
                "max_consistency_weight": weights.max().detach(),
                "full_mean_confidence": (2.0 * torch.abs(p_full - 0.5)).mean().detach(),
                "masked_mean_confidence": (2.0 * torch.abs(p_mask - 0.5)).mean().detach(),
            }

        p_full = torch.sigmoid(logits_super_full)
        target = p_full.detach()
        weights = confidence_weights_from_probs(p_full, gamma=float(self.config.gamma))
        per_label = F.binary_cross_entropy_with_logits(logits_super_mask, target, reduction="none")
        loss = (per_label * weights).mean()
        p_mask = torch.sigmoid(logits_super_mask.detach())
        return {
            "cw_consistency_loss": loss,
            "mean_consistency_weight": weights.mean().detach(),
            "min_consistency_weight": weights.min().detach(),
            "max_consistency_weight": weights.max().detach(),
            "full_mean_confidence": (2.0 * torch.abs(target - 0.5)).mean().detach(),
            "masked_mean_confidence": (2.0 * torch.abs(p_mask - 0.5)).mean().detach(),
        }
