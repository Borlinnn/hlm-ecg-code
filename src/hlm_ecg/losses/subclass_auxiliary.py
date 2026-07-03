"""Loss for subclass auxiliary supervision and optional hierarchy consistency."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from hlm_ecg.losses.hierarchy import ParentChildHierarchyLoss


@dataclass(frozen=True)
class SubclassAuxiliaryLossConfig:
    lambda_sub: float = 0.2
    ignore_only_dropped: bool = True
    use_hierarchy_loss: bool = False
    lambda_hier: float = 0.0
    hierarchy_parent_indices: Sequence[int] | None = None
    hierarchy_violation_eps: float = 0.0


class SubclassAuxiliaryLoss(nn.Module):
    """Compute L = L_super + lambda_sub * L_sub, optionally plus hierarchy loss."""

    def __init__(self, config: SubclassAuxiliaryLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or SubclassAuxiliaryLossConfig()
        self.super_loss = nn.BCEWithLogitsLoss()
        self.sub_loss_none = nn.BCEWithLogitsLoss(reduction="none")
        self.hierarchy_loss = None
        if self.config.use_hierarchy_loss:
            if self.config.hierarchy_parent_indices is None:
                raise RuntimeError("use_hierarchy_loss=true requires hierarchy_parent_indices")
            self.hierarchy_loss = ParentChildHierarchyLoss(
                self.config.hierarchy_parent_indices,
                violation_eps=float(self.config.hierarchy_violation_eps),
            )

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "logits_super" not in outputs or "logits_sub" not in outputs:
            raise RuntimeError("SubclassAuxiliaryLoss requires logits_super and logits_sub")
        y_super = batch["y"].to(device=outputs["logits_super"].device, dtype=torch.float32)
        y_sub = batch["y_sub"].to(device=outputs["logits_sub"].device, dtype=torch.float32)
        loss_super = self.super_loss(outputs["logits_super"], y_super)

        if float(self.config.lambda_sub) == 0.0:
            loss_sub = outputs["logits_sub"].sum() * 0.0
        else:
            per_entry = self.sub_loss_none(outputs["logits_sub"], y_sub)
            per_sample = per_entry.mean(dim=1)
            if self.config.ignore_only_dropped:
                dropped = batch.get("has_only_dropped_subclass")
                if dropped is None:
                    raise RuntimeError("Missing has_only_dropped_subclass for subclass loss masking")
                weights = (1.0 - dropped.to(device=per_sample.device, dtype=torch.float32)).clamp(min=0.0, max=1.0)
                denom = weights.sum().clamp(min=1.0)
                loss_sub = (per_sample * weights).sum() / denom
            else:
                loss_sub = per_sample.mean()

        total = loss_super + float(self.config.lambda_sub) * loss_sub
        info = {
            "loss": total,
            "loss_super": loss_super.detach(),
            "loss_sub": loss_sub.detach(),
        }
        if self.hierarchy_loss is not None:
            hier_info = self.hierarchy_loss(outputs["logits_super"], outputs["logits_sub"])
            total = total + float(self.config.lambda_hier) * hier_info["loss_hier"]
            info["loss"] = total
            info["loss_hier"] = hier_info["loss_hier"].detach()
            info["violation_rate"] = hier_info["violation_rate"].detach()
            info["mean_violation_margin"] = hier_info["mean_violation_margin"].detach()
            info["max_violation_margin"] = hier_info["max_violation_margin"].detach()
        return info
