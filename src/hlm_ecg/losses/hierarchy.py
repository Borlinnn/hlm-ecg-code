"""Parent-child hierarchy consistency loss for diagnostic subclasses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER


def load_parent_indices(
    *,
    vocab_path: Path | str,
    mapping_path: Path | str,
    label_order: Sequence[str] = LABEL_ORDER,
) -> tuple[int, ...]:
    """Load subclass -> superclass parent indices from saved audit artifacts."""
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    subclasses = vocab.get("subclasses")
    if not isinstance(subclasses, list) or not subclasses:
        raise RuntimeError(f"Invalid subclass vocabulary at {vocab_path}")
    if list(vocab.get("label_order", label_order)) != list(label_order):
        raise RuntimeError("Subclass vocabulary label order does not match fixed superclass order")

    mapping_data = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if mapping_data.get("mapping_unique") is not True:
        raise RuntimeError(f"Subclass parent mapping is not unique at {mapping_path}")
    rows = mapping_data.get("mapping")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Invalid subclass parent mapping at {mapping_path}")

    label_to_idx = {label: idx for idx, label in enumerate(label_order)}
    parent_by_subclass: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("Parent mapping rows must be objects")
        subclass = str(row.get("diagnostic_subclass", ""))
        parent = str(row.get("parent_superclass", ""))
        if not subclass:
            raise RuntimeError("Parent mapping contains an empty subclass name")
        if parent not in label_to_idx:
            raise RuntimeError(f"Parent superclass {parent!r} is outside fixed label order")
        if bool(row.get("parent_valid", True)) is not True:
            raise RuntimeError(f"Parent superclass for subclass {subclass!r} is marked invalid")
        previous = parent_by_subclass.get(subclass)
        if previous is not None and previous != parent:
            raise RuntimeError(f"Subclass {subclass!r} maps to multiple parents: {previous!r}, {parent!r}")
        parent_by_subclass[subclass] = parent

    missing = [str(subclass) for subclass in subclasses if str(subclass) not in parent_by_subclass]
    if missing:
        raise RuntimeError(f"Missing parent mapping for subclasses: {missing}")
    return tuple(label_to_idx[parent_by_subclass[str(subclass)]] for subclass in subclasses)


class ParentChildHierarchyLoss(nn.Module):
    """Penalize subclass probability above its parent superclass probability."""

    def __init__(self, parent_indices: Sequence[int], *, violation_eps: float = 0.0) -> None:
        super().__init__()
        parent = torch.as_tensor(list(parent_indices), dtype=torch.long)
        if parent.ndim != 1 or int(parent.numel()) == 0:
            raise ValueError("parent_indices must be a non-empty 1D sequence")
        if int(parent.min().item()) < 0 or int(parent.max().item()) >= len(LABEL_ORDER):
            raise ValueError("parent_indices must point into the fixed 5-superclass label order")
        self.register_buffer("parent_indices", parent, persistent=False)
        self.violation_eps = float(violation_eps)

    def forward(self, logits_super: torch.Tensor, logits_sub: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits_super.ndim != 2:
            raise ValueError(f"logits_super must have shape (batch, 5), got {tuple(logits_super.shape)}")
        if logits_sub.ndim != 2:
            raise ValueError(f"logits_sub must have shape (batch, K), got {tuple(logits_sub.shape)}")
        if logits_super.shape[1] != len(LABEL_ORDER):
            raise ValueError(f"logits_super second dimension must be {len(LABEL_ORDER)}")
        if logits_sub.shape[1] != int(self.parent_indices.numel()):
            raise ValueError("logits_sub subclass dimension does not match parent_indices")

        p_super = torch.sigmoid(logits_super)
        p_sub = torch.sigmoid(logits_sub)
        parent_probs = p_super.index_select(dim=1, index=self.parent_indices.to(device=logits_super.device))
        margin = torch.relu(p_sub - parent_probs)
        violation = p_sub > (parent_probs + self.violation_eps)
        return {
            "loss_hier": margin.square().mean(),
            "violation_rate": violation.to(dtype=torch.float32).mean(),
            "mean_violation_margin": margin.mean(),
            "max_violation_margin": margin.max(),
        }
