"""Task wrappers for backbone-agnostic HLM-ECG heads."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from hlm_ecg.models.availability_embedding import AvailabilityEmbedding
from hlm_ecg.models.backbones import build_feature_backbone
from hlm_ecg.models.mask_token import LearnableLeadMaskToken
from hlm_ecg.models.multitask_heads import SubclassAuxiliaryHead


class MaskTokenBackboneClassifier(nn.Module):
    """Superclass classifier that replaces missing leads with learnable tokens."""

    requires_availability_mask = True

    def __init__(self, *, model_cfg: Mapping[str, object]) -> None:
        super().__init__()
        self.encoder = build_feature_backbone(model_cfg)
        self.mask_token = LearnableLeadMaskToken(
            num_leads=int(model_cfg.get("in_channels", 12)),
            signal_length=int(model_cfg.get("signal_length", 1000)),
        )

    @property
    def feature_dim(self) -> int:
        return int(getattr(self.encoder, "feature_dim"))

    def encode_features(self, x: torch.Tensor, availability_mask: torch.Tensor | None = None) -> torch.Tensor:
        if availability_mask is None:
            raise ValueError("MaskTokenBackboneClassifier requires availability_mask")
        return self.encoder.encode_features(self.mask_token(x, availability_mask))

    def forward(self, x: torch.Tensor, availability_mask: torch.Tensor | None = None) -> torch.Tensor:
        if availability_mask is None:
            raise ValueError("MaskTokenBackboneClassifier requires availability_mask")
        return self.encoder(self.mask_token(x, availability_mask))


class BackboneSubclassClassifier(nn.Module):
    """Backbone classifier with a superclass and subclass auxiliary head."""

    def __init__(self, *, model_cfg: Mapping[str, object]) -> None:
        super().__init__()
        num_subclasses = model_cfg.get("num_subclasses")
        if num_subclasses is None:
            raise ValueError("num_subclasses is required when enable_subclass_auxiliary=true")
        self.requires_availability_mask = bool(model_cfg.get("use_learnable_mask_token", False))
        self.encoder = build_feature_backbone(model_cfg)
        self.mask_token = (
            LearnableLeadMaskToken(
                num_leads=int(model_cfg.get("in_channels", 12)),
                signal_length=int(model_cfg.get("signal_length", 1000)),
            )
            if self.requires_availability_mask
            else None
        )
        self.fc = nn.Linear(int(getattr(self.encoder, "feature_dim")), int(model_cfg.get("num_classes", 5)))
        self.subclass_head = SubclassAuxiliaryHead(
            in_features=int(getattr(self.encoder, "feature_dim")),
            num_subclasses=int(num_subclasses),
        )
        self.feature_dim = int(getattr(self.encoder, "feature_dim"))

    def encode_features(self, x: torch.Tensor, availability_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mask_token is not None:
            if availability_mask is None:
                raise ValueError("BackboneSubclassClassifier requires availability_mask")
            x = self.mask_token(x, availability_mask)
        return self.encoder.encode_features(x)

    def forward(self, x: torch.Tensor, availability_mask: torch.Tensor | None = None) -> Mapping[str, torch.Tensor]:
        features = self.encode_features(x, availability_mask)
        return {
            "logits_super": self.fc(features),
            "logits_sub": self.subclass_head(features),
        }


class AvailabilityConditionedClassifier(nn.Module):
    """Backbone classifier with explicit lead-availability conditioning."""

    requires_availability_mask = True

    def __init__(self, *, model_cfg: Mapping[str, object]) -> None:
        super().__init__()
        self.use_subclass_auxiliary = bool(model_cfg.get("enable_subclass_auxiliary", False))
        self.encoder = build_feature_backbone(model_cfg)
        self.mask_token = (
            LearnableLeadMaskToken(
                num_leads=int(model_cfg.get("in_channels", 12)),
                signal_length=int(model_cfg.get("signal_length", 1000)),
            )
            if bool(model_cfg.get("use_learnable_mask_token", False))
            else None
        )
        availability_embedding_dim = int(model_cfg.get("availability_embedding_dim", 32))
        self.availability_embedding = AvailabilityEmbedding(
            num_leads=int(model_cfg.get("in_channels", 12)),
            hidden_dim=int(model_cfg.get("mask_mlp_hidden_dim", 32)),
            embedding_dim=availability_embedding_dim,
        )
        self.feature_dim = int(getattr(self.encoder, "feature_dim")) + availability_embedding_dim
        self.fc = nn.Linear(self.feature_dim, int(model_cfg.get("num_classes", 5)))
        if self.use_subclass_auxiliary:
            num_subclasses = model_cfg.get("num_subclasses")
            if num_subclasses is None:
                raise ValueError("num_subclasses is required when enable_subclass_auxiliary=true")
            self.subclass_head = SubclassAuxiliaryHead(
                in_features=self.feature_dim,
                num_subclasses=int(num_subclasses),
            )
        else:
            self.subclass_head = None

    def encode_features(self, x: torch.Tensor, availability_mask: torch.Tensor | None = None) -> torch.Tensor:
        if availability_mask is None:
            raise ValueError("AvailabilityConditionedClassifier requires availability_mask")
        if self.mask_token is not None:
            x = self.mask_token(x, availability_mask)
        h = self.encoder.encode_features(x)
        e_m = self.availability_embedding(availability_mask)
        return torch.cat([h, e_m], dim=1)

    def forward(self, x: torch.Tensor, availability_mask: torch.Tensor | None = None):
        features = self.encode_features(x, availability_mask)
        logits_super = self.fc(features)
        if self.subclass_head is None:
            return logits_super
        return {
            "logits_super": logits_super,
            "logits_sub": self.subclass_head(features),
        }
