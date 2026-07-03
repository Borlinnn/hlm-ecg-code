"""Backward-compatible availability-conditioned model wrapper."""

from __future__ import annotations

from hlm_ecg.models.task_wrappers import AvailabilityConditionedClassifier


class ResNet1DAvailability(AvailabilityConditionedClassifier):
    """Availability-conditioned classifier.

    The historical class name is kept for compatibility. New configs may set
    `architecture` to any supported backbone; old configs default to
    `resnet1d_tiny`.
    """

    def __init__(
        self,
        *,
        in_channels: int = 12,
        num_classes: int = 5,
        base_channels: int = 32,
        layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        kernel_size: int = 7,
        availability_embedding_dim: int = 32,
        mask_mlp_hidden_dim: int = 32,
        use_subclass_auxiliary: bool = False,
        num_subclasses: int | None = None,
        architecture: str = "resnet1d_tiny",
        inception_depth: int = 6,
        inception_bottleneck_channels: int | None = None,
        use_learnable_mask_token: bool = False,
        signal_length: int = 1000,
    ) -> None:
        model_cfg = {
            "architecture": architecture,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base_channels": base_channels,
            "layers": list(layers),
            "kernel_size": kernel_size,
            "availability_embedding_dim": availability_embedding_dim,
            "mask_mlp_hidden_dim": mask_mlp_hidden_dim,
            "enable_subclass_auxiliary": use_subclass_auxiliary,
            "num_subclasses": num_subclasses,
            "inception_depth": inception_depth,
            "inception_bottleneck_channels": (
                base_channels if inception_bottleneck_channels is None else inception_bottleneck_channels
            ),
            "use_learnable_mask_token": use_learnable_mask_token,
            "signal_length": signal_length,
        }
        super().__init__(model_cfg=model_cfg)
