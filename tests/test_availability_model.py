import pytest
import torch

from hlm_ecg.models.resnet1d import ResNet1D
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.training.train_baseline import build_model


def test_resnet1d_availability_forward_shape():
    model = ResNet1DAvailability(base_channels=8, layers=(1, 1, 1, 1))
    x = torch.randn(2, 12, 1000)
    mask = torch.ones(2, 12)
    logits = model(x, availability_mask=mask)
    assert tuple(logits.shape) == (2, 5)


def test_resnet1d_availability_requires_mask():
    model = ResNet1DAvailability(base_channels=8, layers=(1, 1, 1, 1))
    x = torch.randn(2, 12, 1000)
    with pytest.raises(ValueError, match="requires availability_mask"):
        model(x)


def test_resnet1d_original_forward_behavior_unchanged():
    model = ResNet1D(base_channels=8, layers=(1, 1, 1, 1))
    x = torch.randn(2, 12, 1000)
    logits = model(x)
    assert tuple(logits.shape) == (2, 5)


def test_build_model_uses_availability_only_when_configured():
    plain = build_model({"model": {"base_channels": 8, "layers": [1, 1, 1, 1]}})
    avail = build_model(
        {
            "model": {
                "base_channels": 8,
                "layers": [1, 1, 1, 1],
                "use_availability_embedding": True,
                "availability_embedding_dim": 32,
                "mask_mlp_hidden_dim": 32,
            }
        }
    )
    assert not getattr(plain, "requires_availability_mask", False)
    assert getattr(avail, "requires_availability_mask", False)
