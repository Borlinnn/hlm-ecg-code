import pytest
import torch

from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability


def test_subclass_aux_model_outputs_super_and_subclass_logits():
    model = ResNet1DAvailability(
        base_channels=8,
        layers=(1, 1, 1, 1),
        use_subclass_auxiliary=True,
        num_subclasses=7,
    )
    out = model(torch.randn(2, 12, 1000), availability_mask=torch.ones(2, 12))
    assert tuple(out["logits_super"].shape) == (2, 5)
    assert tuple(out["logits_sub"].shape) == (2, 7)


def test_availability_model_without_subclass_auxiliary_keeps_old_behavior():
    model = ResNet1DAvailability(base_channels=8, layers=(1, 1, 1, 1))
    out = model(torch.randn(2, 12, 1000), availability_mask=torch.ones(2, 12))
    assert tuple(out.shape) == (2, 5)


def test_subclass_aux_model_requires_num_subclasses():
    with pytest.raises(ValueError, match="num_subclasses"):
        ResNet1DAvailability(base_channels=8, layers=(1, 1, 1, 1), use_subclass_auxiliary=True)
