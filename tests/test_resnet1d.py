import torch

from hlm_ecg.models.resnet1d import ResNet1D


def test_resnet1d_forward_shape():
    model = ResNet1D(base_channels=8, layers=(1, 1, 1, 1))
    x = torch.randn(2, 12, 1000)
    logits = model(x)
    assert tuple(logits.shape) == (2, 5)
