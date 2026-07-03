import torch

from hlm_ecg.models.mask_token import LearnableLeadMaskToken
from hlm_ecg.training.train_baseline import build_model, forward_model


def test_new_backbones_expose_feature_encoder_contract():
    for architecture in ("xresnet1d101_like", "inception_time1d"):
        model = build_model(
            {
                "model": {
                    "architecture": architecture,
                    "base_channels": 4,
                    "layers": [1, 1, 1, 1],
                    "inception_depth": 2,
                    "inception_bottleneck_channels": 4,
                    "num_classes": 5,
                }
            }
        )
        x = torch.randn(2, 12, 1000)
        features = model.encode_features(x)
        logits = model(x)
        assert features.shape == (2, model.feature_dim)
        assert logits.shape == (2, 5)


def test_build_model_preserves_old_default_architecture():
    model = build_model({"model": {"base_channels": 4, "layers": [1, 1, 1, 1]}})
    assert model.__class__.__name__ == "ResNet1D"
    assert model(torch.randn(2, 12, 1000)).shape == (2, 5)


def test_availability_and_subclass_wrappers_work_for_all_architectures():
    for architecture in ("resnet1d_tiny", "xresnet1d101_like", "inception_time1d"):
        model = build_model(
            {
                "model": {
                    "architecture": architecture,
                    "base_channels": 4,
                    "layers": [1, 1, 1, 1],
                    "inception_depth": 2,
                    "inception_bottleneck_channels": 4,
                    "use_availability_embedding": True,
                    "enable_subclass_auxiliary": True,
                    "num_subclasses": 3,
                    "availability_embedding_dim": 5,
                    "mask_mlp_hidden_dim": 6,
                }
            }
        )
        batch = {"x": torch.randn(2, 12, 1000), "availability_mask": torch.ones(2, 12)}
        outputs = model(batch["x"], availability_mask=batch["availability_mask"])
        logits = forward_model(model, batch, device=torch.device("cpu"))
        assert outputs["logits_super"].shape == (2, 5)
        assert outputs["logits_sub"].shape == (2, 3)
        assert logits.shape == (2, 5)


def test_subclass_without_availability_is_supported():
    model = build_model(
        {
            "model": {
                "architecture": "xresnet1d101_like",
                "base_channels": 4,
                "layers": [1, 1, 1, 1],
                "enable_subclass_auxiliary": True,
                "num_subclasses": 4,
            }
        }
    )
    batch = {"x": torch.randn(2, 12, 1000)}
    outputs = model(batch["x"])
    logits = forward_model(model, batch, device=torch.device("cpu"))
    assert outputs["logits_super"].shape == (2, 5)
    assert outputs["logits_sub"].shape == (2, 4)
    assert logits.shape == (2, 5)


def test_learnable_mask_token_only_changes_missing_leads():
    module = LearnableLeadMaskToken(num_leads=12, signal_length=1000)
    with torch.no_grad():
        module.token.fill_(2.0)
    x = torch.randn(2, 12, 1000)
    mask = torch.ones(2, 12)
    mask[:, 3] = 0.0
    out = module(x, mask)
    assert torch.allclose(out[:, :3], x[:, :3])
    assert torch.allclose(out[:, 4:], x[:, 4:])
    assert torch.allclose(out[:, 3], torch.full_like(out[:, 3], 2.0))
