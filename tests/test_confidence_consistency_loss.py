import torch

from hlm_ecg.losses.confidence_consistency import (
    ConfidenceConsistencyLossConfig,
    ConfidenceWeightedConsistencyLoss,
    confidence_weights_from_probs,
)


def test_confidence_weight_shape_and_range():
    p = torch.tensor([[0.5, 0.0, 1.0, 0.25, 0.75]])
    w = confidence_weights_from_probs(p, gamma=1.0)
    assert tuple(w.shape) == (1, 5)
    assert torch.all(w >= 0)
    assert torch.all(w <= 1)


def test_confidence_weight_extremes():
    p = torch.tensor([[0.5, 0.01, 0.99]])
    w = confidence_weights_from_probs(p, gamma=1.0)
    assert torch.isclose(w[0, 0], torch.tensor(0.0))
    assert w[0, 1] > 0.95
    assert w[0, 2] > 0.95


def test_consistency_loss_uses_stop_gradient_teacher():
    logits_full = torch.tensor([[4.0, -4.0, 0.0, 2.0, -2.0]], requires_grad=True)
    logits_mask = torch.zeros(1, 5, requires_grad=True)
    criterion = ConfidenceWeightedConsistencyLoss(
        ConfidenceConsistencyLossConfig(lambda_cons=0.1, gamma=1.0, enabled=True)
    )
    info = criterion(logits_full, logits_mask)
    info["cw_consistency_loss"].backward()
    assert logits_full.grad is None
    assert logits_mask.grad is not None
    assert info["cw_consistency_loss"] > 0


def test_lambda_zero_or_disabled_consistency_returns_zero_loss():
    logits_full = torch.randn(2, 5)
    logits_mask = torch.randn(2, 5)
    criterion = ConfidenceWeightedConsistencyLoss(
        ConfidenceConsistencyLossConfig(lambda_cons=0.0, gamma=1.0, enabled=True)
    )
    info = criterion(logits_full, logits_mask)
    assert torch.allclose(info["cw_consistency_loss"], torch.tensor(0.0))

    disabled = ConfidenceWeightedConsistencyLoss(
        ConfidenceConsistencyLossConfig(lambda_cons=0.1, gamma=1.0, enabled=False)
    )
    disabled_info = disabled(logits_full, logits_mask)
    assert torch.allclose(disabled_info["cw_consistency_loss"], torch.tensor(0.0))


def test_soft_target_bce_runs_and_reports_diagnostics():
    logits_full = torch.randn(3, 5)
    logits_mask = torch.randn(3, 5)
    criterion = ConfidenceWeightedConsistencyLoss()
    info = criterion(logits_full, logits_mask)
    assert set(info) == {
        "cw_consistency_loss",
        "mean_consistency_weight",
        "min_consistency_weight",
        "max_consistency_weight",
        "full_mean_confidence",
        "masked_mean_confidence",
    }
    assert info["cw_consistency_loss"] >= 0
    assert 0 <= info["mean_consistency_weight"] <= 1
    assert 0 <= info["min_consistency_weight"] <= 1
    assert 0 <= info["max_consistency_weight"] <= 1
