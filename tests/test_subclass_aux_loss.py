import torch

from hlm_ecg.losses.subclass_auxiliary import SubclassAuxiliaryLoss, SubclassAuxiliaryLossConfig


def _outputs():
    return {
        "logits_super": torch.zeros(2, 5),
        "logits_sub": torch.zeros(2, 3),
    }


def _batch():
    return {
        "y": torch.zeros(2, 5),
        "y_sub": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "has_only_dropped_subclass": torch.tensor([0.0, 1.0]),
    }


def test_lambda_zero_loss_equals_superclass_only():
    criterion = SubclassAuxiliaryLoss(SubclassAuxiliaryLossConfig(lambda_sub=0.0))
    info = criterion(_outputs(), _batch())
    super_only = torch.nn.BCEWithLogitsLoss()(_outputs()["logits_super"], _batch()["y"])
    assert torch.allclose(info["loss"], super_only)


def test_only_dropped_samples_are_masked_for_subclass_loss():
    criterion = SubclassAuxiliaryLoss(SubclassAuxiliaryLossConfig(lambda_sub=1.0, ignore_only_dropped=True))
    info = criterion(_outputs(), _batch())
    per = torch.nn.BCEWithLogitsLoss(reduction="none")(_outputs()["logits_sub"], _batch()["y_sub"]).mean(dim=1)
    expected_sub = per[0]
    expected_super = torch.nn.BCEWithLogitsLoss()(_outputs()["logits_super"], _batch()["y"])
    assert torch.allclose(info["loss"], expected_super + expected_sub)


def test_disabled_hierarchy_and_consistency_are_not_loss_terms():
    criterion = SubclassAuxiliaryLoss(SubclassAuxiliaryLossConfig(lambda_sub=0.2))
    info = criterion(_outputs(), _batch())
    assert set(info) == {"loss", "loss_super", "loss_sub"}
