import json

import pytest
import torch

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.losses.hierarchy import ParentChildHierarchyLoss, load_parent_indices


def _logit(values):
    return torch.logit(torch.tensor(values, dtype=torch.float32))


def test_parent_indices_loaded_from_mapping_not_hardcoded(tmp_path):
    vocab = {
        "subclasses": ["sub_mi", "sub_hyp", "sub_cd"],
        "label_order": list(LABEL_ORDER),
    }
    mapping = {
        "mapping_unique": True,
        "mapping": [
            {"diagnostic_subclass": "sub_mi", "parent_superclass": "MI", "parent_valid": True},
            {"diagnostic_subclass": "sub_hyp", "parent_superclass": "HYP", "parent_valid": True},
            {"diagnostic_subclass": "sub_cd", "parent_superclass": "CD", "parent_valid": True},
        ],
    }
    vocab_path = tmp_path / "vocab.json"
    mapping_path = tmp_path / "mapping.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    assert load_parent_indices(vocab_path=vocab_path, mapping_path=mapping_path) == (1, 4, 3)


def test_parent_mapping_requires_unique_parent(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    mapping_path = tmp_path / "mapping.json"
    vocab_path.write_text(json.dumps({"subclasses": ["x"], "label_order": list(LABEL_ORDER)}), encoding="utf-8")
    mapping_path.write_text(
        json.dumps(
            {
                "mapping_unique": True,
                "mapping": [
                    {"diagnostic_subclass": "x", "parent_superclass": "MI", "parent_valid": True},
                    {"diagnostic_subclass": "x", "parent_superclass": "CD", "parent_valid": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="multiple parents"):
        load_parent_indices(vocab_path=vocab_path, mapping_path=mapping_path)


def test_hierarchy_loss_zero_when_subclass_not_above_parent():
    criterion = ParentChildHierarchyLoss(parent_indices=[0, 1])
    logits_super = _logit([[0.8, 0.6, 0.2, 0.2, 0.2]])
    logits_sub = _logit([[0.2, 0.5]])
    info = criterion(logits_super, logits_sub)
    assert torch.allclose(info["loss_hier"], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(info["violation_rate"], torch.tensor(0.0), atol=1e-7)


def test_hierarchy_loss_positive_when_subclass_above_parent():
    criterion = ParentChildHierarchyLoss(parent_indices=[0, 1])
    logits_super = _logit([[0.2, 0.4, 0.2, 0.2, 0.2]])
    logits_sub = _logit([[0.5, 0.7]])
    info = criterion(logits_super, logits_sub)
    expected = (((0.5 - 0.2) ** 2) + ((0.7 - 0.4) ** 2)) / 2.0
    assert torch.allclose(info["loss_hier"], torch.tensor(expected), atol=1e-6)
    assert torch.allclose(info["violation_rate"], torch.tensor(1.0), atol=1e-7)
    assert info["mean_violation_margin"] > 0
    assert info["max_violation_margin"] > 0


def test_hierarchy_loss_vectorized_batch_shape_and_values():
    criterion = ParentChildHierarchyLoss(parent_indices=[0, 1])
    logits_super = _logit([[0.8, 0.2, 0.2, 0.2, 0.2], [0.1, 0.9, 0.2, 0.2, 0.2]])
    logits_sub = _logit([[0.7, 0.6], [0.5, 0.8]])
    info = criterion(logits_super, logits_sub)
    margins = torch.tensor([[0.0, 0.4], [0.4, 0.0]])
    assert torch.allclose(info["loss_hier"], margins.square().mean(), atol=1e-6)
    assert torch.allclose(info["violation_rate"], torch.tensor(0.5), atol=1e-7)
