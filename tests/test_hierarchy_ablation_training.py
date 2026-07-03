import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.subclass_labels import build_kept_subclass_artifacts
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.losses.hierarchy import load_parent_indices
from hlm_ecg.losses.subclass_auxiliary import SubclassAuxiliaryLoss, SubclassAuxiliaryLossConfig
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.training.train_subclass_auxiliary import build_subclass_criterion, run_subclass_epoch

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


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


def test_lambda_hier_zero_total_loss_equals_a4a():
    outputs = _outputs()
    batch = _batch()
    a4a = SubclassAuxiliaryLoss(SubclassAuxiliaryLossConfig(lambda_sub=0.2))
    a4b = SubclassAuxiliaryLoss(
        SubclassAuxiliaryLossConfig(
            lambda_sub=0.2,
            use_hierarchy_loss=True,
            lambda_hier=0.0,
            hierarchy_parent_indices=[0, 1, 2],
        )
    )
    assert torch.allclose(a4b(outputs, batch)["loss"], a4a(outputs, batch)["loss"])


def test_use_hierarchy_false_does_not_compute_hierarchy_loss():
    info = SubclassAuxiliaryLoss(SubclassAuxiliaryLossConfig(lambda_sub=0.2))(_outputs(), _batch())
    assert set(info) == {"loss", "loss_super", "loss_sub"}


def test_use_hierarchy_true_without_subclass_logits_raises():
    criterion = SubclassAuxiliaryLoss(
        SubclassAuxiliaryLossConfig(use_hierarchy_loss=True, hierarchy_parent_indices=[0])
    )
    with pytest.raises(RuntimeError, match="logits_super and logits_sub"):
        criterion({"logits_super": torch.zeros(1, 5)}, {"y": torch.zeros(1, 5)})


def test_use_hierarchy_true_without_parent_mapping_raises():
    config = {
        "model": {"use_hierarchy_loss": True, "enable_subclass_auxiliary": True},
        "subclass_auxiliary": {"lambda_sub": 0.2},
        "hierarchy_loss": {"enabled": True, "lambda_hier": 0.1},
        "paths": {},
    }
    with pytest.raises(RuntimeError, match="subclass_vocab and subclass_parent_mapping"):
        build_subclass_criterion(config)


def test_training_epoch_records_hierarchy_diagnostics(tmp_path):
    paths = _write_artifacts(tmp_path)
    ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        subclass_index_csv=paths["subclass_index"],
        subclass_vocab_path=paths["subclass_vocab"],
        limit=2,
    )
    loader = DataLoader(ds, batch_size=2)
    model = ResNet1DAvailability(
        base_channels=4,
        layers=(1, 1, 1, 1),
        use_subclass_auxiliary=True,
        num_subclasses=paths["num_subclasses"],
    )
    criterion = SubclassAuxiliaryLoss(
        SubclassAuxiliaryLossConfig(
            lambda_sub=0.2,
            use_hierarchy_loss=True,
            lambda_hier=0.1,
            hierarchy_parent_indices=paths["parent_indices"],
        )
    )
    losses = run_subclass_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        criterion=criterion,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    )
    assert losses["loss"] > 0
    assert losses["loss_hier"] >= 0
    assert 0 <= losses["violation_rate"] <= 1
    assert losses["mean_violation_margin"] >= 0
    assert losses["max_violation_margin"] >= 0


def test_hierarchy_ablation_does_not_use_records500():
    assert not (ROOT / "records500").exists()


def _write_artifacts(tmp_path):
    artifacts = build_kept_subclass_artifacts(root=ROOT, day1_index=INDEX, min_train_pos=50)
    index_path = tmp_path / "subclass_index.csv"
    vocab_path = tmp_path / "subclass_vocab.json"
    mapping_path = tmp_path / "subclass_parent_mapping.json"
    artifacts["index"].to_csv(index_path, index=False)
    vocab_path.write_text(json.dumps(artifacts["vocab"]), encoding="utf-8")
    mapping_path.write_text(json.dumps(artifacts["parent_mapping"]), encoding="utf-8")
    return {
        "subclass_index": index_path,
        "subclass_vocab": vocab_path,
        "subclass_parent_mapping": mapping_path,
        "num_subclasses": int(artifacts["vocab"]["num_subclasses"]),
        "parent_indices": load_parent_indices(vocab_path=vocab_path, mapping_path=mapping_path),
    }
