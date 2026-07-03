import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.subclass_labels import build_kept_subclass_artifacts
from hlm_ecg.datasets.paired_views import PairedFullMaskedPTBXLDataset
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.losses.confidence_consistency import ConfidenceConsistencyLossConfig, ConfidenceWeightedConsistencyLoss
from hlm_ecg.losses.hierarchy import ParentChildHierarchyLoss
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.training.train_confidence_consistency import run_consistency_epoch

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def test_paired_training_batch_contains_full_and_masked_views(tmp_path):
    paths = _write_artifacts(tmp_path)
    mask = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    base = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="mean_fill",
        lead_mask=mask,
        subclass_index_csv=paths["subclass_index"],
        subclass_vocab_path=paths["subclass_vocab"],
        limit=1,
    )
    sample = PairedFullMaskedPTBXLDataset(base)[0]
    assert tuple(sample["x_full"].shape) == (12, 1000)
    assert tuple(sample["x_mask"].shape) == (12, 1000)
    assert tuple(sample["availability_mask_full"].shape) == (12,)
    assert tuple(sample["availability_mask_mask"].shape) == (12,)
    assert torch.all(sample["availability_mask_full"] == 1)
    missing = sample["availability_mask_mask"] == 0
    assert torch.all(sample["x_mask"][missing] == 0)
    assert tuple(sample["y"].shape) == (5,)
    assert tuple(sample["y_sub"].shape) == (paths["num_subclasses"],)


def test_consistency_training_epoch_runs_without_hierarchy(tmp_path):
    paths = _write_artifacts(tmp_path)
    base = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="mean_fill",
        lead_mask=[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        subclass_index_csv=paths["subclass_index"],
        subclass_vocab_path=paths["subclass_vocab"],
        limit=2,
    )
    loader = DataLoader(PairedFullMaskedPTBXLDataset(base), batch_size=2)
    model = ResNet1DAvailability(
        base_channels=4,
        layers=(1, 1, 1, 1),
        use_subclass_auxiliary=True,
        num_subclasses=paths["num_subclasses"],
    )
    losses = run_consistency_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        consistency_loss=ConfidenceWeightedConsistencyLoss(
            ConfidenceConsistencyLossConfig(lambda_cons=0.1, gamma=1.0, enabled=True)
        ),
        lambda_sub=0.2,
        lambda_cons=0.1,
        ignore_only_dropped=True,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    )
    assert losses["loss"] > 0
    assert losses["cw_consistency_loss"] >= 0
    assert 0 <= losses["mean_consistency_weight"] <= 1
    assert 0 <= losses["min_consistency_weight"] <= 1
    assert 0 <= losses["max_consistency_weight"] <= 1
    assert "loss_hier" not in losses
    assert "violation_rate" not in losses


def test_consistency_training_epoch_can_include_hierarchy(tmp_path):
    paths = _write_artifacts(tmp_path)
    base = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="mean_fill",
        lead_mask=[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        subclass_index_csv=paths["subclass_index"],
        subclass_vocab_path=paths["subclass_vocab"],
        limit=2,
    )
    loader = DataLoader(PairedFullMaskedPTBXLDataset(base), batch_size=2)
    model = ResNet1DAvailability(
        base_channels=4,
        layers=(1, 1, 1, 1),
        use_subclass_auxiliary=True,
        num_subclasses=paths["num_subclasses"],
    )
    hierarchy = ParentChildHierarchyLoss([0] * paths["num_subclasses"])
    losses = run_consistency_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        consistency_loss=ConfidenceWeightedConsistencyLoss(
            ConfidenceConsistencyLossConfig(lambda_cons=0.1, gamma=1.0, enabled=True)
        ),
        lambda_sub=0.2,
        lambda_cons=0.1,
        hierarchy_loss=hierarchy,
        lambda_hier=0.1,
        ignore_only_dropped=True,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    )
    assert losses["loss"] > 0
    assert losses["loss_hier_full"] >= 0
    assert losses["loss_hier_mask"] >= 0
    assert 0 <= losses["hier_violation_rate_full"] <= 1
    assert 0 <= losses["hier_violation_rate_mask"] <= 1


def test_a5_training_does_not_use_records500():
    assert not (ROOT / "records500").exists()


def _write_artifacts(tmp_path):
    artifacts = build_kept_subclass_artifacts(root=ROOT, day1_index=INDEX, min_train_pos=50)
    index_path = tmp_path / "subclass_index.csv"
    vocab_path = tmp_path / "subclass_vocab.json"
    artifacts["index"].to_csv(index_path, index=False)
    vocab_path.write_text(json.dumps(artifacts["vocab"]), encoding="utf-8")
    return {
        "subclass_index": index_path,
        "subclass_vocab": vocab_path,
        "num_subclasses": int(artifacts["vocab"]["num_subclasses"]),
    }
