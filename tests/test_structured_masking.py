from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.lead_dropout import RandomLeadDropout
from hlm_ecg.data.lead_masking import StructuredLeadMasking
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.training.train_baseline import build_dataset, run_epoch

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def test_structured_masking_mean_fill_sets_normalized_missing_leads_to_zero():
    sampler = StructuredLeadMasking(
        random_prob=0.0,
        structured_prob=1.0,
        structured_patterns=("V1_V3_missing",),
        seed=7,
    )
    ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="mean_fill",
        lead_mask_sampler=sampler,
        limit=1,
    )
    sample = ds[0]
    mask = sample["lead_mask"].numpy()
    missing = np.where(mask == 0)[0]
    assert missing.tolist() == [6, 7, 8]
    for idx in missing:
        assert float(sample["x"][idx].abs().sum()) == 0.0


def test_structured_masking_does_not_pass_mask_to_model():
    class OnlyXModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(12 * 1000, 5)

        def forward(self, x):
            return self.linear(x.flatten(1))

    batch = {
        "x": torch.zeros(2, 12, 1000),
        "y": torch.zeros(2, 5),
        "lead_mask": torch.zeros(2, 12),
    }
    loader = DataLoader([batch], batch_size=None)
    model = OnlyXModel()
    loss = run_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        criterion=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    )
    assert loss > 0.0


def test_full_baseline_dataset_default_behavior_still_full_mask():
    config = {
        "paths": {
            "data_root": str(ROOT),
            "day1_index": str(INDEX),
            "norm_stats": str(NORM),
        },
        "smoke": {"train_limit": 1},
    }
    sample = build_dataset(config, "train", smoke_test=True)[0]
    assert sample["lead_mask"].tolist() == [1.0] * 12


def test_random_dropout_baseline_behavior_still_uses_random_sampler():
    config = {
        "seed": 42,
        "paths": {
            "data_root": str(ROOT),
            "day1_index": str(INDEX),
            "norm_stats": str(NORM),
        },
        "train_augmentation": {
            "enabled": True,
            "missing_counts": [3],
            "probabilities": [1.0],
            "fill_mode": "mean_fill",
            "min_available_leads": 1,
            "seed": 42,
        },
        "smoke": {"train_limit": 1},
    }
    sample = build_dataset(config, "train", smoke_test=True)[0]
    assert int((sample["lead_mask"].numpy() == 0).sum()) == 3


def test_structured_and_random_training_augmentation_cannot_both_be_enabled():
    config = {
        "seed": 42,
        "paths": {
            "data_root": str(ROOT),
            "day1_index": str(INDEX),
            "norm_stats": str(NORM),
        },
        "train_augmentation": {"enabled": True},
        "structured_masking": {"enabled": True},
        "smoke": {"train_limit": 1},
    }
    with pytest.raises(ValueError):
        build_dataset(config, "train", smoke_test=True)


def test_records500_not_used_for_structured_masking(tmp_path):
    root = tmp_path / "ptb-xl"
    (root / "records500").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        PTBXLDataset(root=root, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=1)


def test_legacy_random_lead_dropout_dataset_parameter_still_supported():
    sampler = RandomLeadDropout(missing_counts=(1,), probabilities=(1.0,), seed=7)
    ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="mean_fill",
        random_lead_dropout=sampler,
        limit=1,
    )
    sample = ds[0]
    assert int((sample["lead_mask"].numpy() == 0).sum()) == 1
