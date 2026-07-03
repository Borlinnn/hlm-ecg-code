from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.evaluation.evaluate_patterns import build_pattern_dataset
from hlm_ecg.evaluation.missing_patterns import required_patterns
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.training.train_baseline import build_dataset, predict_logits, run_epoch

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def _base_config():
    return {
        "seed": 42,
        "paths": {
            "data_root": str(ROOT),
            "day1_index": str(INDEX),
            "norm_stats": str(NORM),
        },
        "structured_masking": {
            "enabled": True,
            "fill_mode": "mean_fill",
            "random_missing_counts": [0, 1, 3, 6],
            "random_prob": 0.0,
            "structured_prob": 1.0,
            "structured_patterns": ["V1_V3_missing"],
            "min_available_leads": 1,
            "seed": 42,
        },
        "smoke": {"train_limit": 2, "val_limit": 2, "test_limit": 2},
    }


def test_dataset_returns_availability_mask_shape():
    ds = PTBXLDataset(root=ROOT, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=2)
    sample = ds[0]
    assert tuple(sample["availability_mask"].shape) == (12,)
    assert torch.equal(sample["availability_mask"], sample["lead_mask"])


def test_batch_availability_mask_shape():
    ds = PTBXLDataset(root=ROOT, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=2)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert tuple(batch["availability_mask"].shape) == (2, 12)


def test_full_pattern_availability_mask_is_all_one():
    ds = PTBXLDataset(root=ROOT, index_csv=INDEX, norm_stats_path=NORM, split="train", fill_mode="full", limit=1)
    sample = ds[0]
    assert sample["availability_mask"].tolist() == [1.0] * 12


def test_missing_pattern_mask_matches_zeroed_leads():
    pattern = required_patterns()["V1-V3 missing"]
    ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="test",
        fill_mode="mean_fill",
        mask_provider=pattern.mask_for_index,
        limit=1,
    )
    sample = ds[0]
    mask = sample["availability_mask"].numpy()
    missing = np.where(mask == 0)[0]
    assert missing.tolist() == [6, 7, 8]
    for idx in missing:
        assert float(sample["x"][idx].abs().sum()) == 0.0


def test_train_time_structured_mask_is_available_to_model():
    config = _base_config()
    ds = build_dataset(config, "train", smoke_test=True)
    sample = ds[0]
    assert sample["availability_mask"].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]


def test_run_epoch_passes_availability_mask_to_model():
    model = ResNet1DAvailability(base_channels=4, layers=(1, 1, 1, 1))
    batch = {
        "x": torch.zeros(2, 12, 1000),
        "y": torch.zeros(2, 5),
        "availability_mask": torch.ones(2, 12),
    }
    loader = DataLoader([batch], batch_size=None)
    loss = run_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        criterion=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    )
    assert loss > 0.0


def test_predict_logits_passes_availability_mask_to_model():
    model = ResNet1DAvailability(base_channels=4, layers=(1, 1, 1, 1))
    batch = {
        "x": torch.zeros(2, 12, 1000),
        "y": torch.zeros(2, 5),
        "availability_mask": torch.ones(2, 12),
    }
    logits, targets = predict_logits(model, DataLoader([batch], batch_size=None), device=torch.device("cpu"))
    assert tuple(logits.shape) == (2, 5)
    assert tuple(targets.shape) == (2, 5)


def test_evaluation_fixed_random_pattern_mask_is_reproducible():
    config = _base_config()
    pattern = required_patterns(seed=20240604)["random-3"]
    ds_a = build_pattern_dataset(config, pattern, fill_mode="mean_fill", smoke_test=True)
    ds_b = build_pattern_dataset(config, pattern, fill_mode="mean_fill", smoke_test=True)
    assert torch.equal(ds_a[1]["availability_mask"], ds_b[1]["availability_mask"])


def test_records500_not_used_for_availability_embedding(tmp_path):
    root = tmp_path / "ptb-xl"
    (root / "records500").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        PTBXLDataset(root=root, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=1)
