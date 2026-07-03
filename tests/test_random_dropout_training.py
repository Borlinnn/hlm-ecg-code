from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.lead_dropout import RandomLeadDropout
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.evaluation.missing_patterns import required_patterns
from hlm_ecg.training.train_baseline import run_epoch


ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def test_random_dropout_mean_fill_sets_normalized_missing_leads_to_zero():
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
    mask = sample["lead_mask"].numpy()
    missing = np.where(mask == 0)[0]
    assert len(missing) == 1
    assert float(sample["x"][missing[0]].abs().sum()) == 0.0


def test_random_dropout_does_not_pass_mask_to_model():
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


def test_full_baseline_dataset_default_behavior_unchanged():
    ds = PTBXLDataset(root=ROOT, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=1)
    sample = ds[0]
    assert sample["lead_mask"].tolist() == [1.0] * 12


def test_evaluation_random_patterns_fixed_seed_reproducible():
    a = required_patterns(seed=20240604)["random-6"].mask_for_index(10)
    b = required_patterns(seed=20240604)["random-6"].mask_for_index(10)
    assert np.array_equal(a, b)


def test_records500_not_used_for_random_dropout(tmp_path):
    root = tmp_path / "ptb-xl"
    (root / "records500").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        PTBXLDataset(root=root, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=1)
