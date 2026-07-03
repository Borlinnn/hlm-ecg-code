from pathlib import Path

import numpy as np
import pytest

from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset


ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


class FixedSampler:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=np.float32)

    def sample(self):
        return self.mask.copy()

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)


def test_dataset_returns_expected_shapes():
    ds = PTBXLDataset(root=ROOT, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=1)
    sample = ds[0]
    assert tuple(sample["x"].shape) == (12, 1000)
    assert tuple(sample["y"].shape) == (5,)
    assert tuple(sample["lead_mask"].shape) == (12,)
    assert tuple(sample["availability_mask"].shape) == (12,)
    assert sample["filename_lr"].startswith("records100/")


def test_dataset_disallows_records500_root(tmp_path):
    root = tmp_path / "ptb-xl"
    (root / "records500").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        PTBXLDataset(root=root, index_csv=INDEX, norm_stats_path=NORM, split="train", limit=1)


def test_mean_fill_and_zero_fill_have_different_semantics():
    mask = np.ones(12, dtype=np.float32)
    mask[0] = 0.0
    mean_ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="mean_fill",
        lead_mask=mask,
        limit=1,
    )
    zero_ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="zero_fill",
        lead_mask=mask,
        limit=1,
    )
    x_mean = mean_ds[0]["x"]
    x_zero = zero_ds[0]["x"]
    assert float(x_mean[0].abs().sum()) == 0.0
    assert float(x_zero[0].abs().sum()) > 0.0
    assert not np.allclose(x_mean.numpy(), x_zero.numpy())


def test_train_time_sampler_allows_zero_fill_for_sensitivity():
    mask = np.ones(12, dtype=np.float32)
    mask[0] = 0.0
    ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        fill_mode="zero_fill",
        lead_mask_sampler=FixedSampler(mask),
        limit=1,
    )
    sample = ds[0]
    assert tuple(sample["x"].shape) == (12, 1000)
    assert float(sample["lead_mask"][0]) == 0.0
