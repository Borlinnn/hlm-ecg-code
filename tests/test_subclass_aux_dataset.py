from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.subclass_labels import build_kept_subclass_artifacts
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def test_subclass_vocab_not_hardcoded_and_min_train_pos_filters(tmp_path):
    artifacts = build_kept_subclass_artifacts(root=ROOT, day1_index=INDEX, min_train_pos=50)
    vocab = artifacts["vocab"]
    counts = artifacts["counts"]
    assert int(vocab["num_subclasses"]) == len(vocab["subclasses"])
    assert int(vocab["num_subclasses"]) > 0
    assert int(vocab["num_subclasses"]) < 23
    assert counts["train_positive_count"].min() >= 50


def test_dataset_returns_y_sub_and_superclass_shapes(tmp_path):
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
    sample = ds[0]
    assert tuple(sample["x"].shape) == (12, 1000)
    assert tuple(sample["y"].shape) == (5,)
    assert tuple(sample["y_sub"].shape) == (paths["num_subclasses"],)
    assert tuple(sample["availability_mask"].shape) == (12,)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert tuple(batch["y_sub"].shape) == (2, paths["num_subclasses"])
    assert tuple(batch["availability_mask"].shape) == (2, 12)


def test_has_only_dropped_subclass_flag_is_read(tmp_path):
    paths = _write_artifacts(tmp_path)
    index = pd.read_csv(paths["subclass_index"])
    index.loc[index.index[0], "has_only_dropped_subclass"] = 1
    index.to_csv(paths["subclass_index"], index=False)
    ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        subclass_index_csv=paths["subclass_index"],
        subclass_vocab_path=paths["subclass_vocab"],
        limit=1,
    )
    assert float(ds[0]["has_only_dropped_subclass"]) == 1.0


def _write_artifacts(tmp_path):
    artifacts = build_kept_subclass_artifacts(root=ROOT, day1_index=INDEX, min_train_pos=50)
    index_path = tmp_path / "subclass_index.csv"
    vocab_path = tmp_path / "subclass_vocab.json"
    artifacts["index"].to_csv(index_path, index=False)
    vocab_path.write_text(__import__("json").dumps(artifacts["vocab"]), encoding="utf-8")
    return {
        "subclass_index": index_path,
        "subclass_vocab": vocab_path,
        "num_subclasses": int(artifacts["vocab"]["num_subclasses"]),
    }
