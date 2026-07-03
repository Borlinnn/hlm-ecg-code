"""Paired full/masked PTB-XL views for confidence consistency training."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset


class PairedFullMaskedPTBXLDataset(Dataset):
    """Return full and masked views of the same records100 ECG sample."""

    def __init__(self, base_dataset: PTBXLDataset) -> None:
        if base_dataset.split != "train":
            raise ValueError("Paired full/masked views are only intended for train split")
        if base_dataset.fill_mode != "mean_fill":
            raise ValueError("A5 paired training uses A4a mean_fill masking semantics")
        if not base_dataset.subclass_columns:
            raise ValueError("A5 paired training requires subclass labels")
        self.base = base_dataset

    @property
    def root(self) -> Path:
        return self.base.root

    def __len__(self) -> int:
        return len(self.base)

    def set_random_seed(self, seed: int) -> None:
        self.base.set_random_seed(seed)

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        row = self.base.df.iloc[int(idx)]
        filename_lr = str(row["filename_lr"])
        raw, fields = self.base._read_raw(filename_lr)
        mask = self.base._mask_for_index(int(idx))
        full_mask = np.ones(12, dtype=np.float32)

        x_norm = (raw - self.base.mean.reshape(1, 12)) / self.base.std.reshape(1, 12)
        x_full = x_norm.T.astype(np.float32, copy=False)
        x_mask = x_norm.copy()
        x_mask[:, mask == 0] = 0.0
        x_mask = x_mask.T.astype(np.float32, copy=False)

        y = row[list(LABEL_ORDER)].to_numpy(dtype=np.float32)
        y_sub = row[self.base.subclass_columns].to_numpy(dtype=np.float32)
        return {
            "ecg_id": int(row["ecg_id"]),
            "x_full": torch.from_numpy(x_full),
            "availability_mask_full": torch.from_numpy(full_mask),
            "x_mask": torch.from_numpy(x_mask),
            "availability_mask_mask": torch.from_numpy(mask.astype(np.float32)),
            "x": torch.from_numpy(x_mask),
            "availability_mask": torch.from_numpy(mask.astype(np.float32)),
            "lead_mask": torch.from_numpy(mask.astype(np.float32)),
            "y": torch.from_numpy(y),
            "y_sub": torch.from_numpy(y_sub),
            "has_any_kept_subclass": torch.tensor(float(row["has_any_kept_subclass"]), dtype=torch.float32),
            "has_only_dropped_subclass": torch.tensor(float(row["has_only_dropped_subclass"]), dtype=torch.float32),
            "split": str(row["split"]),
            "filename_lr": filename_lr,
            "fs": int(fields.get("fs", -1)),
            "lead_names": list(fields.get("sig_name", [])),
        }
