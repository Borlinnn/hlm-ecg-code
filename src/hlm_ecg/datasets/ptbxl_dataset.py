"""PTB-XL PyTorch Dataset for full-lead and missing-lead evaluation."""

from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hlm_ecg.data.lead_dropout import RandomLeadDropout
from hlm_ecg.data.lead_masking import LeadMaskSampler
from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.data.subclass_labels import load_subclass_vocab
from hlm_ecg.data.waveforms import (
    CANONICAL_LEADS,
    assert_expected_waveform,
    assert_no_records500,
    assert_records100_filename_lr,
)

FillMode = str
MaskProvider = Callable[[int], Sequence[int]]


class PTBXLDataset(Dataset):
    """Read PTB-XL records100 waveforms and fixed 5-superclass labels."""

    def __init__(
        self,
        *,
        root: Path | str,
        index_csv: Path | str,
        norm_stats_path: Path | str,
        split: str,
        fill_mode: FillMode = "full",
        lead_mask: Optional[Sequence[int]] = None,
        mask_provider: Optional[MaskProvider] = None,
        lead_mask_sampler: Optional[LeadMaskSampler] = None,
        random_lead_dropout: Optional[RandomLeadDropout] = None,
        subclass_index_csv: Path | str | None = None,
        subclass_vocab_path: Path | str | None = None,
        limit: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.index_csv = Path(index_csv)
        self.norm_stats_path = Path(norm_stats_path)
        self.split = split
        self.fill_mode = fill_mode
        self.lead_mask = None if lead_mask is None else self._validate_mask(lead_mask)
        self.mask_provider = mask_provider
        self.lead_mask_sampler = lead_mask_sampler
        self.random_lead_dropout = random_lead_dropout
        self.subclass_index_csv = None if subclass_index_csv is None else Path(subclass_index_csv)
        self.subclass_vocab_path = None if subclass_vocab_path is None else Path(subclass_vocab_path)
        self.subclass_columns: list[str] = []

        if fill_mode not in {"full", "mean_fill", "zero_fill"}:
            raise ValueError(f"Unsupported fill_mode: {fill_mode}")
        provided_masks = sum(
            x is not None
            for x in (self.lead_mask, self.mask_provider, self.lead_mask_sampler, self.random_lead_dropout)
        )
        if provided_masks > 1:
            raise ValueError("Use only one of lead_mask, mask_provider, lead_mask_sampler, or random_lead_dropout")
        if (self.random_lead_dropout is not None or self.lead_mask_sampler is not None) and fill_mode not in {
            "mean_fill",
            "zero_fill",
        }:
            raise ValueError("Train-time lead masking uses mean_fill or zero_fill semantics")
        assert_no_records500(self.root)

        df = pd.read_csv(self.index_csv)
        required = {"ecg_id", "split", "filename_lr", *LABEL_ORDER}
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(f"Day 1 index missing columns: {sorted(missing)}")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.subclass_index_csv is not None or self.subclass_vocab_path is not None:
            if self.subclass_index_csv is None or self.subclass_vocab_path is None:
                raise ValueError("Use subclass_index_csv and subclass_vocab_path together")
            vocab = load_subclass_vocab(self.subclass_vocab_path)
            self.subclass_columns = [str(x) for x in vocab.get("subclass_columns", [])]
            if not self.subclass_columns:
                self.subclass_columns = [f"y_sub_{x}" for x in vocab["subclasses"]]
            sub = pd.read_csv(self.subclass_index_csv)
            required_sub = {"ecg_id", "split", "filename_lr", "has_any_kept_subclass", "has_only_dropped_subclass", *self.subclass_columns}
            missing_sub = required_sub.difference(sub.columns)
            if missing_sub:
                raise RuntimeError(f"Subclass index missing columns: {sorted(missing_sub)}")
            self.df = self.df.merge(
                sub[["ecg_id", "has_any_kept_subclass", "has_only_dropped_subclass", *self.subclass_columns]],
                on="ecg_id",
                how="left",
                validate="one_to_one",
            )
            if self.df[self.subclass_columns].isna().any().any():
                raise RuntimeError("Subclass index merge left missing y_sub values")
        if limit is not None:
            self.df = self.df.iloc[: int(limit)].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No rows found for split={split}")

        stats = np.load(self.norm_stats_path)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)
        self.stats_leads = tuple(str(x) for x in stats["lead_names"].tolist())
        if self.stats_leads != CANONICAL_LEADS:
            raise RuntimeError(f"Normalization lead order mismatch: {self.stats_leads}")
        if self.mean.shape != (12,) or self.std.shape != (12,):
            raise RuntimeError("Normalization mean/std must have shape (12,)")
        if np.any(self.std <= 0):
            raise RuntimeError("Normalization std must be positive for every lead")

    def __len__(self) -> int:
        return int(len(self.df))

    @staticmethod
    def _validate_mask(mask: Sequence[int]) -> np.ndarray:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.shape != (12,):
            raise ValueError(f"lead mask must have shape (12,), got {arr.shape}")
        if not np.all(np.isin(arr, [0.0, 1.0])):
            raise ValueError("lead mask values must be 0 or 1")
        if float(arr.sum()) < 1.0:
            raise ValueError("at least one lead must be available")
        return arr

    def _mask_for_index(self, idx: int) -> np.ndarray:
        if self.fill_mode == "full":
            return np.ones(12, dtype=np.float32)
        if self.lead_mask_sampler is not None:
            return self._validate_mask(self.lead_mask_sampler.sample())
        if self.random_lead_dropout is not None:
            return self._validate_mask(self.random_lead_dropout.sample())
        if self.mask_provider is not None:
            return self._validate_mask(self.mask_provider(idx))
        if self.lead_mask is not None:
            return self.lead_mask.copy()
        return np.ones(12, dtype=np.float32)

    def set_random_seed(self, seed: int) -> None:
        if self.lead_mask_sampler is not None:
            self.lead_mask_sampler.set_seed(seed)
        if self.random_lead_dropout is not None:
            self.random_lead_dropout.set_seed(seed)

    def _read_raw(self, filename_lr: str) -> tuple[np.ndarray, Mapping[str, object]]:
        import wfdb  # type: ignore

        assert_records100_filename_lr(filename_lr)
        record = self.root / filename_lr
        sig, fields = wfdb.rdsamp(str(record))
        assert_expected_waveform(sig, fields, filename_lr)
        return sig.astype(np.float32, copy=False), fields

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        row = self.df.iloc[int(idx)]
        filename_lr = str(row["filename_lr"])
        raw, fields = self._read_raw(filename_lr)
        mask = self._mask_for_index(int(idx))

        if self.fill_mode == "zero_fill" and not np.all(mask == 1):
            raw = raw.copy()
            raw[:, mask == 0] = 0.0

        x = (raw - self.mean.reshape(1, 12)) / self.std.reshape(1, 12)

        if self.fill_mode == "mean_fill" and not np.all(mask == 1):
            x = x.copy()
            x[:, mask == 0] = 0.0

        x = x.T.astype(np.float32, copy=False)
        y = row[list(LABEL_ORDER)].to_numpy(dtype=np.float32)
        out = {
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(float(row["patient_id"])) if "patient_id" in row.index and not pd.isna(row["patient_id"]) else -1,
            "strat_fold": int(row["strat_fold"]) if "strat_fold" in row.index and not pd.isna(row["strat_fold"]) else -1,
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "lead_mask": torch.from_numpy(mask.astype(np.float32)),
            "availability_mask": torch.from_numpy(mask.astype(np.float32)),
            "split": str(row["split"]),
            "filename_lr": filename_lr,
            "fs": int(fields.get("fs", -1)),
            "lead_names": list(fields.get("sig_name", [])),
        }

        if self.subclass_columns:
            y_sub = row[self.subclass_columns].to_numpy(dtype=np.float32)
            out["y_sub"] = torch.from_numpy(y_sub)
            out["has_any_kept_subclass"] = torch.tensor(float(row["has_any_kept_subclass"]), dtype=torch.float32)
            out["has_only_dropped_subclass"] = torch.tensor(float(row["has_only_dropped_subclass"]), dtype=torch.float32)
        return out
