"""Prediction artifact helpers for evaluation-only HLM-ECG reruns."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.metrics import sigmoid

PREDICTION_REQUIRED_COLUMNS = (
    "ecg_id",
    "patient_id",
    "split",
    "strat_fold",
    "method_id",
    "pattern",
    "fill_mode",
    "random_seed",
    "threshold_source_split",
    *(f"availability_mask_{idx}" for idx in range(12)),
    *(f"y_true_{label}" for label in LABEL_ORDER),
    *(f"logit_{label}" for label in LABEL_ORDER),
    *(f"prob_{label}" for label in LABEL_ORDER),
    *(f"pred_{label}" for label in LABEL_ORDER),
    *(f"threshold_{label}" for label in LABEL_ORDER),
)


def safe_pattern_name(pattern: str) -> str:
    replacements = {
        "limb-only / precordial-missing": "limb_only_precordial_missing",
        "precordial-only / limb-missing": "precordial_only_limb_missing",
        "V1-V3 missing": "V1_V3_missing",
        "V4-V6 missing": "V4_V6_missing",
    }
    if pattern in replacements:
        return replacements[pattern]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", pattern).strip("_")
    return safe or "pattern"


def build_prediction_output_path(
    predictions_dir: Path | str,
    *,
    method_id: str,
    fill_mode: str,
    split: str,
    pattern: str,
) -> Path:
    return Path(predictions_dir) / method_id / fill_mode / split / f"{safe_pattern_name(pattern)}.csv"


def load_thresholds_val(output_dir: Path | str) -> tuple[np.ndarray, dict[str, float], str]:
    path = Path(output_dir) / "thresholds_val.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    source = str(data.get("source_split", "unknown"))
    if source != "val":
        raise RuntimeError(f"Expected thresholds_val.json source_split=val, got {source}")
    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError(f"Missing thresholds dict in {path}")
    threshold_map = {label: float(thresholds[label]) for label in LABEL_ORDER}
    return np.asarray([threshold_map[label] for label in LABEL_ORDER], dtype=np.float64), threshold_map, source


def _tensor_or_list_to_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    forward_fn: Callable[[torch.nn.Module, Mapping[str, object]], torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    logits_all = []
    targets_all = []
    ecg_ids = []
    patient_ids = []
    strat_folds = []
    availability_masks = []
    splits: list[str] = []
    for batch in loader:
        logits = forward_fn(model, batch).detach().cpu().numpy()
        logits_all.append(logits)
        targets_all.append(batch["y"].detach().cpu().numpy())
        ecg_ids.append(_tensor_or_list_to_numpy(batch["ecg_id"], dtype=np.int64))
        patient_ids.append(_tensor_or_list_to_numpy(batch.get("patient_id", np.full(logits.shape[0], -1)), dtype=np.int64))
        strat_folds.append(_tensor_or_list_to_numpy(batch.get("strat_fold", np.full(logits.shape[0], -1)), dtype=np.int64))
        availability_masks.append(_tensor_or_list_to_numpy(batch["availability_mask"], dtype=np.float32))
        split_value = batch.get("split", "")
        if isinstance(split_value, (list, tuple)):
            splits.extend(str(x) for x in split_value)
        else:
            splits.extend([str(split_value)] * int(logits.shape[0]))
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "targets": np.concatenate(targets_all, axis=0),
        "ecg_ids": np.concatenate(ecg_ids, axis=0),
        "patient_ids": np.concatenate(patient_ids, axis=0),
        "strat_folds": np.concatenate(strat_folds, axis=0),
        "availability_masks": np.concatenate(availability_masks, axis=0),
        "splits": splits,
    }


def prediction_rows(
    *,
    method_id: str,
    pattern: str,
    fill_mode: str,
    split: str,
    random_seed: int,
    threshold_source_split: str,
    thresholds: Sequence[float],
    collected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    logits = np.asarray(collected["logits"], dtype=np.float64)
    targets = np.asarray(collected["targets"], dtype=np.int64)
    probs = sigmoid(logits)
    threshold_arr = np.asarray(thresholds, dtype=np.float64)
    preds = (probs >= threshold_arr.reshape(1, -1)).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for idx in range(logits.shape[0]):
        row: dict[str, Any] = {
            "ecg_id": int(np.asarray(collected["ecg_ids"])[idx]),
            "patient_id": int(np.asarray(collected["patient_ids"])[idx]),
            "split": split,
            "strat_fold": int(np.asarray(collected["strat_folds"])[idx]),
            "method_id": method_id,
            "pattern": pattern,
            "fill_mode": fill_mode,
            "random_seed": int(random_seed),
            "threshold_source_split": threshold_source_split,
        }
        mask = np.asarray(collected["availability_masks"])[idx]
        for lead_idx in range(12):
            row[f"availability_mask_{lead_idx}"] = int(mask[lead_idx])
        for label_idx, label in enumerate(LABEL_ORDER):
            row[f"y_true_{label}"] = int(targets[idx, label_idx])
            row[f"logit_{label}"] = float(logits[idx, label_idx])
            row[f"prob_{label}"] = float(probs[idx, label_idx])
            row[f"pred_{label}"] = int(preds[idx, label_idx])
            row[f"threshold_{label}"] = float(threshold_arr[label_idx])
        rows.append(row)
    return rows


def save_predictions_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PREDICTION_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return prediction_file_info(path)


def validate_prediction_csv_schema(path: Path | str) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    return [column for column in PREDICTION_REQUIRED_COLUMNS if column not in columns]


def count_csv_rows(path: Path | str) -> int:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_file_info(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    return {
        "csv_path": str(path),
        "n_rows": count_csv_rows(path),
        "file_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_split_row_count(path: Path | str, expected_rows: int) -> bool:
    return count_csv_rows(path) == int(expected_rows)
