"""Calibration metrics for multi-label ECG probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.metrics import sigmoid


@dataclass(frozen=True)
class BinaryCalibrationResult:
    ece: float
    mce: float
    brier: float
    nll: float
    bins: list[dict[str, Any]]


def binary_nll(probs: np.ndarray, targets: np.ndarray, *, eps: float = 1e-12) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    targets = np.asarray(targets, dtype=np.float64)
    loss = -(targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs))
    return float(np.mean(loss))


def binary_brier(probs: np.ndarray, targets: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean((probs - targets) ** 2))


def binary_calibration_bins(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 15,
) -> list[dict[str, Any]]:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probs.shape != targets.shape:
        raise ValueError(f"probs/targets shape mismatch: {probs.shape}/{targets.shape}")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    for bin_id in range(n_bins):
        low = float(edges[bin_id])
        high = float(edges[bin_id + 1])
        if bin_id == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        n_bin = int(mask.sum())
        if n_bin:
            mean_conf = float(probs[mask].mean())
            empirical = float(targets[mask].mean())
            gap = abs(mean_conf - empirical)
        else:
            mean_conf = float("nan")
            empirical = float("nan")
            gap = float("nan")
        bins.append(
            {
                "bin_id": bin_id,
                "bin_low": low,
                "bin_high": high,
                "n_bin": n_bin,
                "mean_confidence": mean_conf,
                "empirical_frequency": empirical,
                "abs_gap": gap,
            }
        )
    return bins


def binary_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 15,
) -> tuple[float, float, list[dict[str, Any]]]:
    probs = np.asarray(probs, dtype=np.float64)
    bins = binary_calibration_bins(probs, targets, n_bins=n_bins)
    n_total = int(probs.shape[0])
    weighted = []
    gaps = []
    for row in bins:
        if row["n_bin"] > 0:
            weighted.append((row["n_bin"] / n_total) * row["abs_gap"])
            gaps.append(row["abs_gap"])
    return float(np.sum(weighted)), float(np.max(gaps) if gaps else 0.0), bins


def compute_calibration_metrics(
    *,
    targets: np.ndarray,
    logits: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    n_bins: int = 15,
    label_order: Sequence[str] = LABEL_ORDER,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    if probs is None:
        if logits is None:
            raise ValueError("Either logits or probs must be provided")
        probs = sigmoid(np.asarray(logits, dtype=np.float64))
    else:
        probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != targets.shape:
        raise ValueError(f"probs/targets shape mismatch: {probs.shape}/{targets.shape}")
    if probs.shape[1] != len(label_order):
        raise ValueError(f"Expected {len(label_order)} labels, got {probs.shape[1]}")

    per_class_ece: dict[str, float] = {}
    per_class_mce: dict[str, float] = {}
    per_class_brier: dict[str, float] = {}
    per_class_nll: dict[str, float] = {}
    reliability_rows: list[dict[str, Any]] = []
    for idx, label in enumerate(label_order):
        ece, mce, bins = binary_ece(probs[:, idx], targets[:, idx], n_bins=n_bins)
        per_class_ece[label] = ece
        per_class_mce[label] = mce
        per_class_brier[label] = binary_brier(probs[:, idx], targets[:, idx])
        per_class_nll[label] = binary_nll(probs[:, idx], targets[:, idx])
        for bin_row in bins:
            reliability_rows.append({"class": label, **bin_row})

    return {
        "macro_ece": float(np.mean([per_class_ece[label] for label in label_order])),
        "macro_mce": float(np.mean([per_class_mce[label] for label in label_order])),
        "macro_brier": float(np.mean([per_class_brier[label] for label in label_order])),
        "macro_nll": float(np.mean([per_class_nll[label] for label in label_order])),
        "sample_label_bce": binary_nll(probs.reshape(-1), targets.reshape(-1)),
        "per_class_ece": per_class_ece,
        "per_class_mce": per_class_mce,
        "per_class_brier": per_class_brier,
        "per_class_nll": per_class_nll,
        "reliability_rows": reliability_rows,
        "n_samples": int(targets.shape[0]),
        "n_bins": int(n_bins),
    }


def aggregate_calibration_metrics(metric_dicts: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not metric_dicts:
        raise ValueError("metric_dicts cannot be empty")
    keys = ("macro_ece", "macro_brier", "macro_nll", "sample_label_bce")
    return {key: float(np.mean([float(row[key]) for row in metric_dicts])) for key in keys}

