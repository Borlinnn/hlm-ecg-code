"""Safe metric computations for bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.metrics import bce_with_logits_np, sigmoid


def macro_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    per_label = np.mean((probs - targets) ** 2, axis=0)
    return float(np.mean(per_label))


def _safe_threshold_free_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, bool]:
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan"), False
    return float(metric_fn(y_true, y_score)), True


def compute_bootstrap_metrics(
    *,
    logits: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
    min_valid_labels: int = 3,
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if probs is None:
        probs = sigmoid(logits)
    else:
        probs = np.asarray(probs, dtype=np.float64)
    if logits.shape != targets.shape or logits.shape != preds.shape or logits.shape != probs.shape:
        raise ValueError(f"Shape mismatch: logits={logits.shape}, targets={targets.shape}, preds={preds.shape}, probs={probs.shape}")
    if logits.shape[1] != len(label_order):
        raise ValueError(f"Expected {len(label_order)} labels, got {logits.shape[1]}")

    per_class_auroc: dict[str, float] = {}
    per_class_auprc: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    warnings: list[str] = []
    valid_auroc = 0
    valid_auprc = 0
    for idx, label in enumerate(label_order):
        auroc, ok = _safe_threshold_free_metric(roc_auc_score, targets[:, idx], probs[:, idx])
        per_class_auroc[label] = auroc
        valid_auroc += int(ok)
        if not ok:
            warnings.append(f"AUROC undefined for {label}: y_true has one class")
        auprc, ok = _safe_threshold_free_metric(average_precision_score, targets[:, idx], probs[:, idx])
        per_class_auprc[label] = auprc
        valid_auprc += int(ok)
        if not ok:
            warnings.append(f"AUPRC undefined for {label}: y_true has one class")
        per_class_f1[label] = float(f1_score(targets[:, idx], preds[:, idx], zero_division=0))

    auroc_values = np.asarray([v for v in per_class_auroc.values() if not np.isnan(v)], dtype=np.float64)
    auprc_values = np.asarray([v for v in per_class_auprc.values() if not np.isnan(v)], dtype=np.float64)
    invalid_macro_auroc = valid_auroc < min_valid_labels
    invalid_macro_auprc = valid_auprc < min_valid_labels
    return {
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values.size and not invalid_macro_auroc else float("nan"),
        "macro_auprc": float(np.mean(auprc_values)) if auprc_values.size and not invalid_macro_auprc else float("nan"),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "bce_nll": bce_with_logits_np(logits, targets),
        "macro_brier": macro_brier_score(probs, targets),
        "per_class_auroc": per_class_auroc,
        "per_class_auprc": per_class_auprc,
        "per_class_f1": per_class_f1,
        "n_valid_auroc_labels": int(valid_auroc),
        "n_valid_auprc_labels": int(valid_auprc),
        "invalid_macro_auroc": bool(invalid_macro_auroc),
        "invalid_macro_auprc": bool(invalid_macro_auprc),
        "warnings": warnings,
    }


def flatten_metric(metric_result: Mapping[str, Any], metric: str) -> float:
    if metric.startswith("per_class_"):
        _, _, label_metric = metric.partition("per_class_")
        label, _, name = label_metric.partition("_")
        key = f"per_class_{name}"
        return float(metric_result[key][label])
    return float(metric_result[metric])

