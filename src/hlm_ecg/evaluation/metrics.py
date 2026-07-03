"""Multi-label ECG metrics and validation threshold tuning."""

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def bce_with_logits_np(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(loss.mean())


def _safe_binary_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray, label: str, name: str):
    unique = np.unique(y_true)
    if unique.size < 2:
        return None, f"{name} undefined for {label}: only class {int(unique[0]) if unique.size else 'empty'} present"
    return float(metric_fn(y_true, y_score)), None


def compute_multilabel_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
) -> Dict[str, object]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.shape != targets.shape or logits.shape[1] != len(label_order):
        raise ValueError(f"Expected logits/targets shape (N,{len(label_order)}), got {logits.shape}/{targets.shape}")
    probs = sigmoid(logits)
    if thresholds is None:
        threshold_arr = np.full(len(label_order), 0.5, dtype=np.float64)
    else:
        threshold_arr = np.asarray(thresholds, dtype=np.float64)
        if threshold_arr.shape != (len(label_order),):
            raise ValueError(f"thresholds must have shape ({len(label_order)},)")
    preds = (probs >= threshold_arr.reshape(1, -1)).astype(np.int64)

    warnings = []
    per_class_auroc: Dict[str, float | None] = {}
    per_class_auprc: Dict[str, float | None] = {}
    for idx, label in enumerate(label_order):
        auroc, warning = _safe_binary_metric(roc_auc_score, targets[:, idx], probs[:, idx], label, "AUROC")
        if warning:
            warnings.append(warning)
        auprc, warning = _safe_binary_metric(average_precision_score, targets[:, idx], probs[:, idx], label, "AUPRC")
        if warning:
            warnings.append(warning)
        per_class_auroc[label] = auroc
        per_class_auprc[label] = auprc

    valid_aurocs = [v for v in per_class_auroc.values() if v is not None]
    valid_auprcs = [v for v in per_class_auprc.values() if v is not None]
    per_class_f1 = {
        label: float(f1_score(targets[:, idx], preds[:, idx], zero_division=0))
        for idx, label in enumerate(label_order)
    }
    return {
        "macro_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
        "macro_auprc": float(np.mean(valid_auprcs)) if valid_auprcs else None,
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "per_class_auroc": per_class_auroc,
        "per_class_auprc": per_class_auprc,
        "per_class_f1": per_class_f1,
        "bce_nll": bce_with_logits_np(logits, targets),
        "thresholds": {label: float(threshold_arr[idx]) for idx, label in enumerate(label_order)},
        "warnings": warnings,
    }


def tune_thresholds_on_validation(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    *,
    label_order: Sequence[str] = LABEL_ORDER,
    grid: Iterable[float] | None = None,
) -> Dict[str, object]:
    logits = np.asarray(val_logits, dtype=np.float64)
    targets = np.asarray(val_targets, dtype=np.int64)
    probs = sigmoid(logits)
    if grid is None:
        grid_values = np.linspace(0.05, 0.95, 19)
    else:
        grid_values = np.asarray(list(grid), dtype=np.float64)
    thresholds = []
    best_f1 = {}
    for idx, label in enumerate(label_order):
        scores = []
        for threshold in grid_values:
            pred = (probs[:, idx] >= threshold).astype(np.int64)
            scores.append(float(f1_score(targets[:, idx], pred, zero_division=0)))
        best_idx = int(np.argmax(scores))
        thresholds.append(float(grid_values[best_idx]))
        best_f1[label] = float(scores[best_idx])
    return {
        "thresholds": {label: float(thresholds[idx]) for idx, label in enumerate(label_order)},
        "threshold_array": thresholds,
        "per_class_val_f1": best_f1,
        "source_split": "val",
    }
