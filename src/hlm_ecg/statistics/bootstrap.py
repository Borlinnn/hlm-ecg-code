"""Patient-level paired bootstrap utilities for HLM-ECG predictions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.prediction_artifacts import safe_pattern_name
from hlm_ecg.statistics.bootstrap_metrics import compute_bootstrap_metrics

METHODS = (
    "A0_full_no_masking",
    "A1_random_dropout",
    "A2_structured_masking",
    "A4a_subclass_auxiliary",
    "A5_lite_confidence_consistency_0p05",
)

PATTERNS = (
    "full",
    "random-1",
    "random-3",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)

REPORT_PATTERNS = (
    "full",
    "random-1",
    "random-3",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
    "hard_structured_avg",
    "hard_overall_avg",
    "avg_all_missing",
)

HARD_STRUCTURED_PATTERNS = (
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
HARD_OVERALL_PATTERNS = ("random-6", *HARD_STRUCTURED_PATTERNS)
ALL_MISSING_PATTERNS = PATTERNS[1:]
AGGREGATES = {
    "hard_structured_avg": HARD_STRUCTURED_PATTERNS,
    "hard_overall_avg": HARD_OVERALL_PATTERNS,
    "avg_all_missing": ALL_MISSING_PATTERNS,
}

PRIMARY_COMPARISONS = (
    ("A4a_vs_A1", "A4a_subclass_auxiliary", "A1_random_dropout"),
    ("A4a_vs_A2", "A4a_subclass_auxiliary", "A2_structured_masking"),
    ("A5_lite_vs_A4a", "A5_lite_confidence_consistency_0p05", "A4a_subclass_auxiliary"),
    ("A4a_vs_A0", "A4a_subclass_auxiliary", "A0_full_no_masking"),
)
SECONDARY_COMPARISONS = (
    ("A5_lite_vs_A1", "A5_lite_confidence_consistency_0p05", "A1_random_dropout"),
    ("A2_vs_A1", "A2_structured_masking", "A1_random_dropout"),
)
COMPARISONS = (*PRIMARY_COMPARISONS, *SECONDARY_COMPARISONS)


@dataclass
class PredictionData:
    method_id: str
    pattern: str
    split: str
    fill_mode: str
    ecg_ids: np.ndarray
    patient_ids: np.ndarray
    targets: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    threshold_source_split: str

    def subset(self, indices: np.ndarray) -> "PredictionData":
        return PredictionData(
            method_id=self.method_id,
            pattern=self.pattern,
            split=self.split,
            fill_mode=self.fill_mode,
            ecg_ids=self.ecg_ids[indices],
            patient_ids=self.patient_ids[indices],
            targets=self.targets[indices],
            logits=self.logits[indices],
            probs=self.probs[indices],
            preds=self.preds[indices],
            threshold_source_split=self.threshold_source_split,
        )


def prediction_csv_path(predictions_dir: Path, method_id: str, fill_mode: str, split: str, pattern: str) -> Path:
    return predictions_dir / method_id / fill_mode / split / f"{safe_pattern_name(pattern)}.csv"


def load_prediction_csv(path: Path, *, method_id: str, pattern: str, split: str, fill_mode: str) -> PredictionData:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Empty prediction CSV: {path}")
    threshold_sources = {row["threshold_source_split"] for row in rows}
    if threshold_sources != {"val"}:
        raise RuntimeError(f"{path} threshold_source_split must be val, got {threshold_sources}")
    if {row["split"] for row in rows} != {split}:
        raise RuntimeError(f"{path} split mismatch")
    if {row["fill_mode"] for row in rows} != {fill_mode}:
        raise RuntimeError(f"{path} fill_mode mismatch")
    if {row["method_id"] for row in rows} != {method_id}:
        raise RuntimeError(f"{path} method_id mismatch")
    ecg_ids = np.asarray([int(row["ecg_id"]) for row in rows], dtype=np.int64)
    patient_ids = np.asarray([int(row["patient_id"]) for row in rows], dtype=np.int64)
    if np.any(patient_ids < 0):
        raise RuntimeError(f"{path} missing patient_id; patient-level bootstrap cannot proceed")
    targets = np.asarray([[int(row[f"y_true_{label}"]) for label in LABEL_ORDER] for row in rows], dtype=np.int64)
    logits = np.asarray([[float(row[f"logit_{label}"]) for label in LABEL_ORDER] for row in rows], dtype=np.float64)
    probs = np.asarray([[float(row[f"prob_{label}"]) for label in LABEL_ORDER] for row in rows], dtype=np.float64)
    preds = np.asarray([[int(row[f"pred_{label}"]) for label in LABEL_ORDER] for row in rows], dtype=np.int64)
    return PredictionData(
        method_id=method_id,
        pattern=pattern,
        split=split,
        fill_mode=fill_mode,
        ecg_ids=ecg_ids,
        patient_ids=patient_ids,
        targets=targets,
        logits=logits,
        probs=probs,
        preds=preds,
        threshold_source_split="val",
    )


def load_prediction_data(
    predictions_dir: Path,
    *,
    methods: Sequence[str] = METHODS,
    patterns: Sequence[str] = PATTERNS,
    split: str = "test",
    fill_mode: str = "mean_fill",
) -> dict[str, dict[str, PredictionData]]:
    out: dict[str, dict[str, PredictionData]] = {}
    for method_id in methods:
        out[method_id] = {}
        for pattern in patterns:
            path = prediction_csv_path(predictions_dir, method_id, fill_mode, split, pattern)
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction CSV: {path}")
            out[method_id][pattern] = load_prediction_csv(
                path,
                method_id=method_id,
                pattern=pattern,
                split=split,
                fill_mode=fill_mode,
            )
    return out


def patient_groups(patient_ids: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for idx, patient_id in enumerate(patient_ids.tolist()):
        groups.setdefault(int(patient_id), []).append(idx)
    return {patient_id: np.asarray(indices, dtype=np.int64) for patient_id, indices in groups.items()}


def generate_patient_bootstrap_samples(patient_ids: np.ndarray, *, n_bootstrap: int, seed: int) -> list[np.ndarray]:
    unique_patients = np.asarray(sorted(set(int(x) for x in patient_ids.tolist())), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return [rng.choice(unique_patients, size=unique_patients.size, replace=True) for _ in range(int(n_bootstrap))]


def sampled_indices_from_patients(groups: Mapping[int, np.ndarray], sampled_patients: np.ndarray) -> np.ndarray:
    parts = [groups[int(patient_id)] for patient_id in sampled_patients]
    return np.concatenate(parts, axis=0)


def compute_metric_bundle(data: PredictionData) -> dict[str, Any]:
    return compute_bootstrap_metrics(logits=data.logits, targets=data.targets, preds=data.preds, probs=data.probs)


def metric_value(bundle: Mapping[str, Any], metric: str) -> float:
    if metric.startswith("per_class_"):
        _, _, tail = metric.partition("per_class_")
        label, _, metric_name = tail.partition("_")
        return float(bundle[f"per_class_{metric_name}"][label])
    return float(bundle[metric])


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan"), float("nan"), int(arr.size)
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)), int(arr.size - valid.size)


def summarize_distribution(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    ci_low, ci_high, invalid = percentile_ci(arr)
    return {
        "mean": float(np.mean(valid)) if valid.size else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap_valid": int(valid.size),
        "invalid_replicates": int(invalid),
    }


def paired_delta_summary(values: np.ndarray, observed_delta: float) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    ci_low, ci_high, invalid = percentile_ci(arr)
    if valid.size:
        prob_gt_0 = float(np.mean(valid > 0.0))
        p_two_sided = float(min(1.0, 2.0 * min(np.mean(valid <= 0.0), np.mean(valid >= 0.0))))
        mean = float(np.mean(valid))
    else:
        prob_gt_0 = float("nan")
        p_two_sided = float("nan")
        mean = float("nan")
    return {
        "observed_delta": float(observed_delta),
        "bootstrap_mean_delta": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "probability_delta_gt_0": prob_gt_0,
        "p_two_sided": p_two_sided,
        "n_bootstrap_valid": int(valid.size),
        "invalid_replicates": int(invalid),
    }
