#!/usr/bin/env python3
"""Build final reviewer-defense tables, figures, and claim audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hlm_ecg.calibration.calibration_metrics import compute_calibration_metrics
from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.statistics.bootstrap_metrics import compute_bootstrap_metrics
from scripts.generate_reviewer_defense_configs import build_experiment_plan

PATTERN_FILES = {
    "full": "full",
    "random-1": "random_1",
    "random-3": "random_3",
    "random-6": "random_6",
    "limb-only / precordial-missing": "limb_only_precordial_missing",
    "precordial-only / limb-missing": "precordial_only_limb_missing",
    "V1-V3 missing": "V1_V3_missing",
    "V4-V6 missing": "V4_V6_missing",
}
PATTERN_ORDER = tuple(PATTERN_FILES)
HARD_STRUCTURED = (
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
HARD_OVERALL = ("random-6", *HARD_STRUCTURED)
AGGREGATES = {
    "hard_structured_avg": HARD_STRUCTURED,
    "hard_overall_avg": HARD_OVERALL,
}
PRIMARY_METHODS = (
    "M0_full_no_masking",
    "M1_random_dropout",
    "M2_structured_masking",
    "M3_random_dropout_plus_availability",
    "M4_structured_plus_availability",
    "M6_structured_plus_availability_plus_subclass",
)
CORE_COMPARISONS = (
    ("M2_vs_M1", "M2_structured_masking", "M1_random_dropout"),
    ("M6_vs_M1", "M6_structured_plus_availability_plus_subclass", "M1_random_dropout"),
    ("M6_vs_M2", "M6_structured_plus_availability_plus_subclass", "M2_structured_masking"),
)
STRONG_BACKBONES = ("xresnet1d101_like", "inception_time1d")
MACRO_METRICS = ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll", "macro_brier", "macro_ece")
BOOTSTRAP_METRICS = ("macro_auprc",)


@dataclass(frozen=True)
class PredictionBundle:
    logits: np.ndarray
    probs: np.ndarray
    targets: np.ndarray
    preds: np.ndarray
    patient_ids: np.ndarray

    def subset(self, indices: np.ndarray) -> "PredictionBundle":
        return PredictionBundle(
            logits=self.logits[indices],
            probs=self.probs[indices],
            targets=self.targets[indices],
            preds=self.preds[indices],
            patient_ids=self.patient_ids[indices],
        )


def run_id(row: Mapping[str, Any]) -> str:
    value = f"{row['method_id']}_{row['backbone']}_seed{row['seed']}"
    tag = str(row.get("tag", ""))
    if tag:
        value = f"{value}_{tag}"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prediction_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logits = frame[[f"logit_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.float64)
    probs = frame[[f"prob_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.float64)
    targets = frame[[f"y_true_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.int64)
    preds = frame[[f"pred_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.int64)
    return logits, probs, targets, preds


def read_prediction(path: Path) -> tuple[pd.DataFrame, PredictionBundle]:
    frame = pd.read_csv(path)
    required = {
        "ecg_id",
        "patient_id",
        "split",
        "strat_fold",
        "method_id",
        "pattern",
        "fill_mode",
        "threshold_source_split",
        *(f"logit_{label}" for label in LABEL_ORDER),
        *(f"prob_{label}" for label in LABEL_ORDER),
        *(f"y_true_{label}" for label in LABEL_ORDER),
        *(f"pred_{label}" for label in LABEL_ORDER),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    threshold_sources = set(str(x) for x in frame["threshold_source_split"].dropna().unique())
    if threshold_sources != {"val"}:
        raise RuntimeError(f"{path} threshold_source_split must be val, got {sorted(threshold_sources)}")
    if set(frame["split"].dropna().unique()) != {"test"}:
        raise RuntimeError(f"{path} must contain test split predictions only")
    if set(frame["strat_fold"].dropna().astype(int).unique()) != {10}:
        raise RuntimeError(f"{path} must contain strat_fold 10 predictions only")
    logits, probs, targets, preds = prediction_arrays(frame)
    bundle = PredictionBundle(
        logits=logits,
        probs=probs,
        targets=targets,
        preds=preds,
        patient_ids=frame["patient_id"].to_numpy(),
    )
    return frame, bundle


def metric_row_for_prediction(path: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    frame, bundle = read_prediction(path)
    metrics = compute_bootstrap_metrics(
        logits=bundle.logits,
        targets=bundle.targets,
        preds=bundle.preds,
        probs=bundle.probs,
    )
    calibration = compute_calibration_metrics(targets=bundle.targets, probs=bundle.probs)
    mask_cols = [col for col in frame.columns if col.startswith("availability_mask_")]
    visible_leads = float(frame[mask_cols].sum(axis=1).mean()) if mask_cols else float("nan")
    method_run_id = str(frame["method_id"].iloc[0])
    pattern = str(frame["pattern"].iloc[0])
    fill_mode = str(frame["fill_mode"].iloc[0])
    row: dict[str, Any] = {
        "group": meta["group"],
        "method": meta["method_id"],
        "backbone": meta["backbone"],
        "seed": int(meta["seed"]),
        "tag": str(meta.get("tag", "")),
        "method_run_id": method_run_id,
        "pattern": pattern,
        "pattern_file": path.stem,
        "fill_mode": fill_mode,
        "prediction_path": str(path),
        "n": int(len(frame)),
        "n_patients": int(pd.Series(bundle.patient_ids).nunique()),
        "visible_leads": visible_leads,
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "macro_f1": metrics["macro_f1"],
        "bce_nll": metrics["bce_nll"],
        "macro_brier": metrics["macro_brier"],
        "macro_ece": calibration["macro_ece"],
        "macro_calibration_nll": calibration["macro_nll"],
    }
    for label in LABEL_ORDER:
        row[f"per_class_{label}_auroc"] = metrics["per_class_auroc"][label]
        row[f"per_class_{label}_auprc"] = metrics["per_class_auprc"][label]
        row[f"per_class_{label}_f1"] = metrics["per_class_f1"][label]
        row[f"per_class_{label}_ece"] = calibration["per_class_ece"][label]
        row[f"per_class_{label}_brier"] = calibration["per_class_brier"][label]
    return row


def load_all_metric_rows(predictions_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for plan_row in build_experiment_plan():
        if str(plan_row["method_id"]) == "SPECIALIST_fixed_pattern":
            continue
        rid = run_id(plan_row)
        for fill_mode in ("mean_fill", "zero_fill"):
            for pattern, pattern_file in PATTERN_FILES.items():
                path = predictions_dir / rid / fill_mode / "test" / f"{pattern_file}.csv"
                if not path.exists():
                    missing.append({**plan_row, "run_id": rid, "fill_mode": fill_mode, "pattern": pattern, "path": str(path)})
                    continue
                rows.append(metric_row_for_prediction(path, plan_row))
    return rows, missing


def aggregate_metric_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    key_cols = ("group", "method", "backbone", "seed", "tag", "fill_mode")
    for row in metric_rows:
        by_run[tuple(row[col] for col in key_cols)][str(row["pattern"])] = row
    out: list[dict[str, Any]] = []
    for key, pattern_rows in by_run.items():
        base = dict(zip(key_cols, key))
        for aggregate_name, members in AGGREGATES.items():
            if not all(member in pattern_rows for member in members):
                continue
            agg: dict[str, Any] = {**base, "pattern_or_aggregate": aggregate_name, "n_patterns": len(members)}
            for metric in MACRO_METRICS:
                agg[metric] = float(np.nanmean([float(pattern_rows[member][metric]) for member in members]))
            for label in LABEL_ORDER:
                for suffix in ("auroc", "auprc", "f1", "ece", "brier"):
                    col = f"per_class_{label}_{suffix}"
                    agg[col] = float(np.nanmean([float(pattern_rows[member][col]) for member in members]))
            out.append(agg)
        for pattern, row in pattern_rows.items():
            agg = {
                **base,
                "pattern_or_aggregate": pattern,
                "n_patterns": 1,
            }
            for metric in MACRO_METRICS:
                agg[metric] = row[metric]
            for label in LABEL_ORDER:
                for suffix in ("auroc", "auprc", "f1", "ece", "brier"):
                    agg[f"per_class_{label}_{suffix}"] = row[f"per_class_{label}_{suffix}"]
            out.append(agg)
    return out


def summarize_mean_sd(rows: Sequence[Mapping[str, Any]], *, group: str = "primary", fill_mode: str = "mean_fill") -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    sub = frame[(frame["group"] == group) & (frame["fill_mode"] == fill_mode)]
    summary: list[dict[str, Any]] = []
    for (backbone, method, pattern), part in sub.groupby(["backbone", "method", "pattern_or_aggregate"], dropna=False):
        row: dict[str, Any] = {
            "backbone": backbone,
            "method": method,
            "pattern_or_aggregate": pattern,
            "n_seeds": int(part["seed"].nunique()),
        }
        for metric in MACRO_METRICS:
            values = part[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return sorted(summary, key=lambda r: (str(r["backbone"]), str(r["method"]), str(r["pattern_or_aggregate"])))


def t_critical_975(n: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return table.get(n, 1.96)


def build_seed_paired_delta_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(aggregate_rows)
    primary = frame[(frame["group"] == "primary") & (frame["fill_mode"] == "mean_fill")]
    index_cols = ["backbone", "seed", "pattern_or_aggregate"]
    metric_cols = list(MACRO_METRICS)
    value_map = {
        (row["method"], row["backbone"], int(row["seed"]), row["pattern_or_aggregate"]): row
        for row in primary.to_dict("records")
    }
    delta_rows: list[dict[str, Any]] = []
    for comp_id, method_a, method_b in CORE_COMPARISONS:
        for backbone in sorted(primary["backbone"].unique()):
            seeds = sorted(int(x) for x in primary["seed"].unique())
            patterns = sorted(primary["pattern_or_aggregate"].unique())
            for seed in seeds:
                for pattern in patterns:
                    key_a = (method_a, backbone, seed, pattern)
                    key_b = (method_b, backbone, seed, pattern)
                    if key_a not in value_map or key_b not in value_map:
                        continue
                    for metric in metric_cols:
                        delta_rows.append(
                            {
                                "comparison_id": comp_id,
                                "method_a": method_a,
                                "method_b": method_b,
                                "backbone": backbone,
                                "seed": seed,
                                "pattern_or_aggregate": pattern,
                                "metric": metric,
                                "value_a": float(value_map[key_a][metric]),
                                "value_b": float(value_map[key_b][metric]),
                                "delta": float(value_map[key_a][metric]) - float(value_map[key_b][metric]),
                            }
                        )
    delta_frame = pd.DataFrame(delta_rows)
    summary_rows: list[dict[str, Any]] = []
    for keys, part in delta_frame.groupby(["comparison_id", "method_a", "method_b", "backbone", "pattern_or_aggregate", "metric"]):
        comparison_id, method_a, method_b, backbone, pattern, metric = keys
        deltas = part["delta"].astype(float).to_numpy()
        n = len(deltas)
        mean = float(np.mean(deltas))
        sd = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
        half = t_critical_975(n) * sd / math.sqrt(n) if n > 1 else 0.0
        summary_rows.append(
            {
                "comparison_id": comparison_id,
                "method_a": method_a,
                "method_b": method_b,
                "backbone": backbone,
                "pattern_or_aggregate": pattern,
                "metric": metric,
                "n_seeds": n,
                "delta_mean": mean,
                "delta_sd": sd,
                "delta_ci95_low_seed_t": mean - half,
                "delta_ci95_high_seed_t": mean + half,
                "sign_count_positive": int(np.sum(deltas > 0)),
                "sign_count_negative": int(np.sum(deltas < 0)),
                "all_seed_deltas": ";".join(f"{x:.6f}" for x in deltas),
            }
        )
    return delta_rows, sorted(summary_rows, key=lambda r: (r["comparison_id"], r["backbone"], r["pattern_or_aggregate"], r["metric"]))


def load_bundle(path: Path) -> PredictionBundle:
    _, bundle = read_prediction(path)
    return bundle


def patient_groups(patient_ids: np.ndarray) -> dict[Any, np.ndarray]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, patient_id in enumerate(patient_ids):
        groups[patient_id].append(idx)
    return {patient_id: np.asarray(indices, dtype=np.int64) for patient_id, indices in groups.items()}


def sampled_indices(groups: Mapping[Any, np.ndarray], sampled_patients: np.ndarray) -> np.ndarray:
    return np.concatenate([groups[patient_id] for patient_id in sampled_patients])


def bundle_metric(bundle: PredictionBundle, metric: str) -> float:
    result = compute_bootstrap_metrics(logits=bundle.logits, probs=bundle.probs, targets=bundle.targets, preds=bundle.preds)
    return float(result[metric])


def aggregate_bundle_metric(
    data: Mapping[str, PredictionBundle],
    *,
    pattern_or_aggregate: str,
    metric: str,
    sampled: np.ndarray | None = None,
    groups: Mapping[str, Mapping[Any, np.ndarray]] | None = None,
) -> float:
    patterns = AGGREGATES.get(pattern_or_aggregate, (pattern_or_aggregate,))
    values: list[float] = []
    for pattern in patterns:
        bundle = data[pattern]
        if sampled is not None:
            if groups is None:
                raise ValueError("groups are required when sampled is provided")
            bundle = bundle.subset(sampled_indices(groups[pattern], sampled))
        values.append(bundle_metric(bundle, metric))
    return float(np.nanmean(values))


def summarize_distribution(values: np.ndarray, observed: float) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "observed_delta": observed,
            "bootstrap_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "prob_delta_gt_0": float("nan"),
            "n_bootstrap_valid": 0,
            "invalid_replicates": int(values.size),
        }
    return {
        "observed_delta": observed,
        "bootstrap_mean": float(np.mean(finite)),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "prob_delta_gt_0": float(np.mean(finite > 0)),
        "n_bootstrap_valid": int(finite.size),
        "invalid_replicates": int(values.size - finite.size),
    }


def build_path_index(predictions_dir: Path) -> dict[tuple[str, str, int, str, str, str], Path]:
    index: dict[tuple[str, str, int, str, str, str], Path] = {}
    for row in build_experiment_plan():
        if row["group"] != "primary" or row["method_id"] == "SPECIALIST_fixed_pattern":
            continue
        rid = run_id(row)
        for fill_mode in ("mean_fill", "zero_fill"):
            for pattern, pattern_file in PATTERN_FILES.items():
                path = predictions_dir / rid / fill_mode / "test" / f"{pattern_file}.csv"
                if path.exists():
                    index[(row["method_id"], row["backbone"], int(row["seed"]), fill_mode, pattern, str(row.get("tag", "")))] = path
    return index


def build_bootstrap_delta_rows(
    predictions_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
    run_seeds: Sequence[int],
    backbones: Sequence[str],
    comparisons: Sequence[tuple[str, str, str]],
    pattern_or_aggregates: Sequence[str],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    path_index = build_path_index(predictions_dir)
    rows: list[dict[str, Any]] = []
    methods = sorted({method for _, a, b in comparisons for method in (a, b)})
    needed_patterns = tuple(dict.fromkeys(["full", *HARD_OVERALL]))
    for backbone in backbones:
        seeds = tuple(int(x) for x in run_seeds)
        base_path = path_index[("M1_random_dropout", backbone, seeds[0], "mean_fill", "full", "")]
        base_bundle = load_bundle(base_path)
        patient_ids = np.asarray(sorted(pd.Series(base_bundle.patient_ids).unique()))
        samples = [rng.choice(patient_ids, size=len(patient_ids), replace=True) for _ in range(n_bootstrap)]
        cache: dict[tuple[str, int, str], PredictionBundle] = {}
        groups: dict[tuple[str, int, str], Mapping[Any, np.ndarray]] = {}
        for method in methods:
            for run_seed in seeds:
                for pattern in needed_patterns:
                    key = (method, backbone, run_seed, "mean_fill", pattern, "")
                    cache[(method, run_seed, pattern)] = load_bundle(path_index[key])
                    groups[(method, run_seed, pattern)] = patient_groups(cache[(method, run_seed, pattern)].patient_ids)

        observed_values: dict[tuple[str, int, str, str], float] = {}
        for method in methods:
            for run_seed in seeds:
                data = {pattern: cache[(method, run_seed, pattern)] for pattern in needed_patterns}
                for pattern_or_aggregate in pattern_or_aggregates:
                    for metric in BOOTSTRAP_METRICS:
                        observed_values[(method, run_seed, pattern_or_aggregate, metric)] = aggregate_bundle_metric(
                            data,
                            pattern_or_aggregate=pattern_or_aggregate,
                            metric=metric,
                        )

        dist_map: dict[tuple[str, str], list[float]] = {
            (comparison_id, pattern_or_aggregate): []
            for comparison_id, _, _ in comparisons
            for pattern_or_aggregate in pattern_or_aggregates
        }
        for sample_idx, sampled in enumerate(samples, start=1):
            if sample_idx == 1 or sample_idx % 100 == 0 or sample_idx == n_bootstrap:
                print(f"bootstrap {backbone}: replicate {sample_idx}/{n_bootstrap}", flush=True)
            sampled_values: dict[tuple[str, int, str, str], float] = {}
            for method in methods:
                for run_seed in seeds:
                    pattern_values: dict[tuple[str, str], float] = {}
                    for pattern in needed_patterns:
                        bundle = cache[(method, run_seed, pattern)].subset(sampled_indices(groups[(method, run_seed, pattern)], sampled))
                        for metric in BOOTSTRAP_METRICS:
                            pattern_values[(pattern, metric)] = bundle_metric(bundle, metric)
                    for pattern_or_aggregate in pattern_or_aggregates:
                        members = AGGREGATES.get(pattern_or_aggregate, (pattern_or_aggregate,))
                        for metric in BOOTSTRAP_METRICS:
                            sampled_values[(method, run_seed, pattern_or_aggregate, metric)] = float(
                                np.nanmean([pattern_values[(member, metric)] for member in members])
                            )
            for comparison_id, method_a, method_b in comparisons:
                for pattern_or_aggregate in pattern_or_aggregates:
                    for metric in BOOTSTRAP_METRICS:
                        seed_deltas = [
                            sampled_values[(method_a, run_seed, pattern_or_aggregate, metric)]
                            - sampled_values[(method_b, run_seed, pattern_or_aggregate, metric)]
                            for run_seed in seeds
                        ]
                        dist_map[(comparison_id, pattern_or_aggregate)].append(float(np.nanmean(seed_deltas)))

        for comparison_id, method_a, method_b in comparisons:
            for pattern_or_aggregate in pattern_or_aggregates:
                for metric in BOOTSTRAP_METRICS:
                    observed_seed_deltas = [
                        observed_values[(method_a, run_seed, pattern_or_aggregate, metric)]
                        - observed_values[(method_b, run_seed, pattern_or_aggregate, metric)]
                        for run_seed in seeds
                    ]
                    observed = float(np.nanmean(observed_seed_deltas))
                    dist = np.asarray(dist_map[(comparison_id, pattern_or_aggregate)], dtype=np.float64)
                    rows.append(
                        {
                            "comparison_id": comparison_id,
                            "method_a": method_a,
                            "method_b": method_b,
                            "backbone": backbone,
                            "pattern_or_aggregate": pattern_or_aggregate,
                            "metric": metric,
                            "n_seeds": len(seeds),
                            "bootstrap_run_seeds": ";".join(str(x) for x in seeds),
                            "n_bootstrap": n_bootstrap,
                            "bootstrap_seed": seed,
                            "sampling_unit": "patient_id",
                            **summarize_distribution(dist, observed),
                        }
                    )
                    print(
                        f"bootstrap {comparison_id} {backbone} {pattern_or_aggregate} {metric}: "
                        f"delta={observed:.5f}",
                        flush=True,
                    )
    return rows


def build_pareto_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(aggregate_rows)
    sub = frame[(frame["group"] == "primary") & (frame["fill_mode"] == "mean_fill")]
    rows: list[dict[str, Any]] = []
    for (backbone, method, seed), part in sub.groupby(["backbone", "method", "seed"]):
        full = part[part["pattern_or_aggregate"] == "full"]
        hard = part[part["pattern_or_aggregate"] == "hard_overall_avg"]
        hard_struct = part[part["pattern_or_aggregate"] == "hard_structured_avg"]
        if full.empty or hard.empty:
            continue
        rows.append(
            {
                "backbone": backbone,
                "method": method,
                "seed": int(seed),
                "fill_mode": "mean_fill",
                "full_macro_auprc": float(full["macro_auprc"].iloc[0]),
                "hard_overall_macro_auprc": float(hard["macro_auprc"].iloc[0]),
                "hard_structured_macro_auprc": float(hard_struct["macro_auprc"].iloc[0]) if not hard_struct.empty else float("nan"),
                "hard_minus_full_macro_auprc": float(hard["macro_auprc"].iloc[0] - full["macro_auprc"].iloc[0]),
            }
        )
    return rows


def build_degradation_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    visible = {"full": 12, "random-1": 11, "random-3": 9, "random-6": 6}
    frame = pd.DataFrame(aggregate_rows)
    sub = frame[
        (frame["group"] == "primary")
        & (frame["fill_mode"] == "mean_fill")
        & (frame["pattern_or_aggregate"].isin(visible))
        & (frame["method"].isin(["M1_random_dropout", "M2_structured_masking", "M6_structured_plus_availability_plus_subclass"]))
    ]
    rows: list[dict[str, Any]] = []
    for (backbone, method, pattern), part in sub.groupby(["backbone", "method", "pattern_or_aggregate"]):
        rows.append(
            {
                "backbone": backbone,
                "method": method,
                "pattern_or_aggregate": pattern,
                "visible_leads": visible[str(pattern)],
                "n_seeds": int(part["seed"].nunique()),
                "macro_auprc_mean": float(part["macro_auprc"].mean()),
                "macro_auprc_sd": float(part["macro_auprc"].std(ddof=1)),
                "macro_auroc_mean": float(part["macro_auroc"].mean()),
                "macro_f1_mean": float(part["macro_f1"].mean()),
            }
        )
    return sorted(rows, key=lambda r: (r["backbone"], r["method"], -int(r["visible_leads"])))


def build_heatmap_delta_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(aggregate_rows)
    primary = frame[(frame["group"] == "primary") & (frame["fill_mode"] == "mean_fill")]
    rows: list[dict[str, Any]] = []
    for comparison_id, method_a, method_b in CORE_COMPARISONS[:2]:
        for backbone in sorted(primary["backbone"].unique()):
            for pattern in HARD_OVERALL:
                for label in LABEL_ORDER:
                    values = []
                    for run_seed in sorted(primary["seed"].unique()):
                        a = primary[
                            (primary["method"] == method_a)
                            & (primary["backbone"] == backbone)
                            & (primary["seed"] == run_seed)
                            & (primary["pattern_or_aggregate"] == pattern)
                        ]
                        b = primary[
                            (primary["method"] == method_b)
                            & (primary["backbone"] == backbone)
                            & (primary["seed"] == run_seed)
                            & (primary["pattern_or_aggregate"] == pattern)
                        ]
                        if a.empty or b.empty:
                            continue
                        values.append(float(a[f"per_class_{label}_auprc"].iloc[0]) - float(b[f"per_class_{label}_auprc"].iloc[0]))
                    if values:
                        rows.append(
                            {
                                "comparison_id": comparison_id,
                                "method_a": method_a,
                                "method_b": method_b,
                                "backbone": backbone,
                                "pattern_or_aggregate": pattern,
                                "class": label,
                                "metric": "per_class_auprc",
                                "delta_mean": float(np.mean(values)),
                                "delta_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                "sign_count_positive": int(np.sum(np.asarray(values) > 0)),
                                "n_seeds": len(values),
                            }
                        )
    return rows


def make_pareto_figure(pareto_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    frame = pd.DataFrame(pareto_rows)
    summary = frame.groupby(["backbone", "method"], as_index=False).agg(
        full_macro_auprc=("full_macro_auprc", "mean"),
        hard_overall_macro_auprc=("hard_overall_macro_auprc", "mean"),
        full_sd=("full_macro_auprc", "std"),
        hard_sd=("hard_overall_macro_auprc", "std"),
    )
    methods = list(PRIMARY_METHODS)
    colors = dict(zip(methods, plt.cm.tab10(np.linspace(0, 1, len(methods)))))
    markers = {"resnet1d_tiny": "o", "xresnet1d101_like": "s", "inception_time1d": "^"}
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for _, row in summary.iterrows():
        ax.errorbar(
            row["full_macro_auprc"],
            row["hard_overall_macro_auprc"],
            xerr=row["full_sd"] if not np.isnan(row["full_sd"]) else None,
            yerr=row["hard_sd"] if not np.isnan(row["hard_sd"]) else None,
            fmt=markers.get(row["backbone"], "o"),
            color=colors.get(row["method"], "black"),
            markersize=7,
            capsize=2,
            alpha=0.9,
        )
    for method in methods:
        ax.scatter([], [], color=colors[method], label=method)
    ax.set_xlabel("Full 12-lead Macro AUPRC")
    ax.set_ylabel("Hard overall Macro AUPRC")
    ax.set_title("Clean-vs-robust Pareto trade-off")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"clean_vs_robust_pareto.{suffix}", dpi=300)
    plt.close(fig)


def make_degradation_figure(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    methods = ["M1_random_dropout", "M2_structured_masking", "M6_structured_plus_availability_plus_subclass"]
    colors = dict(zip(methods, ["#4C78A8", "#F58518", "#54A24B"]))
    backbones = sorted(frame["backbone"].unique())
    fig, axes = plt.subplots(1, len(backbones), figsize=(5.0 * len(backbones), 4.2), sharey=True)
    if len(backbones) == 1:
        axes = [axes]
    for ax, backbone in zip(axes, backbones):
        sub_backbone = frame[frame["backbone"] == backbone]
        for method in methods:
            sub = sub_backbone[sub_backbone["method"] == method].sort_values("visible_leads")
            ax.errorbar(
                sub["visible_leads"],
                sub["macro_auprc_mean"],
                yerr=sub["macro_auprc_sd"],
                marker="o",
                color=colors[method],
                label=method,
                capsize=2,
            )
        ax.set_title(backbone)
        ax.set_xlabel("Visible leads")
        ax.grid(True, alpha=0.25)
        ax.invert_xaxis()
    axes[0].set_ylabel("Macro AUPRC")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle("Degradation over random missing lead counts", y=1.02)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"degradation_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_heatmap_figure(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    sub = frame[
        (frame["comparison_id"] == "M6_vs_M1")
        & (frame["backbone"].isin(STRONG_BACKBONES))
    ]
    averaged = sub.groupby(["pattern_or_aggregate", "class"], as_index=False)["delta_mean"].mean()
    matrix = averaged.pivot(index="pattern_or_aggregate", columns="class", values="delta_mean").reindex(index=list(HARD_OVERALL), columns=list(LABEL_ORDER))
    vmax = float(np.nanmax(np.abs(matrix.to_numpy()))) if not matrix.empty else 0.05
    vmax = max(vmax, 0.01)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title("Class-wise AUPRC delta: M6 - M1, strong backbones")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Delta AUPRC")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"classwise_heatmap_m6_vs_m1.{suffix}", dpi=300)
    plt.close(fig)


def build_claim_verdicts(seed_delta_summary: Sequence[Mapping[str, Any]], bootstrap_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(seed_delta_summary)
    boot = pd.DataFrame(bootstrap_rows)
    verdicts: list[dict[str, Any]] = []
    for comparison_id in ("M2_vs_M1", "M6_vs_M1"):
        for backbone in STRONG_BACKBONES:
            for pattern in ("hard_structured_avg", "hard_overall_avg"):
                row = frame[
                    (frame["comparison_id"] == comparison_id)
                    & (frame["backbone"] == backbone)
                    & (frame["pattern_or_aggregate"] == pattern)
                    & (frame["metric"] == "macro_auprc")
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                b = boot[
                    (boot["comparison_id"] == comparison_id)
                    & (boot["backbone"] == backbone)
                    & (boot["pattern_or_aggregate"] == pattern)
                    & (boot["metric"] == "macro_auprc")
                ]
                boot_ci = ""
                if not b.empty:
                    boot_ci = f"{float(b['ci_low'].iloc[0]):+.4f}, {float(b['ci_high'].iloc[0]):+.4f}"
                passes = bool(float(r["delta_mean"]) > 0 and int(r["sign_count_positive"]) >= 4)
                verdicts.append(
                    {
                        "claim": f"{comparison_id} positive on {backbone} / {pattern}",
                        "comparison_id": comparison_id,
                        "backbone": backbone,
                        "pattern_or_aggregate": pattern,
                        "delta_mean": float(r["delta_mean"]),
                        "delta_sd": float(r["delta_sd"]),
                        "sign_count_positive": int(r["sign_count_positive"]),
                        "n_seeds": int(r["n_seeds"]),
                        "seed_ci95_low": float(r["delta_ci95_low_seed_t"]),
                        "seed_ci95_high": float(r["delta_ci95_high_seed_t"]),
                        "bootstrap_ci95": boot_ci,
                        "passes_prespecified_rule": passes,
                    }
                )
    return verdicts


def write_claim_summary(path: Path, verdicts: Sequence[Mapping[str, Any]], seed_delta_summary: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(verdicts)
    lines = ["# Reviewer-Defense Claim Verdict", ""]
    if frame.empty:
        lines.append("No claim verdict rows were available.")
    else:
        passed = int(frame["passes_prespecified_rule"].sum())
        total = len(frame)
        lines.append(f"- Prespecified strong-backbone structured-vs-random checks passed: {passed}/{total}.")
        for row in frame.to_dict("records"):
            verdict = "PASS" if row["passes_prespecified_rule"] else "FAIL"
            lines.append(
                f"- {verdict}: {row['claim']}; delta={row['delta_mean']:+.4f} +/- {row['delta_sd']:.4f}, "
                f"sign={row['sign_count_positive']}/{row['n_seeds']}, seed-CI="
                f"[{row['seed_ci95_low']:+.4f}, {row['seed_ci95_high']:+.4f}], "
                f"bootstrap-CI=[{row['bootstrap_ci95']}]."
            )
        if passed == total:
            lines.append("")
            lines.append("Conclusion: the core claim is supported under the pre-specified rule for both strong backbones.")
        else:
            lines.append("")
            lines.append("Conclusion: the core claim needs narrowing to the backbones/aggregates that pass the rule.")
    sd = pd.DataFrame(seed_delta_summary)
    trade = sd[
        (sd["comparison_id"].isin(["M2_vs_M1", "M6_vs_M1"]))
        & (sd["pattern_or_aggregate"] == "full")
        & (sd["metric"] == "macro_auprc")
    ]
    lines.append("")
    lines.append("## Full-lead trade-off")
    for row in trade.sort_values(["comparison_id", "backbone"]).to_dict("records"):
        flag = "report as trade-off" if float(row["delta_mean"]) < -0.010 else "small/no full-lead cost"
        lines.append(
            f"- {row['comparison_id']} / {row['backbone']}: full-lead delta={float(row['delta_mean']):+.4f} "
            f"({int(row['sign_count_positive'])}/{int(row['n_seeds'])} positive), {flag}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metric_rows, missing_rows = load_all_metric_rows(args.predictions_dir)
    if missing_rows:
        write_csv(out_dir / "missing_prediction_files.csv", missing_rows)
        raise RuntimeError(f"Missing prediction files: {len(missing_rows)}; see {out_dir / 'missing_prediction_files.csv'}")
    aggregate_rows = aggregate_metric_rows(metric_rows)
    primary_table = summarize_mean_sd(aggregate_rows)
    seed_delta_rows, seed_delta_summary = build_seed_paired_delta_rows(aggregate_rows)
    pareto_rows = build_pareto_rows(aggregate_rows)
    degradation_rows = build_degradation_rows(aggregate_rows)
    heatmap_rows = build_heatmap_delta_rows(aggregate_rows)
    bootstrap_rows = build_bootstrap_delta_rows(
        args.predictions_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
        run_seeds=args.bootstrap_run_seeds,
        backbones=args.bootstrap_backbones,
        comparisons=CORE_COMPARISONS[:2],
        pattern_or_aggregates=("full", "hard_structured_avg", "hard_overall_avg"),
    )
    verdicts = build_claim_verdicts(seed_delta_summary, bootstrap_rows)

    write_csv(out_dir / "prediction_metric_rows.csv", metric_rows)
    write_csv(out_dir / "aggregate_metric_rows.csv", aggregate_rows)
    write_csv(out_dir / "primary_table_mean_sd.csv", primary_table)
    write_csv(out_dir / "primary_seed_paired_deltas.csv", seed_delta_rows)
    write_csv(out_dir / "primary_seed_paired_delta_summary.csv", seed_delta_summary)
    write_csv(out_dir / "bootstrap_delta_ci.csv", bootstrap_rows)
    write_csv(out_dir / "clean_vs_robust_pareto_data.csv", pareto_rows)
    write_csv(out_dir / "degradation_curve_data.csv", degradation_rows)
    write_csv(out_dir / "classwise_heatmap_delta_data.csv", heatmap_rows)
    write_csv(out_dir / "claim_verdicts.csv", verdicts)
    write_claim_summary(out_dir / "claim_verdict_summary.md", verdicts, seed_delta_summary)

    make_pareto_figure(pareto_rows, fig_dir)
    make_degradation_figure(degradation_rows, fig_dir)
    make_heatmap_figure(heatmap_rows, fig_dir)

    summary = {
        "predictions_dir": str(args.predictions_dir),
        "output_dir": str(out_dir),
        "n_metric_rows": len(metric_rows),
        "n_aggregate_rows": len(aggregate_rows),
        "n_primary_table_rows": len(primary_table),
        "n_seed_delta_rows": len(seed_delta_rows),
        "n_bootstrap_delta_rows": len(bootstrap_rows),
        "n_claim_verdicts": len(verdicts),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_run_seeds": [int(x) for x in args.bootstrap_run_seeds],
        "label_order": list(LABEL_ORDER),
        "threshold_source_split": "val",
        "test_fold_only": True,
        "external_training_used": False,
    }
    write_json(out_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/reviewer_defense_20260701/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_defense_20260701/final_analysis"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20240604)
    parser.add_argument("--bootstrap-run-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--bootstrap-backbones", nargs="+", default=list(STRONG_BACKBONES))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
