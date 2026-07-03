#!/usr/bin/env python3
"""Patient-level paired bootstrap for the three-strategy benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.statistics.bootstrap import (
    PredictionData,
    compute_metric_bundle,
    generate_patient_bootstrap_samples,
    load_prediction_csv,
    metric_value,
    paired_delta_summary,
    patient_groups,
    prediction_csv_path,
    sampled_indices_from_patients,
)

BACKBONE = "xresnet1d101_like"
SPLIT = "test"
METRICS = ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll", "macro_brier")
DEFAULT_METRICS = ("macro_auprc",)
HARD_STRUCTURED_PATTERNS = (
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
LOW_LEAD_I_II_PATTERNS = (
    "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2",
    "challenge_2_I_II",
)


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
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def reviewer_method_id(method: str, seed: int) -> str:
    return f"{method}_{BACKBONE}_seed{int(seed)}"


def load_strategy_prediction(
    prediction_dir: Path,
    *,
    method: str,
    seed: int,
    fill_mode: str,
    pattern: str,
) -> tuple[PredictionData, Path]:
    method_id = reviewer_method_id(method, seed)
    path = prediction_csv_path(prediction_dir, method_id, fill_mode, SPLIT, pattern)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")
    return (
        load_prediction_csv(path, method_id=method_id, pattern=pattern, split=SPLIT, fill_mode=fill_mode),
        path,
    )


def align_to_reference(reference: PredictionData, other: PredictionData) -> PredictionData:
    if set(reference.ecg_ids.tolist()) != set(other.ecg_ids.tolist()):
        raise RuntimeError(f"Prediction ecg_id sets differ for {reference.pattern}")
    order = {int(ecg_id): idx for idx, ecg_id in enumerate(other.ecg_ids.tolist())}
    indices = np.asarray([order[int(ecg_id)] for ecg_id in reference.ecg_ids.tolist()], dtype=np.int64)
    aligned = other.subset(indices)
    if not np.array_equal(reference.ecg_ids, aligned.ecg_ids):
        raise RuntimeError("Failed to align prediction rows by ecg_id")
    if not np.array_equal(reference.patient_ids, aligned.patient_ids):
        raise RuntimeError("Patient IDs differ after ecg_id alignment")
    if not np.array_equal(reference.targets, aligned.targets):
        raise RuntimeError("Targets differ after ecg_id alignment")
    return aligned


def comparison_specs() -> list[dict[str, Any]]:
    return [
        {
            "comparison_id": "M6_vs_M1_mean_fill",
            "method_a": "M6_structured_plus_availability_plus_subclass",
            "fill_a": "mean_fill",
            "method_b": "M1_random_dropout",
            "fill_b": "mean_fill",
            "patterns": HARD_STRUCTURED_PATTERNS,
            "claim_group": "structured_missing_single_robust",
        },
        {
            "comparison_id": "M6_i_ii_limb_recon_vs_M6_mean_fill",
            "method_a": "M6_structured_plus_availability_plus_subclass",
            "fill_a": "physiology_limb_reconstruction_fill",
            "method_b": "M6_structured_plus_availability_plus_subclass",
            "fill_b": "mean_fill",
            "patterns": LOW_LEAD_I_II_PATTERNS,
            "claim_group": "low_lead_i_ii_reconstruction_complement",
        },
        {
            "comparison_id": "M0_i_ii_limb_recon_vs_M6_mean_fill",
            "method_a": "M0_full_no_masking",
            "fill_a": "physiology_limb_reconstruction_fill",
            "method_b": "M6_structured_plus_availability_plus_subclass",
            "fill_b": "mean_fill",
            "patterns": LOW_LEAD_I_II_PATTERNS,
            "claim_group": "reconstruction_only_vs_single_robust",
        },
    ]


def bootstrap_pair(
    data_a: PredictionData,
    data_b: PredictionData,
    *,
    n_bootstrap: int,
    seed: int,
    metrics: Sequence[str],
) -> dict[str, dict[str, Any]]:
    data_b = align_to_reference(data_a, data_b)
    patients_a = set(int(x) for x in data_a.patient_ids.tolist())
    patients_b = set(int(x) for x in data_b.patient_ids.tolist())
    if patients_a != patients_b:
        raise RuntimeError("Patient sets differ across paired predictions")

    observed_a = metric_bundle_for_requested_metrics(data_a, metrics)
    observed_b = metric_bundle_for_requested_metrics(data_b, metrics)
    groups_a = patient_groups(data_a.patient_ids)
    groups_b = patient_groups(data_b.patient_ids)
    samples = generate_patient_bootstrap_samples(data_a.patient_ids, n_bootstrap=n_bootstrap, seed=seed)
    deltas = {metric: [] for metric in metrics}
    for sampled_patients in samples:
        idx_a = sampled_indices_from_patients(groups_a, sampled_patients)
        idx_b = sampled_indices_from_patients(groups_b, sampled_patients)
        bundle_a = metric_bundle_for_requested_metrics(data_a.subset(idx_a), metrics)
        bundle_b = metric_bundle_for_requested_metrics(data_b.subset(idx_b), metrics)
        for metric in metrics:
            deltas[metric].append(metric_value(bundle_a, metric) - metric_value(bundle_b, metric))

    out: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        observed_delta = metric_value(observed_a, metric) - metric_value(observed_b, metric)
        summary = paired_delta_summary(np.asarray(deltas[metric], dtype=np.float64), observed_delta)
        out[metric] = {
            "observed_a": metric_value(observed_a, metric),
            "observed_b": metric_value(observed_b, metric),
            **summary,
        }
    return out


def macro_auprc_only(data: PredictionData, *, min_valid_labels: int = 3) -> float:
    values: list[float] = []
    for label_idx in range(data.targets.shape[1]):
        y_true = data.targets[:, label_idx]
        if np.unique(y_true).size < 2:
            continue
        values.append(float(average_precision_score(y_true, data.probs[:, label_idx])))
    if len(values) < min_valid_labels:
        return float("nan")
    return float(np.mean(values))


def metric_bundle_for_requested_metrics(data: PredictionData, metrics: Sequence[str]) -> dict[str, Any]:
    metric_set = set(metrics)
    if metric_set == {"macro_auprc"}:
        return {"macro_auprc": macro_auprc_only(data)}
    return compute_metric_bundle(data)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for spec in comparison_specs():
            for pattern_idx, pattern in enumerate(spec["patterns"]):
                data_a, path_a = load_strategy_prediction(
                    args.prediction_dir,
                    method=spec["method_a"],
                    seed=seed,
                    fill_mode=spec["fill_a"],
                    pattern=pattern,
                )
                data_b, path_b = load_strategy_prediction(
                    args.prediction_dir,
                    method=spec["method_b"],
                    seed=seed,
                    fill_mode=spec["fill_b"],
                    pattern=pattern,
                )
                summaries = bootstrap_pair(
                    data_a,
                    data_b,
                    n_bootstrap=args.n_bootstrap,
                    seed=int(args.seed) + int(seed) * 1000 + pattern_idx,
                    metrics=args.metrics,
                )
                for metric, summary in summaries.items():
                    rows.append(
                        {
                            "comparison_id": spec["comparison_id"],
                            "claim_group": spec["claim_group"],
                            "seed": int(seed),
                            "pattern": pattern,
                            "metric": metric,
                            "method_a": spec["method_a"],
                            "fill_a": spec["fill_a"],
                            "method_b": spec["method_b"],
                            "fill_b": spec["fill_b"],
                            **summary,
                            "sampling_unit": "patient_id",
                            "n_bootstrap": int(args.n_bootstrap),
                            "bootstrap_seed": int(args.seed) + int(seed) * 1000 + pattern_idx,
                            "threshold_source_split": "val",
                            "split": SPLIT,
                            "prediction_csv_a": str(path_a),
                            "prediction_csv_b": str(path_b),
                        }
                    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "strategy_headline_patient_bootstrap.csv", rows)
    macro_auprc_rows = [row for row in rows if row["metric"] == "macro_auprc"]
    write_csv(output_dir / "strategy_headline_patient_bootstrap_macro_auprc.csv", macro_auprc_rows)
    summary = {
        "output_dir": str(output_dir),
        "n_rows": len(rows),
        "n_macro_auprc_rows": len(macro_auprc_rows),
        "seeds": [int(seed) for seed in args.seeds],
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "metrics": list(args.metrics),
        "sampling_unit": "patient_id",
        "threshold_source_split": "val",
        "split": SPLIT,
        "records500_used": False,
        "filename_hr_used": False,
    }
    write_json(output_dir / "strategy_headline_patient_bootstrap_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("results/reviewer_defense_20260701/strategy_benchmark/imputation_predictions"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_defense_20260701/strategy_benchmark"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20240604)
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(DEFAULT_METRICS))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
