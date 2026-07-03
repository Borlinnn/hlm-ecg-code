#!/usr/bin/env python3
"""Recompute reviewer-defense tables from saved prediction CSVs only."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.calibration.calibration_metrics import compute_calibration_metrics
from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.statistics.bootstrap_metrics import compute_bootstrap_metrics


HARD_OVERALL_PATTERNS = (
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)


def method_seed(method_run_id: str) -> tuple[str, int | None]:
    match = re.search(r"_seed(\d+)$", method_run_id)
    if not match:
        return method_run_id, None
    return method_run_id[: match.start()], int(match.group(1))


def prediction_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logits = frame[[f"logit_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.float64)
    probs = frame[[f"prob_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.float64)
    targets = frame[[f"y_true_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.int64)
    preds = frame[[f"pred_{label}" for label in LABEL_ORDER]].to_numpy(dtype=np.int64)
    return logits, probs, targets, preds


def read_prediction_metric(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required = {
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
        raise RuntimeError(f"{path} missing prediction columns: {sorted(missing)}")
    thresholds = set(str(x) for x in frame["threshold_source_split"].dropna().unique())
    if thresholds != {"val"}:
        raise RuntimeError(f"{path} threshold_source_split must be val, got {sorted(thresholds)}")
    method_run = str(frame["method_id"].iloc[0])
    method, seed = method_seed(method_run)
    pattern = str(frame["pattern"].iloc[0])
    fill_mode = str(frame["fill_mode"].iloc[0])
    logits, probs, targets, preds = prediction_arrays(frame)
    metrics = compute_bootstrap_metrics(logits=logits, targets=targets, preds=preds, probs=probs)
    calibration = compute_calibration_metrics(targets=targets, probs=probs)
    mask_cols = [col for col in frame.columns if col.startswith("availability_mask_")]
    visible_leads = float(frame[mask_cols].sum(axis=1).mean()) if mask_cols else float("nan")

    row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "method_run_id": method_run,
        "pattern": pattern,
        "fill_mode": fill_mode,
        "prediction_path": str(path),
        "n": int(len(frame)),
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


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_mean_summary(metric_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    rows: list[dict[str, Any]] = []
    metric_cols = ["macro_auroc", "macro_auprc", "macro_f1", "bce_nll", "macro_brier", "macro_ece"]
    for (method, pattern), sub in frame.groupby(["method", "pattern"], dropna=False):
        row: dict[str, Any] = {"method": method, "pattern": pattern, "n_seeds": int(sub["seed"].nunique())}
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_sd"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return rows


def build_pareto_rows(metric_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    rows: list[dict[str, Any]] = []
    for (method, seed, fill_mode), sub in frame.groupby(["method", "seed", "fill_mode"], dropna=False):
        full = sub[sub["pattern"] == "full"]
        hard = sub[sub["pattern"].isin(HARD_OVERALL_PATTERNS)]
        if full.empty or hard.empty:
            continue
        rows.append(
            {
                "method": method,
                "seed": seed,
                "fill_mode": fill_mode,
                "full_macro_auprc": float(full["macro_auprc"].iloc[0]),
                "hard_overall_macro_auprc": float(hard["macro_auprc"].mean()),
                "hard_minus_full_macro_auprc": float(hard["macro_auprc"].mean() - full["macro_auprc"].iloc[0]),
                "n_hard_patterns": int(len(hard)),
            }
        )
    return rows


def build_heatmap_rows(metric_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        for label in LABEL_ORDER:
            rows.append(
                {
                    "method": row["method"],
                    "seed": row["seed"],
                    "pattern": row["pattern"],
                    "class": label,
                    "auroc": row[f"per_class_{label}_auroc"],
                    "auprc": row[f"per_class_{label}_auprc"],
                    "f1": row[f"per_class_{label}_f1"],
                    "ece": row[f"per_class_{label}_ece"],
                }
            )
    return rows


def build_claim_evidence(output_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "claim": "AUROC/F1/Brier/NLL/ECE recomputed from saved predictions without retraining.",
            "evidence_file": str(output_dir / "prediction_metric_rows.csv"),
            "risk": "Requires saved prediction CSVs with validation thresholds.",
        },
        {
            "claim": "Clean-lead versus hard-missing robustness trade-off is explicitly quantified.",
            "evidence_file": str(output_dir / "clean_vs_robust_pareto_data.csv"),
            "risk": "Rows appear only when full and at least one hard pattern are available.",
        },
        {
            "claim": "Class-by-pattern degradation can be audited label-wise.",
            "evidence_file": str(output_dir / "class_by_pattern_heatmap_data.csv"),
            "risk": "Per-class AUROC may be NaN if a class has one target value only.",
        },
    ]


def run_analysis(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    prediction_files = sorted(path for path in input_dir.rglob("*.csv") if path.is_file())
    if not prediction_files:
        raise RuntimeError(f"No prediction CSV files found under {input_dir}")
    metric_rows = [read_prediction_metric(path) for path in prediction_files]
    summary_rows = build_mean_summary(metric_rows)
    pareto_rows = build_pareto_rows(metric_rows)
    heatmap_rows = build_heatmap_rows(metric_rows)
    degradation_rows = [
        {
            "method": row["method"],
            "seed": row["seed"],
            "pattern": row["pattern"],
            "visible_leads": row["visible_leads"],
            "macro_auprc": row["macro_auprc"],
            "macro_auroc": row["macro_auroc"],
            "macro_f1": row["macro_f1"],
        }
        for row in metric_rows
    ]
    claim_rows = build_claim_evidence(output_dir)

    write_csv(output_dir / "prediction_metric_rows.csv", metric_rows)
    write_csv(output_dir / "prediction_metric_mean_summary.csv", summary_rows)
    write_csv(output_dir / "clean_vs_robust_pareto_data.csv", pareto_rows)
    write_csv(output_dir / "degradation_curve_data.csv", degradation_rows)
    write_csv(output_dir / "class_by_pattern_heatmap_data.csv", heatmap_rows)
    write_csv(output_dir / "claim_evidence_matrix.csv", claim_rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_prediction_files": len(prediction_files),
        "n_metric_rows": len(metric_rows),
        "n_pareto_rows": len(pareto_rows),
        "n_heatmap_rows": len(heatmap_rows),
        "threshold_source_split": "val",
        "records500_used": False,
        "external_training_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reviewer_defense_existing_predictions_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/stabilization/component_ablation_runs_20260608_145749/predictions"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_defense_20260701/existing_predictions"))
    args = parser.parse_args()
    print(json.dumps(run_analysis(input_dir=args.input_dir, output_dir=args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
