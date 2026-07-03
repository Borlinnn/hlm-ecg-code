#!/usr/bin/env python3
"""Evaluation-only imputation/reconstruction audit for reviewer-defense runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hlm_ecg.evaluation.supplemental_analysis import MethodRun, write_json
from hlm_ecg.evaluation.week6_defense import (
    CHALLENGE_RECON_PATTERNS,
    HARD_STRUCTURED_PATTERNS,
    IMPUTATION_STRATEGIES,
    evaluate_week6_pattern,
    selected_patterns,
)
from scripts.generate_reviewer_defense_configs import build_experiment_plan

DEFAULT_METHODS = (
    "M0_full_no_masking",
    "M1_random_dropout",
    "M2_structured_masking",
    "M6_structured_plus_availability_plus_subclass",
)
DEFAULT_SEEDS = (7, 42, 123, 2024, 2025)
DEFAULT_PATTERNS = ("full", *HARD_STRUCTURED_PATTERNS, *CHALLENGE_RECON_PATTERNS)
DEFAULT_BACKBONE = "xresnet1d101_like"


def reviewer_run_id(method_id: str, backbone: str, seed: int) -> str:
    return f"{method_id}_{backbone}_seed{int(seed)}"


def csv_safe(value: str) -> str:
    return value.replace(" / ", "_").replace("-", "_").replace(" ", "_")


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


def plan_rows(*, methods: Sequence[str], seeds: Sequence[int], backbone: str) -> list[dict[str, Any]]:
    wanted = {(method, int(seed)) for method in methods for seed in seeds}
    rows: list[dict[str, Any]] = []
    for row in build_experiment_plan():
        if row["group"] != "primary":
            continue
        if row["backbone"] != backbone:
            continue
        key = (str(row["method_id"]), int(row["seed"]))
        if key in wanted:
            rows.append(dict(row))
    rows.sort(key=lambda row: (str(row["method_id"]), int(row["seed"])))
    found = {(str(row["method_id"]), int(row["seed"])) for row in rows}
    missing = sorted(wanted.difference(found))
    if missing:
        raise RuntimeError(f"Missing reviewer-defense config rows for: {missing}")
    return rows


def make_method_run(row: Mapping[str, Any]) -> MethodRun:
    output_dir = Path(str(row["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    run = MethodRun(
        method_id=str(row["method_id"]),
        seed=int(row["seed"]),
        output_dir=output_dir,
        checkpoint_path=output_dir / "best_model.pt",
        config_path=output_dir / "config_used.yaml",
        thresholds_path=output_dir / "thresholds_val.json",
    )
    missing = [path for path in (run.checkpoint_path, run.config_path, run.thresholds_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts for {row}: {[str(path) for path in missing]}")
    for path in (run.output_dir, run.config_path, run.checkpoint_path, run.thresholds_path):
        text = str(path)
        if "records500" in text or "filename_hr" in text:
            raise RuntimeError(f"Forbidden records500/filename_hr reference in {path}")
    config_text = run.config_path.read_text(encoding="utf-8")
    if "records500" in config_text or "filename_hr" in config_text:
        raise RuntimeError(f"Forbidden records500/filename_hr reference in {run.config_path}")
    return run


def audit_prediction_csv(path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(path, nrows=256)
    required = {
        "patient_id",
        "split",
        "strat_fold",
        "threshold_source_split",
        "logit_NORM",
        "prob_NORM",
        "y_true_NORM",
        "pred_NORM",
        "threshold_NORM",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing prediction columns: {sorted(missing)}")
    if set(frame["split"].dropna().unique()) != {"test"}:
        raise RuntimeError(f"{path} contains non-test predictions")
    if set(frame["strat_fold"].dropna().astype(int).unique()) != {10}:
        raise RuntimeError(f"{path} contains non-fold-10 predictions")
    if set(str(x) for x in frame["threshold_source_split"].dropna().unique()) != {"val"}:
        raise RuntimeError(f"{path} does not use validation thresholds")


def mean_sd_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import pandas as pd

    if not rows:
        return []
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    metrics = ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll")
    for keys, group in frame.groupby(["method_id", "backbone", "pattern", "fill_mode"], dropna=False):
        method_id, backbone, pattern, fill_mode = keys
        item: dict[str, Any] = {
            "method_id": method_id,
            "backbone": backbone,
            "pattern": pattern,
            "fill_mode": fill_mode,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return sorted(out, key=lambda r: (r["method_id"], r["backbone"], r["pattern"], r["fill_mode"]))


def delta_vs_mean_fill(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    by_key = {
        (row["method_id"], row["backbone"], int(row["seed"]), row["pattern"], row["fill_mode"]): row
        for row in rows
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["fill_mode"] == "mean_fill":
            continue
        base = by_key.get((row["method_id"], row["backbone"], int(row["seed"]), row["pattern"], "mean_fill"))
        if base is None:
            continue
        out.append(
            {
                "method_id": row["method_id"],
                "backbone": row["backbone"],
                "seed": int(row["seed"]),
                "pattern": row["pattern"],
                "fill_mode": row["fill_mode"],
                "baseline_fill_mode": "mean_fill",
                "delta_macro_auprc": float(row["macro_auprc"]) - float(base["macro_auprc"]),
                "delta_macro_auroc": float(row["macro_auroc"]) - float(base["macro_auroc"]),
                "delta_macro_f1": float(row["macro_f1"]) - float(base["macro_f1"]),
                "delta_bce_nll": float(row["bce_nll"]) - float(base["bce_nll"]),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    predictions_dir = output_dir / "imputation_predictions"
    metric_rows_dir = output_dir / "metric_rows"
    rows = plan_rows(methods=args.methods, seeds=args.seeds, backbone=args.backbone)
    patterns = selected_patterns(args.patterns)
    if args.dry_run:
        result = {
            "status": "dry_run",
            "n_runs": len(rows),
            "n_expected_prediction_csv": len(rows) * len(args.strategies) * len(args.patterns),
            "methods": list(args.methods),
            "seeds": [int(seed) for seed in args.seeds],
            "backbone": args.backbone,
            "strategies": list(args.strategies),
            "patterns": list(args.patterns),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    prediction_files: list[dict[str, Any]] = []
    for row in rows:
        run = make_method_run(row)
        method_run_id = reviewer_run_id(str(row["method_id"]), str(row["backbone"]), int(row["seed"]))
        run_rows: list[dict[str, Any]] = []
        for strategy in args.strategies:
            for pattern_name, pattern in patterns.items():
                metric_row, pred_info = evaluate_week6_pattern(
                    run=run,
                    pattern_name=pattern_name,
                    pattern=pattern,
                    imputation_strategy=strategy,
                    split="test",
                    smoke_test=args.smoke_test,
                    save_predictions=args.save_predictions,
                    predictions_dir=predictions_dir,
                    prediction_method_id=method_run_id,
                )
                metric_row.update(
                    {
                        "method_run_id": method_run_id,
                        "backbone": row["backbone"],
                        "group": "imputation_eval",
                        "source_output_dir": str(run.output_dir),
                        "prediction_saved": bool(pred_info),
                    }
                )
                all_rows.append(metric_row)
                run_rows.append(metric_row)
                if pred_info:
                    pred_path = Path(pred_info["csv_path"])
                    audit_prediction_csv(pred_path)
                    prediction_files.append(pred_info)
        write_csv(metric_rows_dir / f"{method_run_id}.csv", run_rows)

    if not args.per_run_only:
        write_csv(output_dir / "imputation_metric_rows.csv", all_rows)
        write_csv(output_dir / "imputation_mean_sd.csv", mean_sd_rows(all_rows))
        write_csv(output_dir / "imputation_delta_vs_mean_fill.csv", delta_vs_mean_fill(all_rows))
    metadata = {
        "status": "completed",
        "n_runs": len(rows),
        "n_metric_rows": len(all_rows),
        "n_prediction_files": len(prediction_files),
        "expected_prediction_files": len(rows) * len(args.strategies) * len(args.patterns) if args.save_predictions else 0,
        "methods": list(args.methods),
        "seeds": [int(seed) for seed in args.seeds],
        "backbone": args.backbone,
        "strategies": list(args.strategies),
        "patterns": list(args.patterns),
        "threshold_source_split": "val",
        "test_fold_only": True,
        "records500_used": False,
        "filename_hr_used": False,
        "prediction_files": prediction_files,
    }
    if args.per_run_only:
        run_ids = sorted({str(row["method_run_id"]) for row in all_rows})
        summary_name = "__".join(run_ids) if run_ids else "empty"
        write_json(output_dir / "run_summaries" / f"{summary_name}.json", metadata)
    else:
        write_json(output_dir / "imputation_eval_summary.json", metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "prediction_files"}, indent=2, ensure_ascii=False))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--strategies", nargs="+", default=list(IMPUTATION_STRATEGIES))
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_defense_20260701/strategy_benchmark"))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--per-run-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
