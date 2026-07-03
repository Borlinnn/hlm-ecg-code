#!/usr/bin/env python3
"""Summarize Week 6 fixed-pattern specialist baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from hlm_ecg.evaluation.supplemental_analysis import markdown_report, write_csv, write_json, write_markdown_table
from hlm_ecg.evaluation.week6_defense import ROOT, WEEK6_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Week 6 specialist baselines.")
    parser.add_argument("--output-dir", type=Path, default=WEEK6_DIR / "fixed_pattern_specialists")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _baseline_lookup(pattern: str, method_id: str) -> dict[str, float] | None:
    if pattern.startswith("challenge_"):
        path = ROOT / "outputs/week5_bibm_stabilization/challenge_reduced_leads/challenge_reduced_lead_metrics.csv"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["method_id"] == method_id and row["seed"] == "42" and row["pattern"] == pattern:
                    return {key: float(row[key]) for key in ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll")}
    else:
        path = ROOT / "outputs/week3_results_lock/all_methods_all_patterns_mean_fill.csv"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["method_id"] == method_id and row["pattern"] == pattern:
                    return {key: float(row[key]) for key in ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll")}
    return None


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows: list[dict[str, Any]] = []
    for config_path in sorted(output_dir.glob("*/seed*/config_used.yaml")):
        run_dir = config_path.parent
        config = _load_yaml(config_path)
        metrics = _load_json(run_dir / "test_full_metrics.json")
        gate = _load_json(run_dir / "specialist_training_gate_status.json") or {}
        pattern = str((config or {}).get("week6_specialist", {}).get("pattern", "unknown"))
        seed = int((config or {}).get("seed", 42))
        row: dict[str, Any] = {
            "specialist_id": run_dir.parent.name,
            "seed": seed,
            "pattern": pattern,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "trained": bool(metrics is not None and (run_dir / "best_model.pt").exists()),
            "gate_allowed": bool(gate.get("specialist_training_allowed", False)),
            "records500_used": False,
        }
        if metrics:
            for metric in ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll"):
                row[f"specialist_{metric}"] = metrics.get(metric)
            for baseline_method in ("A1_random_dropout", "A4a_subclass_auxiliary"):
                baseline = _baseline_lookup(pattern, baseline_method)
                if baseline:
                    for metric in ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll"):
                        row[f"{baseline_method}_{metric}"] = baseline[metric]
                        row[f"delta_specialist_vs_{baseline_method}_{metric}"] = float(metrics[metric]) - float(baseline[metric])
        rows.append(row)

    columns = [
        "specialist_id",
        "seed",
        "pattern",
        "trained",
        "gate_allowed",
        "specialist_macro_auprc",
        "A1_random_dropout_macro_auprc",
        "delta_specialist_vs_A1_random_dropout_macro_auprc",
        "A4a_subclass_auxiliary_macro_auprc",
        "delta_specialist_vs_A4a_subclass_auxiliary_macro_auprc",
        "specialist_macro_auroc",
        "specialist_macro_f1",
        "specialist_bce_nll",
        "run_dir",
        "records500_used",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "specialist_results_summary.csv", rows, columns)
    write_json(
        output_dir / "specialist_results_summary.json",
        {
            "records500_used": False,
            "note": "Specialist metrics are fixed-pattern baselines trained only when explicitly gated.",
            "rows": rows,
        },
    )
    write_markdown_table(output_dir / "specialist_results_summary.md", rows, columns)
    trained = [row for row in rows if row.get("trained")]
    lines = [
        f"- Specialist runs discovered: `{len(rows)}`.",
        f"- Specialist runs trained: `{len(trained)}`.",
        "- Comparisons use seed42 A1/A4a metrics from Week 3/5 locked outputs.",
        "- These specialists are reviewer-defense baselines, not replacements for A4a.",
    ]
    for row in trained:
        delta_a4a = row.get("delta_specialist_vs_A4a_subclass_auxiliary_macro_auprc")
        lines.append(
            f"- `{row['pattern']}` specialist Macro AUPRC={float(row['specialist_macro_auprc']):.4f}; "
            f"delta vs A4a={float(delta_a4a):+.4f}" if delta_a4a not in (None, "") else f"- `{row['pattern']}` specialist trained."
        )
    markdown_report(output_dir / "specialist_results_report.md", "Week 6 Specialist Baseline Summary", lines)


if __name__ == "__main__":
    main()
