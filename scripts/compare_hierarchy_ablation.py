#!/usr/bin/env python3
"""Compare A4b hierarchy ablation with A4a and earlier baselines."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.evaluation.hierarchy_patterns import evaluate_hierarchy_patterns_in_memory

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
STRUCTURED_HARD_PATTERNS = (
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
HARD_OVERALL_PATTERNS = (
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
METRICS = ("macro_auroc", "macro_auprc", "macro_f1")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pattern_metrics(directory: Path, fill_mode: str) -> dict[str, dict]:
    data = read_json(directory / f"test_missing_patterns_{fill_mode}.json")
    return {name: item["metrics"] for name, item in data["patterns"].items()}


def pattern_diagnostics(directory: Path, fill_mode: str) -> dict[str, dict]:
    data = read_json(directory / f"test_missing_patterns_{fill_mode}.json")
    return {
        name: item.get("hierarchy_diagnostics", {})
        for name, item in data["patterns"].items()
    }


def load_config_from_dir(directory: Path) -> dict:
    path = directory / "config_used.yaml"
    if not path.exists():
        raise RuntimeError(f"Missing config_used.yaml in {directory}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def compute_subclass_aux_diagnostics(directory: Path, fill_mode: str) -> dict[str, dict]:
    config = load_config_from_dir(directory)
    result = evaluate_hierarchy_patterns_in_memory(
        checkpoint_path=directory / "best_model.pt",
        config=config,
        fill_mode=fill_mode,
        smoke_test=False,
    )
    return {
        name: item["hierarchy_diagnostics"]
        for name, item in result["details"]["patterns"].items()
    }


def build_rows(args, fill_mode: str, subclass_diag: dict[str, dict]) -> list[dict[str, object]]:
    stores = {
        "full": pattern_metrics(args.full_dir, fill_mode),
        "random_dropout": pattern_metrics(args.random_dir, fill_mode),
        "structured": pattern_metrics(args.structured_dir, fill_mode),
        "availability": pattern_metrics(args.availability_dir, fill_mode),
        "subclass_aux": pattern_metrics(args.subclass_dir, fill_mode),
        "hierarchy": pattern_metrics(args.hierarchy_dir, fill_mode),
    }
    hierarchy_diag = pattern_diagnostics(args.hierarchy_dir, fill_mode)
    rows = []
    for pattern in stores["full"]:
        row: dict[str, object] = {"fill_mode": fill_mode, "pattern": pattern}
        for metric in METRICS:
            for prefix, data in stores.items():
                row[f"{prefix}_{metric}"] = data[pattern].get(metric)
            for baseline in ("subclass_aux", "availability", "structured", "random_dropout"):
                row[f"hierarchy_minus_{baseline}_{metric}"] = (
                    None
                    if row[f"hierarchy_{metric}"] is None or row[f"{baseline}_{metric}"] is None
                    else row[f"hierarchy_{metric}"] - row[f"{baseline}_{metric}"]
                )
        for label in LABELS:
            for prefix, data in stores.items():
                row[f"{prefix}_{label}_auprc"] = data[pattern]["per_class_auprc"].get(label)
            for baseline in ("subclass_aux", "availability", "structured", "random_dropout"):
                row[f"hierarchy_minus_{baseline}_{label}_auprc"] = (
                    None
                    if row[f"hierarchy_{label}_auprc"] is None or row[f"{baseline}_{label}_auprc"] is None
                    else row[f"hierarchy_{label}_auprc"] - row[f"{baseline}_{label}_auprc"]
                )
        for key in ("hierarchy_loss", "violation_rate", "mean_violation_margin", "max_violation_margin"):
            row[f"subclass_aux_{key}"] = subclass_diag.get(pattern, {}).get(key)
            row[f"hierarchy_{key}"] = hierarchy_diag.get(pattern, {}).get(key)
            row[f"hierarchy_minus_subclass_aux_{key}"] = (
                None
                if row[f"subclass_aux_{key}"] is None or row[f"hierarchy_{key}"] is None
                else row[f"hierarchy_{key}"] - row[f"subclass_aux_{key}"]
            )
        rows.append(row)
    return rows


def average(rows: Iterable[dict[str, object]], patterns: Iterable[str], key: str) -> float:
    pattern_set = set(patterns)
    values = [float(row[key]) for row in rows if row["pattern"] in pattern_set and row.get(key) is not None]
    return sum(values) / len(values)


def write_markdown(path: Path, summary: dict, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Hierarchy Ablation",
        "",
        f"Hierarchy directory: `{summary['hierarchy_dir']}`",
        "",
        "## Summary",
        "",
        f"Full-pattern Macro AUPRC delta vs A4a: `{summary['full_pattern_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Hard structured average delta vs A4a: `{summary['hard_structured_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Hard overall average delta vs A4a: `{summary['hard_overall_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Full-pattern violation-rate delta vs A4a: `{summary['full_pattern_violation_rate_delta_vs_subclass_aux']:.6f}`",
        f"Hard overall mean-margin delta vs A4a: `{summary['hard_overall_mean_violation_margin_delta_vs_subclass_aux']:.6f}`",
        "",
        "| Fill | Pattern | A4b-A4a AUROC | A4b-A4a AUPRC | A4b-A4a F1 | A4b-A3 AUPRC | violation-rate delta | mean-margin delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fill_mode']} | {row['pattern']} | "
            f"{row['hierarchy_minus_subclass_aux_macro_auroc']:.6f} | "
            f"{row['hierarchy_minus_subclass_aux_macro_auprc']:.6f} | "
            f"{row['hierarchy_minus_subclass_aux_macro_f1']:.6f} | "
            f"{row['hierarchy_minus_availability_macro_auprc']:.6f} | "
            f"{row['hierarchy_minus_subclass_aux_violation_rate']:.6f} | "
            f"{row['hierarchy_minus_subclass_aux_mean_violation_margin']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hierarchy ablation.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--availability-dir", type=Path, required=True)
    parser.add_argument("--subclass-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fill_mode in ("zero_fill", "mean_fill"):
        subclass_diag = compute_subclass_aux_diagnostics(args.subclass_dir, fill_mode)
        rows.extend(build_rows(args, fill_mode, subclass_diag))
    mean_rows = [row for row in rows if row["fill_mode"] == "mean_fill"]
    full_row = next(row for row in mean_rows if row["pattern"] == "full")
    summary = {
        "full_dir": str(args.full_dir),
        "random_dir": str(args.random_dir),
        "structured_dir": str(args.structured_dir),
        "availability_dir": str(args.availability_dir),
        "subclass_dir": str(args.subclass_dir),
        "hierarchy_dir": str(args.hierarchy_dir),
        "full_pattern_delta_vs_subclass_aux_macro_auprc": float(full_row["hierarchy_minus_subclass_aux_macro_auprc"]),
        "full_pattern_delta_vs_availability_macro_auprc": float(full_row["hierarchy_minus_availability_macro_auprc"]),
        "full_pattern_delta_vs_structured_macro_auprc": float(full_row["hierarchy_minus_structured_macro_auprc"]),
        "full_pattern_delta_vs_random_macro_auprc": float(full_row["hierarchy_minus_random_dropout_macro_auprc"]),
        "hard_structured_delta_vs_subclass_aux_macro_auprc": average(mean_rows, STRUCTURED_HARD_PATTERNS, "hierarchy_minus_subclass_aux_macro_auprc"),
        "hard_overall_delta_vs_subclass_aux_macro_auprc": average(mean_rows, HARD_OVERALL_PATTERNS, "hierarchy_minus_subclass_aux_macro_auprc"),
        "hard_structured_delta_vs_availability_macro_auprc": average(mean_rows, STRUCTURED_HARD_PATTERNS, "hierarchy_minus_availability_macro_auprc"),
        "hard_overall_delta_vs_availability_macro_auprc": average(mean_rows, HARD_OVERALL_PATTERNS, "hierarchy_minus_availability_macro_auprc"),
        "minority_full_delta_vs_subclass_aux_auprc": {
            label: float(full_row[f"hierarchy_minus_subclass_aux_{label}_auprc"]) for label in ("MI", "CD", "HYP")
        },
        "minority_hard_overall_delta_vs_subclass_aux_auprc": {
            label: average(mean_rows, HARD_OVERALL_PATTERNS, f"hierarchy_minus_subclass_aux_{label}_auprc")
            for label in ("MI", "CD", "HYP")
        },
        "hyp_full_delta_vs_subclass_aux_auprc": float(full_row["hierarchy_minus_subclass_aux_HYP_auprc"]),
        "full_pattern_violation_rate_delta_vs_subclass_aux": float(full_row["hierarchy_minus_subclass_aux_violation_rate"]),
        "hard_structured_violation_rate_delta_vs_subclass_aux": average(mean_rows, STRUCTURED_HARD_PATTERNS, "hierarchy_minus_subclass_aux_violation_rate"),
        "hard_overall_violation_rate_delta_vs_subclass_aux": average(mean_rows, HARD_OVERALL_PATTERNS, "hierarchy_minus_subclass_aux_violation_rate"),
        "full_pattern_mean_violation_margin_delta_vs_subclass_aux": float(full_row["hierarchy_minus_subclass_aux_mean_violation_margin"]),
        "hard_structured_mean_violation_margin_delta_vs_subclass_aux": average(mean_rows, STRUCTURED_HARD_PATTERNS, "hierarchy_minus_subclass_aux_mean_violation_margin"),
        "hard_overall_mean_violation_margin_delta_vs_subclass_aux": average(mean_rows, HARD_OVERALL_PATTERNS, "hierarchy_minus_subclass_aux_mean_violation_margin"),
    }
    with (args.out_dir / "compare_hierarchy_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "compare_hierarchy_ablation.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "compare_hierarchy_ablation.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
