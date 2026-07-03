#!/usr/bin/env python3
"""Compare A4a subclass auxiliary ablation with earlier baselines."""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

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


def build_rows(args, fill_mode: str) -> list[dict[str, object]]:
    stores = {
        "full": pattern_metrics(args.full_dir, fill_mode),
        "random_dropout": pattern_metrics(args.random_dir, fill_mode),
        "structured": pattern_metrics(args.structured_dir, fill_mode),
        "availability": pattern_metrics(args.availability_dir, fill_mode),
        "subclass_aux": pattern_metrics(args.subclass_dir, fill_mode),
    }
    rows = []
    for pattern in stores["full"]:
        row: dict[str, object] = {"fill_mode": fill_mode, "pattern": pattern}
        for metric in METRICS:
            for prefix, data in stores.items():
                row[f"{prefix}_{metric}"] = data[pattern].get(metric)
            for baseline in ("availability", "structured", "random_dropout"):
                row[f"subclass_aux_minus_{baseline}_{metric}"] = (
                    None
                    if row[f"subclass_aux_{metric}"] is None or row[f"{baseline}_{metric}"] is None
                    else row[f"subclass_aux_{metric}"] - row[f"{baseline}_{metric}"]
                )
        for label in LABELS:
            for prefix, data in stores.items():
                row[f"{prefix}_{label}_auprc"] = data[pattern]["per_class_auprc"].get(label)
            for baseline in ("availability", "structured", "random_dropout"):
                row[f"subclass_aux_minus_{baseline}_{label}_auprc"] = (
                    None
                    if row[f"subclass_aux_{label}_auprc"] is None or row[f"{baseline}_{label}_auprc"] is None
                    else row[f"subclass_aux_{label}_auprc"] - row[f"{baseline}_{label}_auprc"]
                )
        rows.append(row)
    return rows


def average_delta(rows: Iterable[dict[str, object]], patterns: Iterable[str], key: str) -> float:
    pattern_set = set(patterns)
    values = [float(row[key]) for row in rows if row["pattern"] in pattern_set and row.get(key) is not None]
    return sum(values) / len(values)


def write_markdown(path: Path, summary: dict, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Subclass Auxiliary Ablation",
        "",
        f"Subclass directory: `{summary['subclass_dir']}`",
        "",
        "## Summary",
        "",
        f"Full-pattern Macro AUPRC delta vs A3: `{summary['full_pattern_delta_vs_availability_macro_auprc']:.6f}`",
        f"Hard structured average delta vs A3: `{summary['hard_structured_delta_vs_availability_macro_auprc']:.6f}`",
        f"Hard overall average delta vs A3: `{summary['hard_overall_delta_vs_availability_macro_auprc']:.6f}`",
        f"HYP full-pattern AUPRC delta vs A3: `{summary['hyp_full_delta_vs_availability_auprc']:.6f}`",
        "",
        "| Fill | Pattern | A4a-A3 AUROC | A4a-A3 AUPRC | A4a-A3 F1 | A4a-A2 AUPRC | A4a-A1 AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fill_mode']} | {row['pattern']} | "
            f"{row['subclass_aux_minus_availability_macro_auroc']:.6f} | "
            f"{row['subclass_aux_minus_availability_macro_auprc']:.6f} | "
            f"{row['subclass_aux_minus_availability_macro_f1']:.6f} | "
            f"{row['subclass_aux_minus_structured_macro_auprc']:.6f} | "
            f"{row['subclass_aux_minus_random_dropout_macro_auprc']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare subclass auxiliary ablation.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--availability-dir", type=Path, required=True)
    parser.add_argument("--subclass-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fill_mode in ("zero_fill", "mean_fill"):
        rows.extend(build_rows(args, fill_mode))
    mean_rows = [row for row in rows if row["fill_mode"] == "mean_fill"]
    full_row = next(row for row in mean_rows if row["pattern"] == "full")
    summary = {
        "full_dir": str(args.full_dir),
        "random_dir": str(args.random_dir),
        "structured_dir": str(args.structured_dir),
        "availability_dir": str(args.availability_dir),
        "subclass_dir": str(args.subclass_dir),
        "full_pattern_delta_vs_availability_macro_auprc": float(full_row["subclass_aux_minus_availability_macro_auprc"]),
        "full_pattern_delta_vs_structured_macro_auprc": float(full_row["subclass_aux_minus_structured_macro_auprc"]),
        "full_pattern_delta_vs_random_macro_auprc": float(full_row["subclass_aux_minus_random_dropout_macro_auprc"]),
        "hard_structured_delta_vs_availability_macro_auprc": average_delta(mean_rows, STRUCTURED_HARD_PATTERNS, "subclass_aux_minus_availability_macro_auprc"),
        "hard_overall_delta_vs_availability_macro_auprc": average_delta(mean_rows, HARD_OVERALL_PATTERNS, "subclass_aux_minus_availability_macro_auprc"),
        "hard_structured_delta_vs_structured_macro_auprc": average_delta(mean_rows, STRUCTURED_HARD_PATTERNS, "subclass_aux_minus_structured_macro_auprc"),
        "hard_overall_delta_vs_structured_macro_auprc": average_delta(mean_rows, HARD_OVERALL_PATTERNS, "subclass_aux_minus_structured_macro_auprc"),
        "minority_full_delta_vs_availability_auprc": {
            label: float(full_row[f"subclass_aux_minus_availability_{label}_auprc"]) for label in ("MI", "CD", "HYP")
        },
        "hyp_full_delta_vs_availability_auprc": float(full_row["subclass_aux_minus_availability_HYP_auprc"]),
    }
    with (args.out_dir / "compare_subclass_auxiliary_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "compare_subclass_auxiliary_ablation.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "compare_subclass_auxiliary_ablation.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
