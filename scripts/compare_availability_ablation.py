#!/usr/bin/env python3
"""Compare full, random dropout, structured masking, and availability ablation."""

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


def build_rows(
    *,
    full_dir: Path,
    random_dir: Path,
    structured_dir: Path,
    availability_dir: Path,
    fill_mode: str,
) -> list[dict[str, object]]:
    full = pattern_metrics(full_dir, fill_mode)
    random = pattern_metrics(random_dir, fill_mode)
    structured = pattern_metrics(structured_dir, fill_mode)
    availability = pattern_metrics(availability_dir, fill_mode)
    rows = []
    for pattern in full:
        row: dict[str, object] = {"fill_mode": fill_mode, "pattern": pattern}
        stores = {
            "full": full[pattern],
            "random_dropout": random[pattern],
            "structured": structured[pattern],
            "availability": availability[pattern],
        }
        for metric in METRICS:
            for prefix, metrics in stores.items():
                row[f"{prefix}_{metric}"] = metrics.get(metric)
            row[f"availability_minus_structured_{metric}"] = (
                None
                if row[f"availability_{metric}"] is None or row[f"structured_{metric}"] is None
                else row[f"availability_{metric}"] - row[f"structured_{metric}"]
            )
            row[f"availability_minus_random_{metric}"] = (
                None
                if row[f"availability_{metric}"] is None or row[f"random_dropout_{metric}"] is None
                else row[f"availability_{metric}"] - row[f"random_dropout_{metric}"]
            )
        for label in LABELS:
            for prefix, metrics in stores.items():
                row[f"{prefix}_{label}_auprc"] = metrics["per_class_auprc"].get(label)
            row[f"availability_minus_structured_{label}_auprc"] = (
                None
                if row[f"availability_{label}_auprc"] is None or row[f"structured_{label}_auprc"] is None
                else row[f"availability_{label}_auprc"] - row[f"structured_{label}_auprc"]
            )
            row[f"availability_minus_random_{label}_auprc"] = (
                None
                if row[f"availability_{label}_auprc"] is None or row[f"random_dropout_{label}_auprc"] is None
                else row[f"availability_{label}_auprc"] - row[f"random_dropout_{label}_auprc"]
            )
        rows.append(row)
    return rows


def average_delta(rows: Iterable[dict[str, object]], patterns: Iterable[str], delta_key: str) -> float:
    pattern_set = set(patterns)
    values = [float(row[delta_key]) for row in rows if row["pattern"] in pattern_set and row.get(delta_key) is not None]
    return sum(values) / len(values)


def write_markdown(path: Path, summary: dict, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Availability Embedding Ablation",
        "",
        f"Full directory: `{summary['full_dir']}`",
        f"Random dropout directory: `{summary['random_dir']}`",
        f"Structured directory: `{summary['structured_dir']}`",
        f"Availability directory: `{summary['availability_dir']}`",
        "",
        "## Summary",
        "",
        f"Full-pattern mean-fill Macro AUPRC delta vs structured: `{summary['full_pattern_mean_fill_delta_vs_structured_macro_auprc']:.6f}`",
        f"Full-pattern mean-fill Macro AUPRC delta vs random: `{summary['full_pattern_mean_fill_delta_vs_random_macro_auprc']:.6f}`",
        f"Structured hard average delta vs structured: `{summary['structured_hard_mean_fill_avg_delta_vs_structured_macro_auprc']:.6f}`",
        f"Hard overall average delta vs structured: `{summary['hard_overall_mean_fill_avg_delta_vs_structured_macro_auprc']:.6f}`",
        f"HYP full-pattern delta vs structured: `{summary['hyp_full_pattern_delta_vs_structured_auprc']:.6f}`",
        "",
        "## Pattern Deltas",
        "",
        "| Fill | Pattern | Avail-Structured AUROC | Avail-Structured AUPRC | Avail-Structured F1 | Avail-Random AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fill_mode']} | {row['pattern']} | "
            f"{row['availability_minus_structured_macro_auroc']:.6f} | "
            f"{row['availability_minus_structured_macro_auprc']:.6f} | "
            f"{row['availability_minus_structured_macro_f1']:.6f} | "
            f"{row['availability_minus_random_macro_auprc']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare availability embedding ablation with prior baselines.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--availability-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fill_mode in ("zero_fill", "mean_fill"):
        rows.extend(
            build_rows(
                full_dir=args.full_dir,
                random_dir=args.random_dir,
                structured_dir=args.structured_dir,
                availability_dir=args.availability_dir,
                fill_mode=fill_mode,
            )
        )

    mean_fill_rows = [row for row in rows if row["fill_mode"] == "mean_fill"]
    full_row = next(row for row in mean_fill_rows if row["pattern"] == "full")
    structured_hard_delta = average_delta(
        mean_fill_rows,
        STRUCTURED_HARD_PATTERNS,
        "availability_minus_structured_macro_auprc",
    )
    hard_overall_delta = average_delta(
        mean_fill_rows,
        HARD_OVERALL_PATTERNS,
        "availability_minus_structured_macro_auprc",
    )
    summary = {
        "full_dir": str(args.full_dir),
        "random_dir": str(args.random_dir),
        "structured_dir": str(args.structured_dir),
        "availability_dir": str(args.availability_dir),
        "full_pattern_mean_fill_delta_vs_structured_macro_auprc": float(
            full_row["availability_minus_structured_macro_auprc"]
        ),
        "full_pattern_mean_fill_delta_vs_random_macro_auprc": float(
            full_row["availability_minus_random_macro_auprc"]
        ),
        "full_pattern_preserves_structured_performance": float(
            full_row["availability_minus_structured_macro_auprc"]
        )
        >= -0.005,
        "structured_hard_patterns": list(STRUCTURED_HARD_PATTERNS),
        "structured_hard_mean_fill_avg_delta_vs_structured_macro_auprc": structured_hard_delta,
        "structured_hard_mean_fill_improved_vs_structured": structured_hard_delta > 0,
        "hard_overall_patterns": list(HARD_OVERALL_PATTERNS),
        "hard_overall_mean_fill_avg_delta_vs_structured_macro_auprc": hard_overall_delta,
        "hard_overall_mean_fill_improved_vs_structured": hard_overall_delta > 0,
        "minority_full_pattern_mean_fill_delta_vs_structured_auprc": {
            label: float(full_row[f"availability_minus_structured_{label}_auprc"])
            for label in ("MI", "CD", "HYP")
        },
        "minority_full_pattern_mean_fill_delta_vs_random_auprc": {
            label: float(full_row[f"availability_minus_random_{label}_auprc"]) for label in ("MI", "CD", "HYP")
        },
        "hyp_full_pattern_delta_vs_structured_auprc": float(
            full_row["availability_minus_structured_HYP_auprc"]
        ),
        "hyp_full_pattern_recovered_vs_structured": float(
            full_row["availability_minus_structured_HYP_auprc"]
        )
        > 0,
    }

    csv_path = args.out_dir / "compare_availability_ablation.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out_dir / "compare_availability_ablation.json"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.out_dir / "compare_availability_ablation.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
