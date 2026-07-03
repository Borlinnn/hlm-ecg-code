#!/usr/bin/env python3
"""Compare full, random dropout, and structured masking baselines."""

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


def build_rows(full_dir: Path, random_dir: Path, structured_dir: Path, fill_mode: str) -> list[dict[str, object]]:
    full = pattern_metrics(full_dir, fill_mode)
    random = pattern_metrics(random_dir, fill_mode)
    structured = pattern_metrics(structured_dir, fill_mode)
    rows = []
    for pattern in full:
        row: dict[str, object] = {"fill_mode": fill_mode, "pattern": pattern}
        for metric in METRICS:
            full_value = full[pattern].get(metric)
            random_value = random[pattern].get(metric)
            structured_value = structured[pattern].get(metric)
            row[f"full_{metric}"] = full_value
            row[f"random_dropout_{metric}"] = random_value
            row[f"structured_{metric}"] = structured_value
            row[f"structured_minus_full_{metric}"] = (
                None if full_value is None or structured_value is None else structured_value - full_value
            )
            row[f"structured_minus_random_{metric}"] = (
                None if random_value is None or structured_value is None else structured_value - random_value
            )
        for label in LABELS:
            full_value = full[pattern]["per_class_auprc"].get(label)
            random_value = random[pattern]["per_class_auprc"].get(label)
            structured_value = structured[pattern]["per_class_auprc"].get(label)
            row[f"full_{label}_auprc"] = full_value
            row[f"random_dropout_{label}_auprc"] = random_value
            row[f"structured_{label}_auprc"] = structured_value
            row[f"structured_minus_random_{label}_auprc"] = (
                None if random_value is None or structured_value is None else structured_value - random_value
            )
        rows.append(row)
    return rows


def average_delta(rows: Iterable[dict[str, object]], patterns: Iterable[str], delta_key: str) -> float:
    pattern_set = set(patterns)
    values = [float(row[delta_key]) for row in rows if row["pattern"] in pattern_set and row.get(delta_key) is not None]
    return sum(values) / len(values)


def write_markdown(path: Path, summary: dict, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Full vs Random Dropout vs Structured Masking",
        "",
        f"Full directory: `{summary['full_dir']}`",
        f"Random dropout directory: `{summary['random_dir']}`",
        f"Structured directory: `{summary['structured_dir']}`",
        "",
        "## Summary",
        "",
        f"Structured full-pattern mean-fill Macro AUPRC delta vs random: `{summary['full_pattern_mean_fill_delta_vs_random_macro_auprc']:.6f}`",
        f"Structured hard average Macro AUPRC delta vs random: `{summary['structured_hard_mean_fill_avg_delta_vs_random_macro_auprc']:.6f}`",
        f"Hard overall Macro AUPRC delta vs random: `{summary['hard_overall_mean_fill_avg_delta_vs_random_macro_auprc']:.6f}`",
        f"MI/CD/HYP full-pattern mean-fill AUPRC deltas vs random: `{summary['minority_full_pattern_mean_fill_delta_vs_random_auprc']}`",
        "",
        "## Pattern Deltas",
        "",
        "| Fill | Pattern | Structured-Random AUROC | Structured-Random AUPRC | Structured-Random F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fill_mode']} | {row['pattern']} | "
            f"{row['structured_minus_random_macro_auroc']:.6f} | "
            f"{row['structured_minus_random_macro_auprc']:.6f} | "
            f"{row['structured_minus_random_macro_f1']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full, random dropout, and structured masking baselines.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fill_mode in ("zero_fill", "mean_fill"):
        rows.extend(build_rows(args.full_dir, args.random_dir, args.structured_dir, fill_mode))

    mean_fill_rows = [row for row in rows if row["fill_mode"] == "mean_fill"]
    full_row = next(row for row in mean_fill_rows if row["pattern"] == "full")
    structured_hard_delta = average_delta(
        mean_fill_rows,
        STRUCTURED_HARD_PATTERNS,
        "structured_minus_random_macro_auprc",
    )
    hard_overall_delta = average_delta(
        mean_fill_rows,
        HARD_OVERALL_PATTERNS,
        "structured_minus_random_macro_auprc",
    )
    summary = {
        "full_dir": str(args.full_dir),
        "random_dir": str(args.random_dir),
        "structured_dir": str(args.structured_dir),
        "full_pattern_mean_fill_delta_vs_random_macro_auprc": float(
            full_row["structured_minus_random_macro_auprc"]
        ),
        "full_pattern_preserves_random_dropout_performance": float(
            full_row["structured_minus_random_macro_auprc"]
        )
        >= -0.01,
        "structured_hard_patterns": list(STRUCTURED_HARD_PATTERNS),
        "structured_hard_mean_fill_avg_delta_vs_random_macro_auprc": structured_hard_delta,
        "structured_hard_mean_fill_improved_vs_random": structured_hard_delta > 0,
        "hard_overall_patterns": list(HARD_OVERALL_PATTERNS),
        "hard_overall_mean_fill_avg_delta_vs_random_macro_auprc": hard_overall_delta,
        "hard_overall_mean_fill_improved_vs_random": hard_overall_delta > 0,
        "minority_full_pattern_mean_fill_delta_vs_random_auprc": {
            label: float(full_row[f"structured_minus_random_{label}_auprc"]) for label in ("MI", "CD", "HYP")
        },
    }

    csv_path = args.out_dir / "compare_full_random_structured.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out_dir / "compare_full_random_structured.json"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.out_dir / "compare_full_random_structured.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
