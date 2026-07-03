#!/usr/bin/env python3
"""Compare full-lead baseline and random lead dropout baseline outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
HARD_PATTERNS = (
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
METRICS = ("macro_auroc", "macro_auprc", "macro_f1")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pattern_metrics(directory: Path, fill_mode: str) -> Dict[str, dict]:
    data = read_json(directory / f"test_missing_patterns_{fill_mode}.json")
    return {name: item["metrics"] for name, item in data["patterns"].items()}


def metric_delta_rows(full_dir: Path, random_dir: Path, fill_mode: str) -> list[dict[str, object]]:
    full = pattern_metrics(full_dir, fill_mode)
    random = pattern_metrics(random_dir, fill_mode)
    rows = []
    for pattern in full:
        row: dict[str, object] = {"fill_mode": fill_mode, "pattern": pattern}
        for metric in METRICS:
            full_value = full[pattern].get(metric)
            random_value = random[pattern].get(metric)
            row[f"full_{metric}"] = full_value
            row[f"random_dropout_{metric}"] = random_value
            row[f"delta_{metric}"] = None if full_value is None or random_value is None else random_value - full_value
        for label in LABELS:
            full_value = full[pattern]["per_class_auprc"].get(label)
            random_value = random[pattern]["per_class_auprc"].get(label)
            row[f"full_{label}_auprc"] = full_value
            row[f"random_dropout_{label}_auprc"] = random_value
            row[f"delta_{label}_auprc"] = None if full_value is None or random_value is None else random_value - full_value
        rows.append(row)
    return rows


def average_delta(rows: Iterable[dict[str, object]], patterns: Iterable[str], metric: str) -> float:
    selected = [row for row in rows if row["pattern"] in set(patterns)]
    values = [float(row[f"delta_{metric}"]) for row in selected if row.get(f"delta_{metric}") is not None]
    return sum(values) / len(values)


def write_markdown(path: Path, summary: dict, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Full Baseline vs Random Lead Dropout",
        "",
        f"Full directory: `{summary['full_dir']}`",
        f"Random dropout directory: `{summary['random_dir']}`",
        "",
        "## Summary",
        "",
        f"Full-pattern mean-fill Macro AUPRC delta: `{summary['full_pattern_mean_fill_delta_macro_auprc']:.6f}`",
        f"Hard missing mean-fill Macro AUPRC average delta: `{summary['hard_missing_mean_fill_avg_delta_macro_auprc']:.6f}`",
        f"MI/CD/HYP full-pattern mean-fill AUPRC deltas: `{summary['minority_full_pattern_mean_fill_delta_auprc']}`",
        "",
        "## Pattern Deltas",
        "",
        "| Fill | Pattern | Delta AUROC | Delta AUPRC | Delta F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fill_mode']} | {row['pattern']} | "
            f"{row['delta_macro_auroc']:.6f} | {row['delta_macro_auprc']:.6f} | {row['delta_macro_f1']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full baseline with random lead dropout baseline.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fill_mode in ("zero_fill", "mean_fill"):
        rows.extend(metric_delta_rows(args.full_dir, args.random_dir, fill_mode))

    mean_fill_rows = [row for row in rows if row["fill_mode"] == "mean_fill"]
    full_row = next(row for row in mean_fill_rows if row["pattern"] == "full")
    summary = {
        "full_dir": str(args.full_dir),
        "random_dir": str(args.random_dir),
        "full_pattern_mean_fill_delta_macro_auprc": float(full_row["delta_macro_auprc"]),
        "full_pattern_mean_fill_preserves_performance": float(full_row["delta_macro_auprc"]) >= -0.02,
        "hard_patterns": list(HARD_PATTERNS),
        "hard_missing_mean_fill_avg_delta_macro_auprc": average_delta(mean_fill_rows, HARD_PATTERNS, "macro_auprc"),
        "hard_missing_mean_fill_improved": average_delta(mean_fill_rows, HARD_PATTERNS, "macro_auprc") > 0,
        "minority_full_pattern_mean_fill_delta_auprc": {
            label: float(full_row[f"delta_{label}_auprc"]) for label in ("MI", "CD", "HYP")
        },
    }

    csv_path = args.out_dir / "compare_full_vs_random_dropout.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out_dir / "compare_full_vs_random_dropout.json"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.out_dir / "compare_full_vs_random_dropout.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
