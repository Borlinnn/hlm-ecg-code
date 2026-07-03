#!/usr/bin/env python3
"""Compare A5 confidence consistency ablation with previous baselines."""

import argparse
import csv
import json
from pathlib import Path

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
        "hierarchy": pattern_metrics(args.hierarchy_dir, fill_mode),
        "consistency": pattern_metrics(args.consistency_dir, fill_mode),
    }
    rows = []
    for pattern in stores["full"]:
        row: dict[str, object] = {"fill_mode": fill_mode, "pattern": pattern}
        for metric in METRICS:
            for prefix, data in stores.items():
                row[f"{prefix}_{metric}"] = data[pattern].get(metric)
            for baseline in ("subclass_aux", "hierarchy", "availability", "structured", "random_dropout"):
                row[f"consistency_minus_{baseline}_{metric}"] = (
                    None
                    if row[f"consistency_{metric}"] is None or row[f"{baseline}_{metric}"] is None
                    else row[f"consistency_{metric}"] - row[f"{baseline}_{metric}"]
                )
        for label in LABELS:
            for prefix, data in stores.items():
                row[f"{prefix}_{label}_auprc"] = data[pattern]["per_class_auprc"].get(label)
            for baseline in ("subclass_aux", "hierarchy", "availability", "structured", "random_dropout"):
                row[f"consistency_minus_{baseline}_{label}_auprc"] = (
                    None
                    if row[f"consistency_{label}_auprc"] is None or row[f"{baseline}_{label}_auprc"] is None
                    else row[f"consistency_{label}_auprc"] - row[f"{baseline}_{label}_auprc"]
                )
        rows.append(row)
    return rows


def average(rows: list[dict[str, object]], patterns: tuple[str, ...], key: str) -> float:
    values = [float(row[key]) for row in rows if row["pattern"] in set(patterns) and row.get(key) is not None]
    return sum(values) / len(values)


def read_consistency_diagnostics(directory: Path) -> dict[str, object]:
    path = directory / "train_log.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get("val_macro_auprc") or -1.0))
    last = rows[-1]
    keys = [
        "train_cw_consistency_loss",
        "train_mean_consistency_weight",
        "train_min_consistency_weight",
        "train_max_consistency_weight",
        "train_full_mean_confidence",
        "train_masked_mean_confidence",
    ]
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_macro_auprc": float(best["val_macro_auprc"]),
        "best_epoch_train_diagnostics": {
            key: float(best[key]) for key in keys if best.get(key) not in (None, "")
        },
        "last_epoch": int(last["epoch"]),
        "last_epoch_train_diagnostics": {
            key: float(last[key]) for key in keys if last.get(key) not in (None, "")
        },
    }


def write_markdown(path: Path, summary: dict, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Confidence Consistency Ablation",
        "",
        f"Consistency directory: `{summary['consistency_dir']}`",
        "",
        "## Summary",
        "",
        f"Full-pattern Macro AUPRC delta vs A4a: `{summary['full_pattern_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Hard structured average delta vs A4a: `{summary['hard_structured_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Hard overall average delta vs A4a: `{summary['hard_overall_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Random-6 Macro AUPRC delta vs A4a: `{summary['random_6_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        "",
        "| Fill | Pattern | A5-A4a AUROC | A5-A4a AUPRC | A5-A4a F1 | A5-A4b AUPRC | A5-A1 AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fill_mode']} | {row['pattern']} | "
            f"{row['consistency_minus_subclass_aux_macro_auroc']:.6f} | "
            f"{row['consistency_minus_subclass_aux_macro_auprc']:.6f} | "
            f"{row['consistency_minus_subclass_aux_macro_f1']:.6f} | "
            f"{row['consistency_minus_hierarchy_macro_auprc']:.6f} | "
            f"{row['consistency_minus_random_dropout_macro_auprc']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare confidence consistency ablation.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--availability-dir", type=Path, required=True)
    parser.add_argument("--subclass-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--consistency-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fill_mode in ("zero_fill", "mean_fill"):
        rows.extend(build_rows(args, fill_mode))
    mean_rows = [row for row in rows if row["fill_mode"] == "mean_fill"]
    full_row = next(row for row in mean_rows if row["pattern"] == "full")
    random6 = next(row for row in mean_rows if row["pattern"] == "random-6")
    summary = {
        "full_dir": str(args.full_dir),
        "random_dir": str(args.random_dir),
        "structured_dir": str(args.structured_dir),
        "availability_dir": str(args.availability_dir),
        "subclass_dir": str(args.subclass_dir),
        "hierarchy_dir": str(args.hierarchy_dir),
        "consistency_dir": str(args.consistency_dir),
        "full_pattern_delta_vs_subclass_aux_macro_auprc": float(full_row["consistency_minus_subclass_aux_macro_auprc"]),
        "full_pattern_delta_vs_hierarchy_macro_auprc": float(full_row["consistency_minus_hierarchy_macro_auprc"]),
        "full_pattern_delta_vs_availability_macro_auprc": float(full_row["consistency_minus_availability_macro_auprc"]),
        "full_pattern_delta_vs_structured_macro_auprc": float(full_row["consistency_minus_structured_macro_auprc"]),
        "full_pattern_delta_vs_random_macro_auprc": float(full_row["consistency_minus_random_dropout_macro_auprc"]),
        "random_6_delta_vs_subclass_aux_macro_auprc": float(random6["consistency_minus_subclass_aux_macro_auprc"]),
        "hard_structured_delta_vs_subclass_aux_macro_auprc": average(mean_rows, STRUCTURED_HARD_PATTERNS, "consistency_minus_subclass_aux_macro_auprc"),
        "hard_overall_delta_vs_subclass_aux_macro_auprc": average(mean_rows, HARD_OVERALL_PATTERNS, "consistency_minus_subclass_aux_macro_auprc"),
        "hard_structured_delta_vs_hierarchy_macro_auprc": average(mean_rows, STRUCTURED_HARD_PATTERNS, "consistency_minus_hierarchy_macro_auprc"),
        "hard_overall_delta_vs_hierarchy_macro_auprc": average(mean_rows, HARD_OVERALL_PATTERNS, "consistency_minus_hierarchy_macro_auprc"),
        "hard_structured_delta_vs_random_macro_auprc": average(mean_rows, STRUCTURED_HARD_PATTERNS, "consistency_minus_random_dropout_macro_auprc"),
        "hard_overall_delta_vs_random_macro_auprc": average(mean_rows, HARD_OVERALL_PATTERNS, "consistency_minus_random_dropout_macro_auprc"),
        "minority_full_delta_vs_subclass_aux_auprc": {
            label: float(full_row[f"consistency_minus_subclass_aux_{label}_auprc"]) for label in ("MI", "CD", "HYP")
        },
        "minority_hard_overall_delta_vs_subclass_aux_auprc": {
            label: average(mean_rows, HARD_OVERALL_PATTERNS, f"consistency_minus_subclass_aux_{label}_auprc")
            for label in ("MI", "CD", "HYP")
        },
        "hyp_full_delta_vs_subclass_aux_auprc": float(full_row["consistency_minus_subclass_aux_HYP_auprc"]),
        "consistency_diagnostics": read_consistency_diagnostics(args.consistency_dir),
    }
    with (args.out_dir / "compare_confidence_consistency_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "compare_confidence_consistency_ablation.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "compare_confidence_consistency_ablation.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
