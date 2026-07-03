#!/usr/bin/env python3
"""Compare A5-lite confidence consistency with prior HLM-ECG ablations."""

import argparse
import csv
import json
from pathlib import Path

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
MINORITY_LABELS = ("MI", "CD", "HYP")
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
METHOD_DIR_ARGS = (
    ("full", "full_dir"),
    ("random_dropout", "random_dir"),
    ("structured", "structured_dir"),
    ("availability", "availability_dir"),
    ("subclass_aux", "subclass_dir"),
    ("hierarchy", "hierarchy_dir"),
    ("consistency", "consistency_dir"),
    ("consistency_lite", "consistency_lite_dir"),
)
BASELINES = (
    "subclass_aux",
    "consistency",
    "random_dropout",
    "structured",
    "availability",
    "hierarchy",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pattern_metrics(directory: Path, fill_mode: str) -> dict[str, dict]:
    data = read_json(directory / f"test_missing_patterns_{fill_mode}.json")
    return {name: item["metrics"] for name, item in data["patterns"].items()}


def build_rows(args: argparse.Namespace, fill_mode: str) -> list[dict[str, object]]:
    stores = {
        method: pattern_metrics(getattr(args, arg_name), fill_mode)
        for method, arg_name in METHOD_DIR_ARGS
    }
    rows: list[dict[str, object]] = []
    for pattern in stores["full"]:
        row: dict[str, object] = {"fill_mode": fill_mode, "pattern": pattern}
        for metric in METRICS:
            for prefix, data in stores.items():
                row[f"{prefix}_{metric}"] = data[pattern].get(metric)
            for baseline in BASELINES:
                lite_value = row[f"consistency_lite_{metric}"]
                baseline_value = row[f"{baseline}_{metric}"]
                row[f"consistency_lite_minus_{baseline}_{metric}"] = (
                    None
                    if lite_value is None or baseline_value is None
                    else lite_value - baseline_value
                )
        for label in LABELS:
            for prefix, data in stores.items():
                row[f"{prefix}_{label}_auprc"] = data[pattern]["per_class_auprc"].get(label)
            for baseline in BASELINES:
                lite_value = row[f"consistency_lite_{label}_auprc"]
                baseline_value = row[f"{baseline}_{label}_auprc"]
                row[f"consistency_lite_minus_{baseline}_{label}_auprc"] = (
                    None
                    if lite_value is None or baseline_value is None
                    else lite_value - baseline_value
                )
        rows.append(row)
    return rows


def average(rows: list[dict[str, object]], patterns: tuple[str, ...], key: str) -> float:
    pattern_set = set(patterns)
    values = [float(row[key]) for row in rows if row["pattern"] in pattern_set and row.get(key) is not None]
    if not values:
        raise ValueError(f"No values found for {key} over {patterns}")
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
    keys = (
        "train_cw_consistency_loss",
        "train_mean_consistency_weight",
        "train_min_consistency_weight",
        "train_max_consistency_weight",
        "train_full_mean_confidence",
        "train_masked_mean_confidence",
    )
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


def row_for(rows: list[dict[str, object]], pattern: str) -> dict[str, object]:
    return next(row for row in rows if row["pattern"] == pattern)


def build_summary(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, object]:
    mean_rows = [row for row in rows if row["fill_mode"] == "mean_fill"]
    full_row = row_for(mean_rows, "full")
    random6 = row_for(mean_rows, "random-6")
    limb_only = row_for(mean_rows, "limb-only / precordial-missing")
    summary: dict[str, object] = {
        "method": "A5_lite_confidence_consistency_lambda_0_05",
        "full_dir": str(args.full_dir),
        "random_dir": str(args.random_dir),
        "structured_dir": str(args.structured_dir),
        "availability_dir": str(args.availability_dir),
        "subclass_dir": str(args.subclass_dir),
        "hierarchy_dir": str(args.hierarchy_dir),
        "consistency_dir": str(args.consistency_dir),
        "consistency_lite_dir": str(args.consistency_lite_dir),
        "consistency_lite_full_macro_auroc": float(full_row["consistency_lite_macro_auroc"]),
        "consistency_lite_full_macro_auprc": float(full_row["consistency_lite_macro_auprc"]),
        "consistency_lite_full_macro_f1": float(full_row["consistency_lite_macro_f1"]),
        "consistency_lite_random_6_macro_auprc": float(random6["consistency_lite_macro_auprc"]),
        "consistency_lite_limb_only_macro_auprc": float(limb_only["consistency_lite_macro_auprc"]),
        "consistency_lite_hard_structured_avg_macro_auprc": average(mean_rows, STRUCTURED_HARD_PATTERNS, "consistency_lite_macro_auprc"),
        "consistency_lite_hard_overall_avg_macro_auprc": average(mean_rows, HARD_OVERALL_PATTERNS, "consistency_lite_macro_auprc"),
        "full_pattern_delta_vs_subclass_aux_macro_auprc": float(full_row["consistency_lite_minus_subclass_aux_macro_auprc"]),
        "full_pattern_delta_vs_consistency_macro_auprc": float(full_row["consistency_lite_minus_consistency_macro_auprc"]),
        "full_pattern_delta_vs_random_macro_auprc": float(full_row["consistency_lite_minus_random_dropout_macro_auprc"]),
        "full_pattern_delta_vs_structured_macro_auprc": float(full_row["consistency_lite_minus_structured_macro_auprc"]),
        "full_pattern_delta_vs_availability_macro_auprc": float(full_row["consistency_lite_minus_availability_macro_auprc"]),
        "full_pattern_delta_vs_hierarchy_macro_auprc": float(full_row["consistency_lite_minus_hierarchy_macro_auprc"]),
        "random_6_delta_vs_subclass_aux_macro_auprc": float(random6["consistency_lite_minus_subclass_aux_macro_auprc"]),
        "random_6_delta_vs_consistency_macro_auprc": float(random6["consistency_lite_minus_consistency_macro_auprc"]),
        "limb_only_delta_vs_subclass_aux_macro_auprc": float(limb_only["consistency_lite_minus_subclass_aux_macro_auprc"]),
        "limb_only_delta_vs_consistency_macro_auprc": float(limb_only["consistency_lite_minus_consistency_macro_auprc"]),
    }
    for baseline in BASELINES:
        short = "random" if baseline == "random_dropout" else baseline
        summary[f"hard_structured_delta_vs_{short}_macro_auprc"] = average(
            mean_rows,
            STRUCTURED_HARD_PATTERNS,
            f"consistency_lite_minus_{baseline}_macro_auprc",
        )
        summary[f"hard_overall_delta_vs_{short}_macro_auprc"] = average(
            mean_rows,
            HARD_OVERALL_PATTERNS,
            f"consistency_lite_minus_{baseline}_macro_auprc",
        )
    summary["minority_full_delta_vs_subclass_aux_auprc"] = {
        label: float(full_row[f"consistency_lite_minus_subclass_aux_{label}_auprc"])
        for label in MINORITY_LABELS
    }
    summary["minority_hard_overall_delta_vs_subclass_aux_auprc"] = {
        label: average(mean_rows, HARD_OVERALL_PATTERNS, f"consistency_lite_minus_subclass_aux_{label}_auprc")
        for label in MINORITY_LABELS
    }
    summary["hyp_full_delta_vs_subclass_aux_auprc"] = float(
        full_row["consistency_lite_minus_subclass_aux_HYP_auprc"]
    )
    summary["consistency_lite_diagnostics"] = read_consistency_diagnostics(args.consistency_lite_dir)
    summary["consistency_reference_diagnostics"] = read_consistency_diagnostics(args.consistency_dir)
    return summary


def write_markdown(path: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    lines = [
        "# Confidence Consistency Lite Ablation",
        "",
        f"Consistency-lite directory: `{summary['consistency_lite_dir']}`",
        f"Reference A5 directory: `{summary['consistency_dir']}`",
        "",
        "## Summary",
        "",
        f"Full-pattern Macro AUPRC delta vs A4a: `{summary['full_pattern_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Full-pattern Macro AUPRC delta vs A5 lambda=0.1: `{summary['full_pattern_delta_vs_consistency_macro_auprc']:.6f}`",
        f"Hard structured average delta vs A4a: `{summary['hard_structured_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Hard structured average delta vs A5 lambda=0.1: `{summary['hard_structured_delta_vs_consistency_macro_auprc']:.6f}`",
        f"Hard overall average delta vs A4a: `{summary['hard_overall_delta_vs_subclass_aux_macro_auprc']:.6f}`",
        f"Hard overall average delta vs A5 lambda=0.1: `{summary['hard_overall_delta_vs_consistency_macro_auprc']:.6f}`",
        f"Random-6 Macro AUPRC delta vs A5 lambda=0.1: `{summary['random_6_delta_vs_consistency_macro_auprc']:.6f}`",
        f"Limb-only Macro AUPRC delta vs A5 lambda=0.1: `{summary['limb_only_delta_vs_consistency_macro_auprc']:.6f}`",
        "",
        "| Fill | Pattern | A5-lite AUPRC | Lite-A4a AUPRC | Lite-A5 AUPRC | Lite-A1 AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fill_mode']} | {row['pattern']} | "
            f"{row['consistency_lite_macro_auprc']:.6f} | "
            f"{row['consistency_lite_minus_subclass_aux_macro_auprc']:.6f} | "
            f"{row['consistency_lite_minus_consistency_macro_auprc']:.6f} | "
            f"{row['consistency_lite_minus_random_dropout_macro_auprc']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare A5-lite confidence consistency ablation.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--availability-dir", type=Path, required=True)
    parser.add_argument("--subclass-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--consistency-dir", type=Path, required=True)
    parser.add_argument("--consistency-lite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for fill_mode in ("zero_fill", "mean_fill"):
        rows.extend(build_rows(args, fill_mode))
    summary = build_summary(args, rows)

    csv_path = args.out_dir / "compare_confidence_consistency_lite.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "compare_confidence_consistency_lite.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "compare_confidence_consistency_lite.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
