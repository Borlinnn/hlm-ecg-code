#!/usr/bin/env python3
"""Summarize three-seed reproducibility for locked HLM-ECG methods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER

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
ALL_MISSING_PATTERNS = (
    "random-1",
    "random-3",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
REPORT_PATTERNS = (
    "random-3",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
METHOD_ORDER = (
    "A1_random_dropout",
    "A4a_subclass_auxiliary",
    "A5_lite_confidence_consistency_0p05",
)


@dataclass(frozen=True)
class SeedRun:
    method_id: str
    seed: int
    output_dir: Path


DEFAULT_REGISTRY = (
    SeedRun("A1_random_dropout", 42, Path("outputs/week1_random_dropout/random_dropout_seed42")),
    SeedRun("A1_random_dropout", 7, Path("outputs/week3_multiseed/A1_random_dropout/seed7")),
    SeedRun("A1_random_dropout", 123, Path("outputs/week3_multiseed/A1_random_dropout/seed123")),
    SeedRun("A4a_subclass_auxiliary", 42, Path("outputs/week2_subclass_auxiliary/subclass_aux_seed42")),
    SeedRun("A4a_subclass_auxiliary", 7, Path("outputs/week3_multiseed/A4a_subclass_auxiliary/seed7")),
    SeedRun("A4a_subclass_auxiliary", 123, Path("outputs/week3_multiseed/A4a_subclass_auxiliary/seed123")),
    SeedRun("A5_lite_confidence_consistency_0p05", 42, Path("outputs/week2_confidence_consistency_lite/consistency_lite_seed42")),
    SeedRun("A5_lite_confidence_consistency_0p05", 7, Path("outputs/week3_multiseed/A5_lite_confidence_consistency_0p05/seed7")),
    SeedRun("A5_lite_confidence_consistency_0p05", 123, Path("outputs/week3_multiseed/A5_lite_confidence_consistency_0p05/seed123")),
)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(header for header, _ in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    colspec = "l" + "r" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines.append(" & ".join(header.replace("_", "\\_") for header, _ in columns) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(fmt(row.get(key)).replace("_", "\\_") for _, key in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_val_metric_from_train_log(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {"best_epoch": None, "best_val_macro_auprc": None}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"best_epoch": None, "best_val_macro_auprc": None}
    best = max(rows, key=lambda row: float(row.get("val_macro_auprc") or -1.0))
    return {"best_epoch": int(best["epoch"]), "best_val_macro_auprc": float(best["val_macro_auprc"])}


def read_threshold_source(output_dir: Path) -> str:
    path = output_dir / "thresholds_val.json"
    data = read_json(path)
    return str(data.get("source_split", "unknown"))


def read_pattern_store(output_dir: Path) -> dict[str, dict[str, Any]]:
    data = read_json(output_dir / "test_missing_patterns_mean_fill.json")
    return {pattern: entry["metrics"] for pattern, entry in data["patterns"].items()}


def average_pattern_metric(store: Mapping[str, Mapping[str, Any]], patterns: Sequence[str], metric: str) -> float:
    return float(mean(float(store[pattern][metric]) for pattern in patterns))


def average_per_class_auprc(store: Mapping[str, Mapping[str, Any]], patterns: Sequence[str], label: str) -> float:
    return float(mean(float(store[pattern]["per_class_auprc"][label]) for pattern in patterns))


def run_complete(run: SeedRun) -> bool:
    required = (
        "best_model.pt",
        "train_log.csv",
        "val_metrics.json",
        "test_full_metrics.json",
        "thresholds_val.json",
        "test_missing_patterns_mean_fill.csv",
        "test_missing_patterns_mean_fill.json",
    )
    return all((run.output_dir / name).exists() for name in required)


def manifest_rows(registry: Sequence[SeedRun]) -> list[dict[str, Any]]:
    rows = []
    for run in registry:
        complete = run_complete(run)
        source = read_threshold_source(run.output_dir) if complete else ""
        rows.append(
            {
                "method_id": run.method_id,
                "seed": run.seed,
                "output_dir": str(run.output_dir),
                "complete": complete,
                "has_best_model": (run.output_dir / "best_model.pt").exists(),
                "has_mean_fill_patterns": (run.output_dir / "test_missing_patterns_mean_fill.csv").exists(),
                "thresholds_source_split": source,
            }
        )
    return rows


def per_seed_metric_rows(registry: Sequence[SeedRun]) -> list[dict[str, Any]]:
    rows = []
    missing = [run for run in registry if not run_complete(run)]
    if missing:
        raise RuntimeError("Missing incomplete seed outputs: " + ", ".join(f"{run.method_id}/seed{run.seed}" for run in missing))
    for run in registry:
        full = read_json(run.output_dir / "test_full_metrics.json")
        store = read_pattern_store(run.output_dir)
        train = latest_val_metric_from_train_log(run.output_dir / "train_log.csv")
        if read_threshold_source(run.output_dir) != "val":
            raise RuntimeError(f"{run.method_id}/seed{run.seed} thresholds source is not val")
        row: dict[str, Any] = {
            "method_id": run.method_id,
            "seed": run.seed,
            "output_dir": str(run.output_dir),
            "best_epoch": train["best_epoch"],
            "best_val_macro_auprc": train["best_val_macro_auprc"],
            "full_macro_auroc": full["macro_auroc"],
            "full_macro_auprc": full["macro_auprc"],
            "full_macro_f1": full["macro_f1"],
            "avg_all_missing_macro_auprc": average_pattern_metric(store, ALL_MISSING_PATTERNS, "macro_auprc"),
            "hard_structured_avg_macro_auprc": average_pattern_metric(store, STRUCTURED_HARD_PATTERNS, "macro_auprc"),
            "hard_overall_avg_macro_auprc": average_pattern_metric(store, HARD_OVERALL_PATTERNS, "macro_auprc"),
            "thresholds_source_split": "val",
        }
        for pattern in REPORT_PATTERNS:
            key = pattern.replace(" / ", "_").replace("-", "_").replace(" ", "_")
            row[f"{key}_macro_auprc"] = float(store[pattern]["macro_auprc"])
        for label in LABEL_ORDER:
            row[f"full_{label}_auprc"] = float(full["per_class_auprc"][label])
            row[f"hard_overall_{label}_auprc"] = average_per_class_auprc(store, HARD_OVERALL_PATTERNS, label)
        rows.append(row)
    return rows


def summarize_mean_std(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "full_macro_auroc",
        "full_macro_auprc",
        "full_macro_f1",
        "random_3_macro_auprc",
        "random_6_macro_auprc",
        "limb_only_precordial_missing_macro_auprc",
        "precordial_only_limb_missing_macro_auprc",
        "V1_V3_missing_macro_auprc",
        "V4_V6_missing_macro_auprc",
        "avg_all_missing_macro_auprc",
        "hard_structured_avg_macro_auprc",
        "hard_overall_avg_macro_auprc",
    ]
    out = []
    for method_id in METHOD_ORDER:
        selected = [row for row in rows if row["method_id"] == method_id]
        item: dict[str, Any] = {"method_id": method_id, "n_seeds": len(selected), "seeds": ",".join(str(row["seed"]) for row in selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = float(mean(values))
            item[f"{metric}_std"] = float(stdev(values)) if len(values) > 1 else 0.0
        out.append(item)
    return out


def row_by_method_seed(rows: Sequence[Mapping[str, Any]], method_id: str, seed: int) -> Mapping[str, Any]:
    return next(row for row in rows if row["method_id"] == method_id and int(row["seed"]) == int(seed))


def build_delta_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = (
        ("A4a_minus_A1", "A4a_subclass_auxiliary", "A1_random_dropout"),
        ("A5_lite_minus_A4a", "A5_lite_confidence_consistency_0p05", "A4a_subclass_auxiliary"),
        ("A5_lite_minus_A1", "A5_lite_confidence_consistency_0p05", "A1_random_dropout"),
    )
    metrics = (
        "full_macro_auprc",
        "hard_structured_avg_macro_auprc",
        "hard_overall_avg_macro_auprc",
        "hard_overall_MI_auprc",
        "hard_overall_CD_auprc",
        "hard_overall_HYP_auprc",
    )
    out = []
    for comparison_id, method_a, method_b in comparisons:
        for seed in (42, 7, 123):
            a = row_by_method_seed(rows, method_a, seed)
            b = row_by_method_seed(rows, method_b, seed)
            row: dict[str, Any] = {"comparison_id": comparison_id, "method_a": method_a, "method_b": method_b, "seed": seed}
            for metric in metrics:
                row[f"{metric}_delta"] = float(a[metric]) - float(b[metric])
            out.append(row)
        selected = [row for row in out if row["comparison_id"] == comparison_id and row["seed"] in (42, 7, 123)]
        summary: dict[str, Any] = {"comparison_id": comparison_id, "method_a": method_a, "method_b": method_b, "seed": "mean_std"}
        for metric in metrics:
            values = [float(row[f"{metric}_delta"]) for row in selected]
            summary[f"{metric}_delta_mean"] = float(mean(values))
            summary[f"{metric}_delta_std"] = float(stdev(values)) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def make_decision(rows: Sequence[Mapping[str, Any]], delta_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a4a_vs_a1 = [row for row in delta_rows if row["comparison_id"] == "A4a_minus_A1" and row["seed"] != "mean_std"]
    a5_vs_a4a = [row for row in delta_rows if row["comparison_id"] == "A5_lite_minus_A4a" and row["seed"] != "mean_std"]
    a4a_hard_structured_all_positive = all(float(row["hard_structured_avg_macro_auprc_delta"]) > 0 for row in a4a_vs_a1)
    a4a_hard_overall_all_positive = all(float(row["hard_overall_avg_macro_auprc_delta"]) > 0 for row in a4a_vs_a1)
    a4a_full_deltas = [float(row["full_macro_auprc_delta"]) for row in a4a_vs_a1]
    a5_full_all_positive = all(float(row["full_macro_auprc_delta"]) > 0 for row in a5_vs_a4a)
    a5_hard_tradeoff_all_negative = all(float(row["hard_overall_avg_macro_auprc_delta"]) < 0 for row in a5_vs_a4a)
    minority_positive_count = sum(
        any(float(row[f"hard_overall_{label}_auprc_delta"]) > 0 for label in ("MI", "CD", "HYP"))
        for row in a4a_vs_a1
    )
    return {
        "a4a_hard_structured_all_seeds_gt_A1": a4a_hard_structured_all_positive,
        "a4a_hard_overall_all_seeds_gt_A1": a4a_hard_overall_all_positive,
        "a4a_full_delta_vs_A1_mean": float(mean(a4a_full_deltas)),
        "a4a_full_delta_vs_A1_std": float(stdev(a4a_full_deltas)) if len(a4a_full_deltas) > 1 else 0.0,
        "a4a_minority_hard_overall_improves_in_seed_count": minority_positive_count,
        "a4a_stable_final_robustness_candidate": bool(
            a4a_hard_structured_all_positive and a4a_hard_overall_all_positive and abs(mean(a4a_full_deltas)) <= 0.01
        ),
        "a5_lite_full_all_seeds_gt_A4a": a5_full_all_positive,
        "a5_lite_hard_overall_tradeoff_all_seeds_lt_A4a": a5_hard_tradeoff_all_negative,
        "a5_lite_stable_balanced_candidate": bool(a5_full_all_positive and a5_hard_tradeoff_all_negative),
        "records500_used": False,
        "recommended_next_step": "paper_tables_and_writing" if a4a_hard_overall_all_positive else "review_multiseed_instability",
    }


def write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "multiseed_manifest.csv", rows)
    write_json(out_dir / "multiseed_manifest.json", {"rows": rows})
    columns = [("method", "method_id"), ("seed", "seed"), ("complete", "complete"), ("thresholds", "thresholds_source_split"), ("output", "output_dir")]
    (out_dir / "multiseed_manifest.md").write_text("# Multiseed Manifest\n\n" + markdown_table(rows, columns), encoding="utf-8")


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "multiseed_summary_mean_std.csv", rows)
    write_json(out_dir / "multiseed_summary_mean_std.json", {"rows": rows})
    columns = [
        ("method", "method_id"),
        ("n", "n_seeds"),
        ("full AUPRC mean", "full_macro_auprc_mean"),
        ("full AUPRC std", "full_macro_auprc_std"),
        ("hard structured mean", "hard_structured_avg_macro_auprc_mean"),
        ("hard structured std", "hard_structured_avg_macro_auprc_std"),
        ("hard overall mean", "hard_overall_avg_macro_auprc_mean"),
        ("hard overall std", "hard_overall_avg_macro_auprc_std"),
    ]
    (out_dir / "multiseed_summary_mean_std.md").write_text("# Multiseed Mean ± Std\n\n" + markdown_table(rows, columns), encoding="utf-8")
    table_rows = [
        {
            "method_id": row["method_id"],
            "full_macro_auprc": f"{row['full_macro_auprc_mean']:.4f} ± {row['full_macro_auprc_std']:.4f}",
            "hard_structured_avg_macro_auprc": f"{row['hard_structured_avg_macro_auprc_mean']:.4f} ± {row['hard_structured_avg_macro_auprc_std']:.4f}",
            "hard_overall_avg_macro_auprc": f"{row['hard_overall_avg_macro_auprc_mean']:.4f} ± {row['hard_overall_avg_macro_auprc_std']:.4f}",
        }
        for row in rows
    ]
    write_csv(out_dir / "multiseed_table_main.csv", table_rows)
    table_columns = [(key, key) for key in table_rows[0]]
    (out_dir / "multiseed_table_main.md").write_text(markdown_table(table_rows, table_columns), encoding="utf-8")
    (out_dir / "multiseed_table_main.tex").write_text(latex_table(table_rows, table_columns), encoding="utf-8")


def write_deltas(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "multiseed_delta_summary.csv", rows)
    write_json(out_dir / "multiseed_delta_summary.json", {"rows": rows})
    columns = [
        ("comparison", "comparison_id"),
        ("seed", "seed"),
        ("full delta", "full_macro_auprc_delta"),
        ("hard structured delta", "hard_structured_avg_macro_auprc_delta"),
        ("hard overall delta", "hard_overall_avg_macro_auprc_delta"),
    ]
    filtered = [row for row in rows if row["seed"] != "mean_std"]
    (out_dir / "multiseed_delta_summary.md").write_text("# Multiseed Deltas\n\n" + markdown_table(filtered, columns), encoding="utf-8")


def write_decision(out_dir: Path, decision: Mapping[str, Any]) -> None:
    write_json(out_dir / "multiseed_decision.json", dict(decision))
    lines = ["# Multiseed Decision", ""]
    for key, value in decision.items():
        lines.append(f"- `{key}`: `{value}`")
    (out_dir / "multiseed_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out_dir: Path, registry: Sequence[SeedRun] = DEFAULT_REGISTRY) -> dict[str, Any]:
    if Path("data/ptb-xl/records500").exists():
        raise RuntimeError("records500 exists; refusing multiseed summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_rows(registry)
    write_manifest(out_dir, manifest)
    metric_rows = per_seed_metric_rows(registry)
    summary = summarize_mean_std(metric_rows)
    deltas = build_delta_rows(metric_rows)
    decision = make_decision(metric_rows, deltas)
    write_csv(out_dir / "multiseed_per_seed_metrics.csv", metric_rows)
    write_json(out_dir / "multiseed_per_seed_metrics.json", {"rows": metric_rows})
    write_summary(out_dir, summary)
    write_deltas(out_dir, deltas)
    write_decision(out_dir, decision)
    result = {
        "out_dir": str(out_dir),
        "n_seed_runs": len(metric_rows),
        "records500_used": False,
        "a4a_stable_final_robustness_candidate": decision["a4a_stable_final_robustness_candidate"],
        "a5_lite_stable_balanced_candidate": decision["a5_lite_stable_balanced_candidate"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HLM-ECG three-seed reproducibility.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/week3_multiseed_summary"))
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
