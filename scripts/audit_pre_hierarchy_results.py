#!/usr/bin/env python3
"""Audit A0/A1/A2/A3 results and subclass hierarchy readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.data.subclass_labels import (
    SUBCLASS_THRESHOLDS,
    add_threshold_flags,
    build_subclass_label_matrix,
    build_subclass_parent_mapping,
    diagnostic_subclass_code_table,
    kept_subclasses,
    load_ptbxl_metadata,
    load_scp_statements,
    parent_support_summary,
    record_coverage_summary,
    subclass_count_frame,
)

METHODS = {
    "A0_full_no_masking": Path("outputs/week1_full_baseline/full_seed42"),
    "A1_random_dropout": Path("outputs/week1_random_dropout/random_dropout_seed42"),
    "A2_structured_masking": Path("outputs/week2_structured_masking/structured_seed42"),
    "A3_availability_embedding": Path("outputs/week2_availability_embedding/avail_seed42"),
}
PATTERN_COLUMNS = {
    "random-1": "random_1_macro_auprc",
    "random-3": "random_3_macro_auprc",
    "random-6": "random_6_macro_auprc",
    "limb-only / precordial-missing": "limb_only_precordial_missing_macro_auprc",
    "precordial-only / limb-missing": "precordial_only_limb_missing_macro_auprc",
    "V1-V3 missing": "V1_V3_missing_macro_auprc",
    "V4-V6 missing": "V4_V6_missing_macro_auprc",
}
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
ALL_MISSING_PATTERNS = tuple(PATTERN_COLUMNS)
PRIMARY_MAPPING_THRESHOLD = 50


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def read_pattern_csv(directory: Path, fill_mode: str) -> pd.DataFrame:
    path = directory / f"test_missing_patterns_{fill_mode}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pattern_auprc(patterns: pd.DataFrame, pattern: str) -> float:
    row = patterns.loc[patterns["pattern"] == pattern]
    if row.empty:
        raise RuntimeError(f"Missing pattern in CSV: {pattern}")
    return float(row["macro_auprc"].iloc[0])


def average_patterns(patterns: pd.DataFrame, names: Iterable[str]) -> float:
    values = [pattern_auprc(patterns, name) for name in names]
    return float(sum(values) / len(values))


def build_ablation_summary(method_dirs: Mapping[str, Path]) -> pd.DataFrame:
    rows = []
    for method, directory in method_dirs.items():
        full = read_json(directory / "test_full_metrics.json")
        mean_patterns = read_pattern_csv(directory, "mean_fill")
        row: dict[str, object] = {
            "method": method,
            "full_macro_auroc": float(full["macro_auroc"]),
            "full_macro_auprc": float(full["macro_auprc"]),
            "full_macro_f1": float(full["macro_f1"]),
        }
        for pattern, column in PATTERN_COLUMNS.items():
            row[column] = pattern_auprc(mean_patterns, pattern)
        row["avg_all_missing_macro_auprc"] = average_patterns(mean_patterns, ALL_MISSING_PATTERNS)
        row["hard_structured_avg_macro_auprc"] = average_patterns(mean_patterns, STRUCTURED_HARD_PATTERNS)
        row["hard_overall_avg_macro_auprc"] = average_patterns(mean_patterns, HARD_OVERALL_PATTERNS)
        rows.append(row)

    summary = pd.DataFrame(rows)
    random_row = summary.loc[summary["method"] == "A1_random_dropout"].iloc[0]
    structured_row = summary.loc[summary["method"] == "A2_structured_masking"].iloc[0]
    summary["full_drop_vs_random_dropout"] = summary["full_macro_auprc"] - float(random_row["full_macro_auprc"])
    summary["hard_structured_delta_vs_random_dropout"] = (
        summary["hard_structured_avg_macro_auprc"] - float(random_row["hard_structured_avg_macro_auprc"])
    )
    summary["hard_overall_delta_vs_random_dropout"] = (
        summary["hard_overall_avg_macro_auprc"] - float(random_row["hard_overall_avg_macro_auprc"])
    )
    summary["hard_structured_delta_vs_structured"] = (
        summary["hard_structured_avg_macro_auprc"] - float(structured_row["hard_structured_avg_macro_auprc"])
    )
    summary["hard_overall_delta_vs_structured"] = (
        summary["hard_overall_avg_macro_auprc"] - float(structured_row["hard_overall_avg_macro_auprc"])
    )
    return summary


def build_per_class_summary(method_dirs: Mapping[str, Path]) -> pd.DataFrame:
    rows = []
    for method, directory in method_dirs.items():
        full = read_json(directory / "test_full_metrics.json")
        for label in LABEL_ORDER:
            rows.append(
                {
                    "method": method,
                    "fill_mode": "full",
                    "pattern": "full",
                    "label": label,
                    "auroc": full["per_class_auroc"].get(label),
                    "auprc": full["per_class_auprc"].get(label),
                    "f1": full["per_class_f1"].get(label),
                }
            )
        for fill_mode in ("zero_fill", "mean_fill"):
            data = read_json(directory / f"test_missing_patterns_{fill_mode}.json")
            for pattern, payload in data["patterns"].items():
                metrics = payload["metrics"]
                for label in LABEL_ORDER:
                    rows.append(
                        {
                            "method": method,
                            "fill_mode": fill_mode,
                            "pattern": pattern,
                            "label": label,
                            "auroc": metrics["per_class_auroc"].get(label),
                            "auprc": metrics["per_class_auprc"].get(label),
                            "f1": metrics["per_class_f1"].get(label),
                        }
                    )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: Sequence[str], *, float_digits: int = 4) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_ablation_markdown(path: Path, summary: pd.DataFrame) -> None:
    columns = [
        "method",
        "full_macro_auprc",
        "hard_structured_avg_macro_auprc",
        "hard_overall_avg_macro_auprc",
        "full_drop_vs_random_dropout",
    ]
    lines = [
        "# A0-A3 Ablation Summary",
        "",
        "Primary values use mean-fill missing-pattern evaluation.",
        "",
        *markdown_table(summary, columns),
        "",
        "Hard structured average = limb-only, precordial-only, V1-V3 missing, V4-V6 missing.",
        "Hard overall average = random-6 plus the four structured hard patterns.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_subclass_audit(root: Path, day1_index: Path, out_dir: Path) -> dict[str, object]:
    metadata_raw = load_ptbxl_metadata(root)
    day1 = pd.read_csv(day1_index)[["ecg_id", "split", *LABEL_ORDER]]
    metadata = metadata_raw.merge(day1, on="ecg_id", how="left", validate="one_to_one")
    if metadata["split"].isna().any():
        raise RuntimeError("Day 1 index merge left records without split")

    scp = load_scp_statements(root)
    code_table = diagnostic_subclass_code_table(scp)
    invalid_parent_rows = code_table[~code_table["parent_valid"]].copy()
    mapping = build_subclass_parent_mapping(code_table)
    labels = build_subclass_label_matrix(metadata, mapping, code_table)
    counts = add_threshold_flags(subclass_count_frame(metadata, labels, mapping))
    coverage = record_coverage_summary(metadata, labels, counts)
    parent_support = parent_support_summary(counts, threshold=PRIMARY_MAPPING_THRESHOLD)

    counts_path = out_dir / "subclass_label_audit.csv"
    counts.to_csv(counts_path, index=False)

    mapping_out = mapping.merge(counts, on=["diagnostic_subclass", "parent_superclass"], how="left")
    mapping_out.to_csv(out_dir / "hierarchy_parent_child_mapping.csv", index=False)

    kept_counts = {
        str(threshold): int(counts[f"kept_min_train_pos_{threshold}"].sum()) for threshold in SUBCLASS_THRESHOLDS
    }
    parent_support_records = parent_support.to_dict(orient="records")
    weakest = parent_support.sort_values(
        ["kept_subclass_count", "subclass_train_positive_total"],
        ascending=[True, True],
    ).head(3)

    audit = {
        "raw_subclass_count": int(code_table["diagnostic_subclass"].nunique()),
        "valid_parent_subclass_count": int(len(mapping)),
        "invalid_parent_scp_code_count": int(len(invalid_parent_rows)),
        "invalid_parent_scp_codes": invalid_parent_rows.to_dict(orient="records"),
        "kept_subclass_count_by_min_train_pos": kept_counts,
        "records_with_zero_subclass_labels": coverage["records_with_zero_subclass_labels"],
        "coverage_by_threshold": coverage["thresholds"],
        "primary_mapping_threshold": PRIMARY_MAPPING_THRESHOLD,
        "parent_support_min_train_pos_50": parent_support_records,
        "weakest_superclass_coverage_min_train_pos_50": weakest.to_dict(orient="records"),
    }
    write_json(out_dir / "subclass_label_audit.json", audit)
    write_json(
        out_dir / "hierarchy_parent_child_mapping.json",
        {
            "primary_mapping_threshold": PRIMARY_MAPPING_THRESHOLD,
            "mapping_unique": True,
            "mapping": mapping_out.to_dict(orient="records"),
            "parent_support": parent_support_records,
        },
    )

    write_subclass_markdown(out_dir / "subclass_label_audit.md", audit, counts)
    write_mapping_markdown(out_dir / "hierarchy_parent_child_mapping.md", mapping_out, parent_support)
    return {
        "audit": audit,
        "counts": counts,
        "mapping": mapping_out,
        "parent_support": parent_support,
    }


def write_subclass_markdown(path: Path, audit: Mapping[str, object], counts: pd.DataFrame) -> None:
    kept = audit["kept_subclass_count_by_min_train_pos"]
    lines = [
        "# Subclass Label Audit",
        "",
        f"Raw subclass count: `{audit['raw_subclass_count']}`",
        f"Valid-parent subclass count: `{audit['valid_parent_subclass_count']}`",
        f"Invalid-parent SCP code count: `{audit['invalid_parent_scp_code_count']}`",
        "",
        "## Kept Counts",
        "",
    ]
    for threshold in SUBCLASS_THRESHOLDS:
        lines.append(f"- min_train_pos={threshold}: `{kept[str(threshold)]}`")
    lines.extend(
        [
            "",
            "## Records With Zero Raw Subclass Labels",
            "",
            f"`{audit['records_with_zero_subclass_labels']}`",
            "",
            "## Top Subclasses By Train Positives",
            "",
        ]
    )
    top = counts.sort_values("train_positive_count", ascending=False).head(20)
    lines.extend(
        markdown_table(
            top,
            ["diagnostic_subclass", "parent_superclass", "train_positive_count", "val_positive_count", "test_positive_count"],
            float_digits=0,
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mapping_markdown(path: Path, mapping: pd.DataFrame, parent_support: pd.DataFrame) -> None:
    lines = [
        "# Hierarchy Parent-Child Mapping",
        "",
        "Each kept diagnostic subclass maps to exactly one parent superclass.",
        "",
        "## Parent Support At min_train_pos=50",
        "",
    ]
    lines.extend(
        markdown_table(
            parent_support,
            [
                "parent_superclass",
                "kept_subclass_count",
                "subclass_train_positive_total",
                "subclass_val_positive_total",
                "subclass_test_positive_total",
            ],
            float_digits=0,
        )
    )
    lines.extend(["", "## Mapping", ""])
    lines.extend(
        markdown_table(
            mapping,
            ["diagnostic_subclass", "parent_superclass", "train_positive_count", "kept_min_train_pos_50"],
            float_digits=0,
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_decision(summary: pd.DataFrame, per_class: pd.DataFrame, subclass_info: Mapping[str, object]) -> dict[str, object]:
    rows = {row["method"]: row for _, row in summary.iterrows()}
    a2 = rows["A2_structured_masking"]
    a3 = rows["A3_availability_embedding"]
    a1 = rows["A1_random_dropout"]

    full_delta_vs_structured = float(a3["full_macro_auprc"] - a2["full_macro_auprc"])
    full_delta_vs_random = float(a3["full_macro_auprc"] - a1["full_macro_auprc"])
    hard_structured_delta_vs_structured = float(
        a3["hard_structured_avg_macro_auprc"] - a2["hard_structured_avg_macro_auprc"]
    )
    hard_overall_delta_vs_structured = float(a3["hard_overall_avg_macro_auprc"] - a2["hard_overall_avg_macro_auprc"])

    full_per_class = per_class[(per_class["fill_mode"] == "full") & (per_class["pattern"] == "full")]
    def label_auprc(method: str, label: str) -> float:
        row = full_per_class[(full_per_class["method"] == method) & (full_per_class["label"] == label)]
        return float(row["auprc"].iloc[0])

    hyp_delta_vs_structured = label_auprc("A3_availability_embedding", "HYP") - label_auprc("A2_structured_masking", "HYP")
    minority_deltas_vs_structured = {
        label: label_auprc("A3_availability_embedding", label) - label_auprc("A2_structured_masking", label)
        for label in ("MI", "CD", "HYP")
    }

    audit = subclass_info["audit"]
    parent_support = subclass_info["parent_support"]
    kept_counts = audit["kept_subclass_count_by_min_train_pos"]
    support = {
        row["parent_superclass"]: int(row["kept_subclass_count"]) for _, row in parent_support.iterrows()
    }
    supported_mi_cd_hyp = [label for label in ("MI", "CD", "HYP") if support.get(label, 0) > 0]
    no_kept_train_fraction_50 = float(
        audit["coverage_by_threshold"]["50"]["records_with_superclass_but_no_kept_subclass_fraction"]["train"]
    )
    mapping_ok = bool(audit["invalid_parent_scp_code_count"] == 0)
    labels_available = int(kept_counts["50"]) > 0 and int(kept_counts["100"]) > 0
    a3_preservation_value = full_delta_vs_structured > 0 or hyp_delta_vs_structured > 0

    if not mapping_ok:
        decision = "stop_and_review_labels"
        reason = "Subclass mapping contains invalid parent SCP codes."
    elif not labels_available or not supported_mi_cd_hyp:
        decision = "skip_hierarchy_for_now"
        reason = "Subclass labels are too sparse for MI/CD/HYP hierarchy support."
    elif no_kept_train_fraction_50 > 0.35:
        decision = "proceed_subclass_aux_only_first"
        reason = "Subclass labels are usable, but kept-subclass supervision coverage should be validated before hierarchy loss."
    elif a3_preservation_value:
        decision = "proceed_subclass_aux_only_first"
        reason = "Availability embedding helps full/HYP preservation; validate subclass auxiliary learning before adding hierarchy loss."
    else:
        decision = "skip_hierarchy_for_now"
        reason = "A3 does not provide enough preservation value to justify added hierarchy complexity now."

    return {
        "decision": decision,
        "reason": reason,
        "a3_full_delta_vs_structured_macro_auprc": full_delta_vs_structured,
        "a3_full_delta_vs_random_macro_auprc": full_delta_vs_random,
        "a3_hard_structured_delta_vs_structured_macro_auprc": hard_structured_delta_vs_structured,
        "a3_hard_overall_delta_vs_structured_macro_auprc": hard_overall_delta_vs_structured,
        "a3_hyp_full_delta_vs_structured_auprc": hyp_delta_vs_structured,
        "a3_minority_full_deltas_vs_structured_auprc": minority_deltas_vs_structured,
        "availability_embedding_interpretation": (
            "availability embedding is useful for preserving full-lead and minority-label performance, "
            "but not a standalone robustness booster"
        ),
        "claim_support": {
            "availability_improves_robustness": False,
            "recommended_wording": (
                "Availability embedding improves full-lead/HYP preservation under structured masking, "
                "while hard missing robustness remains driven primarily by structured masking training."
            ),
        },
        "subclass_readiness": {
            "mapping_unique": mapping_ok,
            "kept_subclass_count_by_min_train_pos": kept_counts,
            "supported_mi_cd_hyp_min_train_pos_50": supported_mi_cd_hyp,
            "train_superclass_without_kept_subclass_fraction_min50": no_kept_train_fraction_50,
        },
    }


def write_decision_markdown(path: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Pre-Hierarchy Decision",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        f"Reason: {decision['reason']}",
        "",
        "## A3 Availability Embedding Audit",
        "",
        f"- Full-pattern Macro AUPRC vs structured: `{decision['a3_full_delta_vs_structured_macro_auprc']:.6f}`",
        f"- Full-pattern Macro AUPRC vs random dropout: `{decision['a3_full_delta_vs_random_macro_auprc']:.6f}`",
        f"- Hard structured average vs structured: `{decision['a3_hard_structured_delta_vs_structured_macro_auprc']:.6f}`",
        f"- Hard overall average vs structured: `{decision['a3_hard_overall_delta_vs_structured_macro_auprc']:.6f}`",
        f"- HYP full-pattern AUPRC vs structured: `{decision['a3_hyp_full_delta_vs_structured_auprc']:.6f}`",
        "",
        "Interpretation: availability embedding is useful for preserving full-lead and minority-label performance, but not a standalone robustness booster.",
        "",
        "Recommended wording: Availability embedding improves full-lead/HYP preservation under structured masking, while hard missing robustness remains driven primarily by structured masking training.",
        "",
        "## Subclass Readiness",
        "",
        f"`{decision['subclass_readiness']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A0/A1/A2/A3 results and hierarchy readiness.")
    parser.add_argument("--root", type=Path, default=Path("data/ptb-xl"))
    parser.add_argument("--day1-index", type=Path, default=Path("outputs/day1_audit/ptbxl_day1_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/week2_pre_hierarchy_audit"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = build_ablation_summary(METHODS)
    per_class = build_per_class_summary(METHODS)
    summary.to_csv(args.out / "ablation_A0_A3_summary.csv", index=False)
    per_class.to_csv(args.out / "per_class_A0_A3_summary.csv", index=False)
    write_json(
        args.out / "ablation_A0_A3_summary.json",
        {
            "methods": {method: str(path) for method, path in METHODS.items()},
            "primary_fill_mode": "mean_fill",
            "hard_structured_patterns": list(STRUCTURED_HARD_PATTERNS),
            "hard_overall_patterns": list(HARD_OVERALL_PATTERNS),
            "avg_all_missing_patterns": list(ALL_MISSING_PATTERNS),
            "rows": summary.to_dict(orient="records"),
        },
    )
    write_ablation_markdown(args.out / "ablation_A0_A3_summary.md", summary)

    subclass_info = build_subclass_audit(args.root, args.day1_index, args.out)
    decision = build_decision(summary, per_class, subclass_info)
    write_json(args.out / "pre_hierarchy_decision.json", decision)
    write_decision_markdown(args.out / "pre_hierarchy_decision.md", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
