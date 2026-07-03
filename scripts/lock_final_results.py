#!/usr/bin/env python3
"""Consolidate A0-A5-lite HLM-ECG results and lock final candidates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")

PATTERNS = (
    "full",
    "random-1",
    "random-3",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)

ALL_MISSING_PATTERNS = PATTERNS[1:]
HARD_STRUCTURED_PATTERNS = (
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
HARD_OVERALL_PATTERNS = ("random-6", *HARD_STRUCTURED_PATTERNS)

PATTERN_SLUGS = {
    "full": "full",
    "random-1": "random_1",
    "random-3": "random_3",
    "random-6": "random_6",
    "limb-only / precordial-missing": "limb_only_precordial_missing",
    "precordial-only / limb-missing": "precordial_only_limb_missing",
    "V1-V3 missing": "V1_V3_missing",
    "V4-V6 missing": "V4_V6_missing",
}


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    output_dir: Path
    recommended_role: str
    eval_script: str
    config: str


METHOD_REGISTRY = (
    MethodSpec(
        "A0_full_no_masking",
        "A0 full no masking",
        Path("outputs/week1_full_baseline/full_seed42"),
        "baseline",
        "scripts/evaluate_full_baseline_patterns.py",
        "configs/full_baseline.yaml",
    ),
    MethodSpec(
        "A1_random_dropout",
        "A1 random dropout",
        Path("outputs/week1_random_dropout/random_dropout_seed42"),
        "strong baseline",
        "scripts/evaluate_random_dropout_patterns.py",
        "configs/random_dropout.yaml",
    ),
    MethodSpec(
        "A2_structured_masking",
        "A2 structured masking",
        Path("outputs/week2_structured_masking/structured_seed42"),
        "positive module",
        "scripts/evaluate_structured_masking_patterns.py",
        "configs/structured_masking.yaml",
    ),
    MethodSpec(
        "A3_availability_embedding",
        "A3 availability embedding",
        Path("outputs/week2_availability_embedding/avail_seed42"),
        "mixed module",
        "scripts/evaluate_availability_embedding_patterns.py",
        "configs/availability_embedding.yaml",
    ),
    MethodSpec(
        "A4a_subclass_auxiliary",
        "A4a subclass auxiliary",
        Path("outputs/week2_subclass_auxiliary/subclass_aux_seed42"),
        "final robustness candidate",
        "scripts/evaluate_subclass_auxiliary_patterns.py",
        "configs/subclass_auxiliary.yaml",
    ),
    MethodSpec(
        "A4b_hierarchy_loss",
        "A4b hierarchy loss",
        Path("outputs/week2_hierarchy_ablation/hierarchy_seed42"),
        "negative / weak ablation",
        "scripts/evaluate_hierarchy_ablation_patterns.py",
        "configs/hierarchy_ablation.yaml",
    ),
    MethodSpec(
        "A5_confidence_consistency_0p10",
        "A5 confidence consistency 0.10",
        Path("outputs/week2_confidence_consistency/consistency_seed42"),
        "full-preserving ablation",
        "scripts/evaluate_confidence_consistency_patterns.py",
        "configs/confidence_consistency.yaml",
    ),
    MethodSpec(
        "A5_lite_confidence_consistency_0p05",
        "A5-lite confidence consistency 0.05",
        Path("outputs/week2_confidence_consistency_lite/consistency_lite_seed42"),
        "full-preserving ablation",
        "scripts/evaluate_confidence_consistency_patterns.py",
        "configs/confidence_consistency_lite.yaml",
    ),
)

KEY_PREDICTION_METHODS = (
    "A0_full_no_masking",
    "A1_random_dropout",
    "A2_structured_masking",
    "A4a_subclass_auxiliary",
    "A5_lite_confidence_consistency_0p05",
)
KEY_PREDICTION_PATTERNS = (
    "full",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)

REQUIRED_PREDICTION_COLUMNS = (
    "ecg_id",
    "method",
    "pattern",
    "split",
    "fill_mode",
    "threshold_source_split",
    *(f"y_true_{label}" for label in LABELS),
    *(f"prob_{label}" for label in LABELS),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def average_metric(pattern_store: dict[str, dict[str, Any]], patterns: Iterable[str], metric: str) -> float:
    values = [float(pattern_store[name][metric]) for name in patterns]
    return sum(values) / len(values)


def average_per_class(pattern_store: dict[str, dict[str, Any]], patterns: Iterable[str], label: str) -> float:
    values = [float(pattern_store[name]["per_class_auprc"][label]) for name in patterns]
    return sum(values) / len(values)


def read_pattern_metrics(method: MethodSpec, fill_mode: str) -> dict[str, dict[str, Any]]:
    path = method.output_dir / f"test_missing_patterns_{fill_mode}.json"
    data = read_json(path)
    patterns = data.get("patterns", {})
    missing = set(PATTERNS).difference(patterns)
    if missing:
        raise RuntimeError(f"{method.method_id} {fill_mode} missing patterns: {sorted(missing)}")
    return {name: patterns[name]["metrics"] for name in PATTERNS}


def read_best_epoch_and_val(method: MethodSpec) -> tuple[int | None, float | None]:
    path = method.output_dir / "train_log.csv"
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    if "val_macro_auprc" not in rows[0]:
        return None, None
    best = max(rows, key=lambda row: float(row.get("val_macro_auprc") or -1.0))
    return int(best["epoch"]), float(best["val_macro_auprc"])


def read_threshold_source(method: MethodSpec) -> str:
    path = method.output_dir / "thresholds_val.json"
    data = read_json(path)
    source = str(data.get("source_split", "unknown"))
    if source != "val":
        raise RuntimeError(f"{method.method_id} thresholds source split is {source}, expected val")
    return source


def records500_used(method: MethodSpec) -> bool:
    if Path("data/ptb-xl/records500").exists():
        return True
    for path in (method.output_dir / "config_used.yaml", method.output_dir / "config.yaml"):
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "records500" in text or "filename_hr" in text:
                return True
    return False


def discover_read_paths(method: MethodSpec) -> dict[str, list[str]]:
    return {
        "json": sorted(str(path) for path in method.output_dir.glob("*.json")),
        "csv": sorted(str(path) for path in method.output_dir.glob("*.csv")),
        "markdown": sorted(str(path) for path in method.output_dir.glob("*.md")),
    }


def build_all_pattern_rows(fill_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_REGISTRY:
        store = read_pattern_metrics(method, fill_mode)
        for pattern in PATTERNS:
            metrics = store[pattern]
            row: dict[str, Any] = {
                "method_id": method.method_id,
                "method_display_name": method.display_name,
                "output_dir": str(method.output_dir),
                "fill_mode": fill_mode,
                "pattern": pattern,
                "n": metrics.get("n", ""),
                "macro_auroc": metrics.get("macro_auroc"),
                "macro_auprc": metrics.get("macro_auprc"),
                "macro_f1": metrics.get("macro_f1"),
                "bce_nll": metrics.get("bce_nll"),
            }
            for label in LABELS:
                row[f"{label}_auroc"] = metrics.get("per_class_auroc", {}).get(label)
                row[f"{label}_auprc"] = metrics.get("per_class_auprc", {}).get(label)
                row[f"{label}_f1"] = metrics.get("per_class_f1", {}).get(label)
            rows.append(row)
    return rows


def build_summary_rows() -> list[dict[str, Any]]:
    stores = {method.method_id: read_pattern_metrics(method, "mean_fill") for method in METHOD_REGISTRY}
    a1 = stores["A1_random_dropout"]
    a4a = stores["A4a_subclass_auxiliary"]
    rows: list[dict[str, Any]] = []
    for method in METHOD_REGISTRY:
        store = stores[method.method_id]
        full = store["full"]
        best_epoch, best_val = read_best_epoch_and_val(method)
        row = {
            "method_id": method.method_id,
            "method_display_name": method.display_name,
            "output_dir": str(method.output_dir),
            "best_epoch": best_epoch,
            "best_val_macro_auprc": best_val,
            "full_macro_auroc": full["macro_auroc"],
            "full_macro_auprc": full["macro_auprc"],
            "full_macro_f1": full["macro_f1"],
            "full_bce_or_nll": full.get("bce_nll"),
            "random_1_macro_auprc": store["random-1"]["macro_auprc"],
            "random_3_macro_auprc": store["random-3"]["macro_auprc"],
            "random_6_macro_auprc": store["random-6"]["macro_auprc"],
            "limb_only_precordial_missing_macro_auprc": store["limb-only / precordial-missing"]["macro_auprc"],
            "precordial_only_limb_missing_macro_auprc": store["precordial-only / limb-missing"]["macro_auprc"],
            "V1_V3_missing_macro_auprc": store["V1-V3 missing"]["macro_auprc"],
            "V4_V6_missing_macro_auprc": store["V4-V6 missing"]["macro_auprc"],
            "avg_all_missing_macro_auprc": average_metric(store, ALL_MISSING_PATTERNS, "macro_auprc"),
            "hard_structured_avg_macro_auprc": average_metric(store, HARD_STRUCTURED_PATTERNS, "macro_auprc"),
            "hard_overall_avg_macro_auprc": average_metric(store, HARD_OVERALL_PATTERNS, "macro_auprc"),
            "thresholds_source_split": read_threshold_source(method),
            "records500_used": records500_used(method),
            "recommended_role": method.recommended_role,
        }
        row["delta_full_vs_A1_random_dropout"] = row["full_macro_auprc"] - a1["full"]["macro_auprc"]
        row["delta_hard_structured_vs_A1_random_dropout"] = row["hard_structured_avg_macro_auprc"] - average_metric(
            a1, HARD_STRUCTURED_PATTERNS, "macro_auprc"
        )
        row["delta_hard_overall_vs_A1_random_dropout"] = row["hard_overall_avg_macro_auprc"] - average_metric(
            a1, HARD_OVERALL_PATTERNS, "macro_auprc"
        )
        row["delta_full_vs_A4a"] = row["full_macro_auprc"] - a4a["full"]["macro_auprc"]
        row["delta_hard_structured_vs_A4a"] = row["hard_structured_avg_macro_auprc"] - average_metric(
            a4a, HARD_STRUCTURED_PATTERNS, "macro_auprc"
        )
        row["delta_hard_overall_vs_A4a"] = row["hard_overall_avg_macro_auprc"] - average_metric(
            a4a, HARD_OVERALL_PATTERNS, "macro_auprc"
        )
        rows.append(row)
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(header for header, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    colspec = "l" + "r" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines.append(" & ".join(header.replace("_", "\\_") for header, _ in columns) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(fmt(row.get(key)) for _, key in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_table_bundle(out_dir: Path, stem: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    fieldnames = [key for _, key in columns]
    write_csv(out_dir / f"{stem}.csv", [{key: row.get(key) for key in fieldnames} for row in rows], fieldnames)
    (out_dir / f"{stem}.md").write_text(markdown_table(rows, columns), encoding="utf-8")
    (out_dir / f"{stem}.tex").write_text(latex_table(rows, columns), encoding="utf-8")


def build_table1_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "A0_full_no_masking",
        "A1_random_dropout",
        "A2_structured_masking",
        "A4a_subclass_auxiliary",
        "A5_lite_confidence_consistency_0p05",
    }
    return [row for row in summary_rows if row["method_id"] in keep]


def build_table3_rows() -> list[dict[str, Any]]:
    method_ids = (
        "A1_random_dropout",
        "A4a_subclass_auxiliary",
        "A5_lite_confidence_consistency_0p05",
    )
    methods = {m.method_id: m for m in METHOD_REGISTRY}
    rows: list[dict[str, Any]] = []
    for method_id in method_ids:
        method = methods[method_id]
        store = read_pattern_metrics(method, "mean_fill")
        blocks = {
            "full": ("full",),
            "hard_overall_average": HARD_OVERALL_PATTERNS,
            "hard_structured_average": HARD_STRUCTURED_PATTERNS,
        }
        for block, patterns in blocks.items():
            row = {"method_id": method_id, "method_display_name": method.display_name, "block": block}
            for label in LABELS:
                row[f"{label}_auprc"] = average_per_class(store, patterns, label)
            rows.append(row)
    return rows


def build_figure2_rows() -> list[dict[str, Any]]:
    method_ids = (
        "A0_full_no_masking",
        "A1_random_dropout",
        "A4a_subclass_auxiliary",
        "A5_lite_confidence_consistency_0p05",
    )
    methods = {m.method_id: m for m in METHOD_REGISTRY}
    rows = []
    for method_id in method_ids:
        method = methods[method_id]
        store = read_pattern_metrics(method, "mean_fill")
        for order, pattern in enumerate(("full", "random-1", "random-3", "random-6")):
            metrics = store[pattern]
            rows.append(
                {
                    "method_id": method_id,
                    "method_display_name": method.display_name,
                    "pattern": pattern,
                    "pattern_order": order,
                    "macro_auprc": metrics["macro_auprc"],
                    "macro_auroc": metrics["macro_auroc"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
    return rows


def build_heatmap_rows() -> list[dict[str, Any]]:
    methods = {m.method_id: m for m in METHOD_REGISTRY}
    comparisons = (
        ("A4a_minus_A1", "A4a_subclass_auxiliary", "A1_random_dropout"),
        ("A5_lite_minus_A4a", "A5_lite_confidence_consistency_0p05", "A4a_subclass_auxiliary"),
        ("A4a_minus_A2", "A4a_subclass_auxiliary", "A2_structured_masking"),
    )
    rows = []
    for comparison, left_id, right_id in comparisons:
        left = read_pattern_metrics(methods[left_id], "mean_fill")
        right = read_pattern_metrics(methods[right_id], "mean_fill")
        for pattern in HARD_OVERALL_PATTERNS:
            for label in LABELS:
                rows.append(
                    {
                        "comparison": comparison,
                        "left_method_id": left_id,
                        "right_method_id": right_id,
                        "pattern": pattern,
                        "label": label,
                        "value_auprc_delta": left[pattern]["per_class_auprc"][label]
                        - right[pattern]["per_class_auprc"][label],
                        "left_auprc": left[pattern]["per_class_auprc"][label],
                        "right_auprc": right[pattern]["per_class_auprc"][label],
                        "source_artifact_type": "per_pattern_metrics_json",
                    }
                )
    return rows


def make_decision(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["method_id"]: row for row in summary_rows}
    a4a = by_id["A4a_subclass_auxiliary"]
    a5_lite = by_id["A5_lite_confidence_consistency_0p05"]
    a5 = by_id["A5_confidence_consistency_0p10"]
    balanced = "A5_lite_confidence_consistency_0p05"
    if a5["full_macro_auprc"] > a5_lite["full_macro_auprc"] and a5["hard_overall_avg_macro_auprc"] >= a5_lite[
        "hard_overall_avg_macro_auprc"
    ]:
        balanced = "A5_confidence_consistency_0p10"
    return {
        "continue_model_structure_experiments": False,
        "stop_model_structure_experiments": True,
        "final_robustness_candidate": "A4a_subclass_auxiliary",
        "final_balanced_or_full_preserving_candidate": balanced,
        "negative_or_weak_ablation": [
            "A4b_hierarchy_loss",
            "A5_confidence_consistency_0p10 as hard-robustness booster",
            "A5_lite_confidence_consistency_0p05 as hard-robustness booster",
        ],
        "hierarchy_loss_as_main_component": False,
        "consistency_as_main_robustness_method": False,
        "hierarchy_claim_should_be_downgraded": True,
        "recommended_title": "Pattern-aware Lead Masking for Missing-lead Robust ECG Diagnosis",
        "recommended_main_claim": (
            "Pattern-aware structured masking with subclass auxiliary supervision improves hard missing-lead robustness "
            "over random lead dropout, while confidence consistency mainly preserves full-lead and HYP performance."
        ),
        "bibm_supporting_results": [
            "A4a has the strongest hard structured average Macro AUPRC.",
            "A4a has the strongest hard overall average Macro AUPRC.",
            "A4a improves hard structured and hard overall averages over A1 random dropout.",
            "A5-lite improves full Macro AUPRC and HYP full AUPRC but does not improve hard robustness over A4a.",
        ],
        "current_risks": [
            "Single-seed results need paired bootstrap confidence intervals.",
            "Per-sample prediction artifacts are not yet complete for bootstrap and calibration.",
            "Hierarchy loss is weak/negative, so hierarchical wording should be softened.",
            "Consistency should not be claimed as the primary robustness module.",
        ],
        "next_step": "bootstrap_paired_ci_then_calibration_audit",
        "need_more_lambda_cons": False,
        "need_more_hierarchy": False,
        "need_external_dataset": False,
        "a4a_hard_structured_avg_macro_auprc": a4a["hard_structured_avg_macro_auprc"],
        "a4a_hard_overall_avg_macro_auprc": a4a["hard_overall_avg_macro_auprc"],
        "a5_lite_full_delta_vs_a4a_macro_auprc": a5_lite["delta_full_vs_A4a"],
        "a5_lite_hard_overall_delta_vs_a4a_macro_auprc": a5_lite["delta_hard_overall_vs_A4a"],
    }


def script_supports_save_predictions(script_path: str) -> bool:
    path = Path(script_path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "--save-predictions" in text and "--predictions-dir" in text


def audit_prediction_artifacts(out_dir: Path) -> dict[str, Any]:
    method_map = {m.method_id: m for m in METHOD_REGISTRY}
    prediction_root = out_dir / "predictions"
    existing: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for method_id in KEY_PREDICTION_METHODS:
        method = method_map[method_id]
        for pattern in KEY_PREDICTION_PATTERNS:
            path = prediction_root / method_id / "mean_fill" / f"{PATTERN_SLUGS[pattern]}.csv"
            record = {
                "method_id": method_id,
                "pattern": pattern,
                "fill_mode": "mean_fill",
                "expected_path": str(path),
            }
            if not path.exists():
                missing.append(record)
                continue
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                columns = set(reader.fieldnames or [])
            missing_columns = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in columns]
            if missing_columns:
                invalid.append({**record, "missing_columns": missing_columns})
            else:
                existing.append(record)
    support = {method.method_id: script_supports_save_predictions(method.eval_script) for method in METHOD_REGISTRY}
    commands_after_cli_added = []
    for method_id in KEY_PREDICTION_METHODS:
        method = method_map[method_id]
        commands_after_cli_added.append(
            " ".join(
                [
                    "python3",
                    method.eval_script,
                    "--checkpoint",
                    str(method.output_dir / "best_model.pt"),
                    "--config",
                    method.config,
                    "--output-dir",
                    str(method.output_dir),
                    "--save-predictions",
                    "--predictions-dir",
                    str(prediction_root / method_id),
                ]
            )
        )
    return {
        "prediction_root": str(prediction_root),
        "artifact_complete": not missing and not invalid,
        "required_methods": list(KEY_PREDICTION_METHODS),
        "required_patterns": list(KEY_PREDICTION_PATTERNS),
        "required_fill_modes_this_round": ["mean_fill"],
        "required_columns": list(REQUIRED_PREDICTION_COLUMNS),
        "existing_count": len(existing),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "existing": existing,
        "missing": missing,
        "invalid": invalid,
        "evaluation_scripts_support_save_predictions": support,
        "all_key_scripts_support_save_predictions": all(support[method_id] for method_id in KEY_PREDICTION_METHODS),
        "required_script_updates": [
            "src/hlm_ecg/evaluation/evaluate_patterns.py: return ecg_id, targets, logits, probabilities and write per-pattern CSV.",
            "scripts/evaluate_full_baseline_patterns.py: add --save-predictions and --predictions-dir.",
            "scripts/evaluate_random_dropout_patterns.py: add --save-predictions and --predictions-dir.",
            "scripts/evaluate_structured_masking_patterns.py: add --save-predictions and --predictions-dir.",
            "scripts/evaluate_subclass_auxiliary_patterns.py: add --save-predictions and --predictions-dir.",
            "scripts/evaluate_confidence_consistency_patterns.py: add --save-predictions and --predictions-dir.",
        ],
        "evaluation_only_rerun_commands_after_cli_added": commands_after_cli_added,
        "needs_slurm": True,
        "note": "Prediction artifacts are needed for paired bootstrap and calibration. Do not retrain; rerun evaluation only after adding save-predictions support.",
    }


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        ("Method", "method_id"),
        ("Full AUPRC", "full_macro_auprc"),
        ("All missing", "avg_all_missing_macro_auprc"),
        ("Hard structured", "hard_structured_avg_macro_auprc"),
        ("Hard overall", "hard_overall_avg_macro_auprc"),
        ("Role", "recommended_role"),
    ]
    text = "# A0-A5-lite Summary\n\n" + markdown_table(rows, columns)
    path.write_text(text, encoding="utf-8")


def write_decision_markdown(path: Path, decision: dict[str, Any]) -> None:
    lines = [
        "# Final Candidate Decision",
        "",
        f"- 是否继续结构实验：`{decision['continue_model_structure_experiments']}`",
        f"- 是否停止模型结构实验：`{decision['stop_model_structure_experiments']}`",
        f"- final robustness candidate：`{decision['final_robustness_candidate']}`",
        f"- final balanced/full-preserving candidate：`{decision['final_balanced_or_full_preserving_candidate']}`",
        f"- hierarchy loss 是否作为主组件：`{decision['hierarchy_loss_as_main_component']}`",
        f"- consistency 是否作为主 robustness method：`{decision['consistency_as_main_robustness_method']}`",
        f"- 是否需要弱化 Hierarchical claim：`{decision['hierarchy_claim_should_be_downgraded']}`",
        f"- 推荐标题：{decision['recommended_title']}",
        f"- 推荐主 claim：{decision['recommended_main_claim']}",
        f"- 下一步：`{decision['next_step']}`",
        f"- 是否需要继续调 lambda_cons：`{decision['need_more_lambda_cons']}`",
        f"- 是否需要继续 hierarchy：`{decision['need_more_hierarchy']}`",
        f"- 是否需要 external dataset：`{decision['need_external_dataset']}`",
        "",
        "## Negative / Weak Ablations",
        "",
        *[f"- {item}" for item in decision["negative_or_weak_ablation"]],
        "",
        "## Risks",
        "",
        *[f"- {item}" for item in decision["current_risks"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prediction_audit_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Prediction Artifact Audit",
        "",
        f"- artifact complete：`{audit['artifact_complete']}`",
        f"- existing count：`{audit['existing_count']}`",
        f"- missing count：`{audit['missing_count']}`",
        f"- invalid count：`{audit['invalid_count']}`",
        f"- prediction root：`{audit['prediction_root']}`",
        "",
        "## Evaluation Script Support",
        "",
    ]
    for method_id, supported in audit["evaluation_scripts_support_save_predictions"].items():
        lines.append(f"- {method_id}: `{supported}`")
    lines.extend(["", "## Missing Method-patterns", ""])
    for item in audit["missing"][:60]:
        lines.append(f"- {item['method_id']} / {item['fill_mode']} / {item['pattern']} -> `{item['expected_path']}`")
    if len(audit["missing"]) > 60:
        lines.append(f"- ... plus {len(audit['missing']) - 60} more")
    lines.extend(["", "## Required Evaluation-only Commands After CLI Support", ""])
    for command in audit["evaluation_only_rerun_commands_after_cli_added"]:
        lines.append(f"```bash\n{command}\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_lock_markdown(path: Path, decision: dict[str, Any], summary_rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    by_id = {row["method_id"]: row for row in summary_rows}
    a4a = by_id["A4a_subclass_auxiliary"]
    a1 = by_id["A1_random_dropout"]
    a5_lite = by_id["A5_lite_confidence_consistency_0p05"]
    lines = [
        "# Week 3 Step 4A Final Results Lock",
        "",
        "本报告只汇总已有 A0-A5-lite 单 seed 结果；没有训练新模型，没有提交 Slurm job，没有调参。",
        "",
        "## Locked Candidates",
        "",
        f"- final robustness candidate：`{decision['final_robustness_candidate']}`",
        f"- final balanced/full-preserving candidate：`{decision['final_balanced_or_full_preserving_candidate']}`",
        "- hierarchy loss：negative / weak ablation，不作为主方法组件。",
        "- confidence consistency：full/HYP preservation ablation，不作为 hard missing robustness booster。",
        "",
        "## Key Numbers",
        "",
        f"- A4a hard structured avg Macro AUPRC：`{a4a['hard_structured_avg_macro_auprc']:.6f}`",
        f"- A4a hard overall avg Macro AUPRC：`{a4a['hard_overall_avg_macro_auprc']:.6f}`",
        f"- A4a vs A1 hard structured delta：`{a4a['hard_structured_avg_macro_auprc'] - a1['hard_structured_avg_macro_auprc']:.6f}`",
        f"- A4a vs A1 hard overall delta：`{a4a['hard_overall_avg_macro_auprc'] - a1['hard_overall_avg_macro_auprc']:.6f}`",
        f"- A5-lite full Macro AUPRC：`{a5_lite['full_macro_auprc']:.6f}`",
        f"- A5-lite hard overall delta vs A4a：`{a5_lite['delta_hard_overall_vs_A4a']:.6f}`",
        "",
        "## Prediction Audit",
        "",
        f"- prediction artifacts complete：`{audit['artifact_complete']}`",
        f"- missing prediction artifacts：`{audit['missing_count']}`",
        "- 当前已有 per-pattern per-class metrics，足够生成 Table 3 和 heatmap 数据源；bootstrap/calibration 仍需要 per-sample predictions。",
        "",
        "## Next Step",
        "",
        "建议下一步先补 evaluation-only prediction saving，然后进入 paired bootstrap / CI；之后再做 calibration audit。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def lock_results(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = build_summary_rows()
    all_mean = build_all_pattern_rows("mean_fill")
    all_zero = build_all_pattern_rows("zero_fill")
    decision = make_decision(summary_rows)
    prediction_audit = audit_prediction_artifacts(out_dir)
    read_paths = {method.method_id: discover_read_paths(method) for method in METHOD_REGISTRY}

    write_csv(out_dir / "all_methods_all_patterns_mean_fill.csv", all_mean)
    write_csv(out_dir / "all_methods_all_patterns_zero_fill.csv", all_zero)
    write_csv(out_dir / "all_methods_summary.csv", summary_rows)
    write_json(out_dir / "all_methods_summary.json", {"rows": summary_rows, "read_paths": read_paths})
    write_summary_markdown(out_dir / "all_methods_summary.md", summary_rows)

    table1_rows = build_table1_rows(summary_rows)
    table1_cols = [
        ("method", "method_id"),
        ("full", "full_macro_auprc"),
        ("random-1", "random_1_macro_auprc"),
        ("random-3", "random_3_macro_auprc"),
        ("random-6", "random_6_macro_auprc"),
        ("limb-only", "limb_only_precordial_missing_macro_auprc"),
        ("precordial-only", "precordial_only_limb_missing_macro_auprc"),
        ("V1-V3", "V1_V3_missing_macro_auprc"),
        ("V4-V6", "V4_V6_missing_macro_auprc"),
        ("hard structured", "hard_structured_avg_macro_auprc"),
        ("hard overall", "hard_overall_avg_macro_auprc"),
    ]
    write_table_bundle(out_dir, "table1_main_robustness", table1_rows, table1_cols)

    table2_cols = [
        ("method", "method_id"),
        ("full AUROC", "full_macro_auroc"),
        ("full AUPRC", "full_macro_auprc"),
        ("all missing", "avg_all_missing_macro_auprc"),
        ("hard structured", "hard_structured_avg_macro_auprc"),
        ("hard overall", "hard_overall_avg_macro_auprc"),
        ("delta hard overall vs A1", "delta_hard_overall_vs_A1_random_dropout"),
        ("delta full vs A1", "delta_full_vs_A1_random_dropout"),
        ("role", "recommended_role"),
    ]
    write_table_bundle(out_dir, "table2_ablation", summary_rows, table2_cols)

    table3_rows = build_table3_rows()
    table3_cols = [("method", "method_id"), ("block", "block"), *[(label, f"{label}_auprc") for label in LABELS]]
    write_table_bundle(out_dir, "table3_per_class", table3_rows, table3_cols)
    write_csv(out_dir / "per_class_summary_full.csv", [row for row in table3_rows if row["block"] == "full"])
    write_csv(
        out_dir / "per_class_summary_hard_overall.csv",
        [row for row in table3_rows if row["block"] == "hard_overall_average"],
    )
    write_csv(
        out_dir / "per_class_summary_hard_structured.csv",
        [row for row in table3_rows if row["block"] == "hard_structured_average"],
    )

    figure2_rows = build_figure2_rows()
    write_csv(out_dir / "figure2_degradation_curve_data.csv", figure2_rows)
    write_json(out_dir / "figure2_degradation_curve_data.json", {"rows": figure2_rows})

    heatmap_rows = build_heatmap_rows()
    write_csv(out_dir / "figure3_class_by_pattern_heatmap_data.csv", heatmap_rows)
    write_json(
        out_dir / "figure3_class_by_pattern_heatmap_data.json",
        {
            "default_comparison": "A4a_minus_A1",
            "rows": heatmap_rows,
            "note": "Values are computed from existing per-pattern per-class AUPRC metrics, not per-sample predictions.",
        },
    )

    write_json(out_dir / "final_candidate_decision.json", decision)
    write_decision_markdown(out_dir / "final_candidate_decision.md", decision)
    write_json(out_dir / "prediction_artifact_audit.json", prediction_audit)
    write_prediction_audit_markdown(out_dir / "prediction_artifact_audit.md", prediction_audit)
    final_lock = {
        "method_registry": [
            {
                "method_id": method.method_id,
                "method_display_name": method.display_name,
                "output_dir": str(method.output_dir),
                "recommended_role": method.recommended_role,
            }
            for method in METHOD_REGISTRY
        ],
        "patterns": list(PATTERNS),
        "averages": {
            "avg_all_missing": list(ALL_MISSING_PATTERNS),
            "hard_structured_avg": list(HARD_STRUCTURED_PATTERNS),
            "hard_overall_avg": list(HARD_OVERALL_PATTERNS),
        },
        "summary_rows": summary_rows,
        "decision": decision,
        "prediction_artifact_audit": prediction_audit,
        "read_paths": read_paths,
        "records500_used_anywhere": any(row["records500_used"] for row in summary_rows),
    }
    write_json(out_dir / "final_results_lock.json", final_lock)
    write_final_lock_markdown(out_dir / "final_results_lock.md", decision, summary_rows, prediction_audit)
    return final_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock final A0-A5-lite HLM-ECG result summaries.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/week3_results_lock"))
    args = parser.parse_args()
    result = lock_results(args.out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "final_robustness_candidate": result["decision"]["final_robustness_candidate"],
                "final_balanced_or_full_preserving_candidate": result["decision"][
                    "final_balanced_or_full_preserving_candidate"
                ],
                "prediction_artifacts_complete": result["prediction_artifact_audit"]["artifact_complete"],
                "records500_used_anywhere": result["records500_used_anywhere"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
