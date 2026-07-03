#!/usr/bin/env python3
"""Run calibration audit from saved HLM-ECG prediction CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.calibration.calibration_metrics import compute_calibration_metrics
from hlm_ecg.calibration.temperature_scaling import apply_temperatures, fit_classwise_temperatures
from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.metrics import sigmoid
from hlm_ecg.evaluation.prediction_artifacts import safe_pattern_name
from hlm_ecg.statistics.bootstrap import (
    AGGREGATES,
    HARD_OVERALL_PATTERNS,
    HARD_STRUCTURED_PATTERNS,
    METHODS,
    PATTERNS,
    PredictionData,
    load_prediction_data,
)

PROTOCOL_UNCALIBRATED = "uncalibrated"
PROTOCOL_FULL_VAL = "full_val_classwise_ts"
PROTOCOL_PATTERN_WISE = "pattern_wise_classwise_ts"
PROTOCOLS = (PROTOCOL_UNCALIBRATED, PROTOCOL_FULL_VAL, PROTOCOL_PATTERN_WISE)

PRIMARY_TABLE_METHODS = (
    "A1_random_dropout",
    "A2_structured_masking",
    "A4a_subclass_auxiliary",
    "A5_lite_confidence_consistency_0p05",
)

FIGURE_METHODS = ("A1_random_dropout", "A4a_subclass_auxiliary")
FIGURE_PATTERNS = ("full", "random-6", "limb-only / precordial-missing")
FIGURE_CLASSES = ("MI", "CD", "HYP")
FIGURE_PROTOCOLS = (PROTOCOL_UNCALIBRATED, PROTOCOL_FULL_VAL)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def tex_fmt(value: Any, digits: int = 4) -> str:
    return fmt(value, digits=digits).replace("_", "\\_")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(header for header, _ in columns) + " |"]
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
        lines.append(" & ".join(tex_fmt(row.get(key)) for _, key in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def prediction_csv_path(predictions_dir: Path, method_id: str, fill_mode: str, split: str, pattern: str) -> Path:
    return predictions_dir / method_id / fill_mode / split / f"{safe_pattern_name(pattern)}.csv"


def load_raw_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def temperature_parameter_rows(
    *,
    method_id: str,
    protocol: str,
    pattern_used_for_fit: str,
    results: Sequence[Any],
) -> list[dict[str, Any]]:
    return [
        {
            "method_id": method_id,
            "calibration_protocol": protocol,
            "pattern_used_for_fit": pattern_used_for_fit,
            "class": result.class_name,
            "temperature": result.temperature,
            "val_nll_before": result.val_nll_before,
            "val_nll_after": result.val_nll_after,
            "converged": result.converged,
            "n_val_samples": result.n_val_samples,
            "label_prevalence": result.label_prevalence,
        }
        for result in results
    ]


def uncalibrated_temperature_rows(method_id: str, val_full: PredictionData) -> list[dict[str, Any]]:
    rows = []
    probs = sigmoid(val_full.logits)
    for idx, label in enumerate(LABEL_ORDER):
        targets = val_full.targets[:, idx]
        nll = -np.mean(
            targets * np.log(np.clip(probs[:, idx], 1e-12, 1.0 - 1e-12))
            + (1 - targets) * np.log(np.clip(1.0 - probs[:, idx], 1e-12, 1.0))
        )
        rows.append(
            {
                "method_id": method_id,
                "calibration_protocol": PROTOCOL_UNCALIBRATED,
                "pattern_used_for_fit": "none",
                "class": label,
                "temperature": 1.0,
                "val_nll_before": float(nll),
                "val_nll_after": float(nll),
                "converged": True,
                "n_val_samples": int(targets.shape[0]),
                "label_prevalence": float(targets.mean()),
            }
        )
    return rows


def fit_temperature_protocols(
    val_data: Mapping[str, Mapping[str, PredictionData]],
    *,
    methods: Sequence[str],
    patterns: Sequence[str],
) -> tuple[dict[tuple[str, str, str], np.ndarray], list[dict[str, Any]]]:
    temps: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for method_id in methods:
        for pattern in patterns:
            temps[(method_id, PROTOCOL_UNCALIBRATED, pattern)] = np.ones(len(LABEL_ORDER), dtype=np.float64)
        rows.extend(uncalibrated_temperature_rows(method_id, val_data[method_id]["full"]))

        full_temps, full_results = fit_classwise_temperatures(
            val_data[method_id]["full"].logits,
            val_data[method_id]["full"].targets,
        )
        rows.extend(
            temperature_parameter_rows(
                method_id=method_id,
                protocol=PROTOCOL_FULL_VAL,
                pattern_used_for_fit="full",
                results=full_results,
            )
        )
        for pattern in patterns:
            temps[(method_id, PROTOCOL_FULL_VAL, pattern)] = full_temps

        for pattern in patterns:
            pattern_temps, pattern_results = fit_classwise_temperatures(
                val_data[method_id][pattern].logits,
                val_data[method_id][pattern].targets,
            )
            temps[(method_id, PROTOCOL_PATTERN_WISE, pattern)] = pattern_temps
            rows.extend(
                temperature_parameter_rows(
                    method_id=method_id,
                    protocol=PROTOCOL_PATTERN_WISE,
                    pattern_used_for_fit=pattern,
                    results=pattern_results,
                )
            )
    return temps, rows


def metric_row(
    *,
    method_id: str,
    protocol: str,
    pattern: str,
    split: str,
    fill_mode: str,
    data: PredictionData,
    temperatures: np.ndarray,
    n_bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    calibrated_logits = apply_temperatures(data.logits, temperatures)
    calibrated_probs = sigmoid(calibrated_logits)
    metrics = compute_calibration_metrics(
        targets=data.targets,
        probs=calibrated_probs,
        n_bins=n_bins,
    )
    row: dict[str, Any] = {
        "method_id": method_id,
        "calibration_protocol": protocol,
        "pattern": pattern,
        "split": split,
        "macro_ece": metrics["macro_ece"],
        "macro_brier": metrics["macro_brier"],
        "macro_nll": metrics["macro_nll"],
        "sample_label_bce": metrics["sample_label_bce"],
        "n_samples": metrics["n_samples"],
        "n_bins": metrics["n_bins"],
        "fill_mode": fill_mode,
    }
    for label in LABEL_ORDER:
        row[f"{label}_ece"] = metrics["per_class_ece"][label]
        row[f"{label}_brier"] = metrics["per_class_brier"][label]
        row[f"{label}_nll"] = metrics["per_class_nll"][label]
    reliability = []
    for bin_row in metrics["reliability_rows"]:
        reliability.append(
            {
                "method_id": method_id,
                "calibration_protocol": protocol,
                "pattern": pattern,
                "split": split,
                "fill_mode": fill_mode,
                **bin_row,
            }
        )
    return row, reliability, calibrated_probs


def save_calibrated_predictions(
    *,
    out_dir: Path,
    predictions_dir: Path,
    method_id: str,
    protocol: str,
    pattern: str,
    split: str,
    fill_mode: str,
    temperatures: np.ndarray,
    calibrated_probs: np.ndarray,
) -> None:
    source = load_raw_prediction_rows(prediction_csv_path(predictions_dir, method_id, fill_mode, split, pattern))
    path = out_dir / "calibrated_predictions" / protocol / method_id / split / f"{safe_pattern_name(pattern)}.csv"
    rows: list[dict[str, Any]] = []
    for idx, source_row in enumerate(source):
        row: dict[str, Any] = {
            "ecg_id": source_row["ecg_id"],
            "method_id": method_id,
            "pattern": pattern,
            "split": split,
            "fill_mode": fill_mode,
            "calibration_protocol": protocol,
        }
        for label_idx, label in enumerate(LABEL_ORDER):
            row[f"y_true_{label}"] = source_row[f"y_true_{label}"]
            row[f"logit_{label}"] = source_row[f"logit_{label}"]
            row[f"original_prob_{label}"] = source_row[f"prob_{label}"]
            row[f"calibrated_prob_{label}"] = float(calibrated_probs[idx, label_idx])
            row[f"temperature_{label}"] = float(temperatures[label_idx])
        rows.append(row)
    write_csv(path, rows)


def aggregate_rows(pattern_rows: Sequence[Mapping[str, Any]], members: Sequence[str]) -> dict[str, float]:
    selected = [row for row in pattern_rows if row["pattern"] in members]
    if len(selected) != len(members):
        return {
            "macro_ece": float("nan"),
            "macro_brier": float("nan"),
            "macro_nll": float("nan"),
            "sample_label_bce": float("nan"),
        }
    return {
        "macro_ece": float(np.mean([float(row["macro_ece"]) for row in selected])),
        "macro_brier": float(np.mean([float(row["macro_brier"]) for row in selected])),
        "macro_nll": float(np.mean([float(row["macro_nll"]) for row in selected])),
        "sample_label_bce": float(np.mean([float(row["sample_label_bce"]) for row in selected])),
    }


def build_aggregate_summary(metric_rows: list[dict[str, Any]], *, methods: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for method_id in methods:
        for protocol in PROTOCOLS:
            selected = [row for row in metric_rows if row["method_id"] == method_id and row["calibration_protocol"] == protocol]
            full = next(row for row in selected if row["pattern"] == "full")
            all_missing = aggregate_rows(selected, PATTERNS[1:])
            hard_structured = aggregate_rows(selected, HARD_STRUCTURED_PATTERNS)
            hard_overall = aggregate_rows(selected, HARD_OVERALL_PATTERNS)
            row = {
                "method_id": method_id,
                "calibration_protocol": protocol,
                "full_macro_ece": full["macro_ece"],
                "full_macro_brier": full["macro_brier"],
                "full_macro_nll": full["macro_nll"],
                "avg_all_missing_macro_ece": all_missing["macro_ece"],
                "avg_all_missing_macro_brier": all_missing["macro_brier"],
                "avg_all_missing_macro_nll": all_missing["macro_nll"],
                "hard_structured_avg_macro_ece": hard_structured["macro_ece"],
                "hard_structured_avg_macro_brier": hard_structured["macro_brier"],
                "hard_structured_avg_macro_nll": hard_structured["macro_nll"],
                "hard_overall_avg_macro_ece": hard_overall["macro_ece"],
                "hard_overall_avg_macro_brier": hard_overall["macro_brier"],
                "hard_overall_avg_macro_nll": hard_overall["macro_nll"],
                "recommended_calibration_role": recommended_role(method_id, protocol),
            }
            by_key[(method_id, protocol)] = row
            rows.append(row)
    for protocol in PROTOCOLS:
        a1 = by_key.get(("A1_random_dropout", protocol))
        a4a = by_key.get(("A4a_subclass_auxiliary", protocol))
        for row in rows:
            if row["calibration_protocol"] != protocol:
                continue
            for metric in ("ece", "brier", "nll"):
                key = f"hard_overall_avg_macro_{metric}"
                row[f"delta_hard_overall_{metric}_vs_A1"] = (
                    float(row[key]) - float(a1[key]) if a1 is not None else float("nan")
                )
                row[f"delta_hard_overall_{metric}_vs_A4a"] = (
                    float(row[key]) - float(a4a[key]) if a4a is not None else float("nan")
                )
    return rows


def recommended_role(method_id: str, protocol: str) -> str:
    if method_id == "A4a_subclass_auxiliary":
        return "final robustness candidate calibration audit"
    if method_id == "A5_lite_confidence_consistency_0p05":
        return "balanced/full-preserving calibration audit"
    if method_id == "A1_random_dropout":
        return "strong baseline calibration audit"
    if method_id == "A2_structured_masking":
        return "structured masking calibration audit"
    return "context baseline calibration audit"


def write_temperature_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "calibration_temperature_parameters.csv", rows)
    write_json(out_dir / "calibration_temperature_parameters.json", {"rows": rows})
    columns = [
        ("method", "method_id"),
        ("protocol", "calibration_protocol"),
        ("fit_pattern", "pattern_used_for_fit"),
        ("class", "class"),
        ("T", "temperature"),
        ("val_nll_before", "val_nll_before"),
        ("val_nll_after", "val_nll_after"),
        ("converged", "converged"),
    ]
    (out_dir / "calibration_temperature_parameters.md").write_text(
        "# Calibration Temperature Parameters\n\n" + markdown_table(rows, columns),
        encoding="utf-8",
    )


def write_metric_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "calibration_metrics_by_pattern.csv", rows)
    write_json(out_dir / "calibration_metrics_by_pattern.json", {"rows": rows})
    columns = [
        ("method", "method_id"),
        ("protocol", "calibration_protocol"),
        ("pattern", "pattern"),
        ("ECE", "macro_ece"),
        ("Brier", "macro_brier"),
        ("NLL", "macro_nll"),
    ]
    (out_dir / "calibration_metrics_by_pattern.md").write_text(
        "# Calibration Metrics by Pattern\n\n" + markdown_table(rows, columns),
        encoding="utf-8",
    )


def write_aggregate_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "calibration_aggregate_summary.csv", rows)
    write_json(out_dir / "calibration_aggregate_summary.json", {"rows": rows})
    columns = [
        ("method", "method_id"),
        ("protocol", "calibration_protocol"),
        ("full_ECE", "full_macro_ece"),
        ("hard_overall_ECE", "hard_overall_avg_macro_ece"),
        ("hard_overall_Brier", "hard_overall_avg_macro_brier"),
        ("hard_overall_NLL", "hard_overall_avg_macro_nll"),
        ("role", "recommended_calibration_role"),
    ]
    (out_dir / "calibration_aggregate_summary.md").write_text(
        "# Calibration Aggregate Summary\n\n" + markdown_table(rows, columns),
        encoding="utf-8",
    )


def build_table_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["method_id"], row["calibration_protocol"]): row for row in summary_rows}
    rows: list[dict[str, Any]] = []
    for method_id in PRIMARY_TABLE_METHODS:
        for variant, protocol in (
            ("uncalibrated", PROTOCOL_UNCALIBRATED),
            ("full_val_ts", PROTOCOL_FULL_VAL),
        ):
            src = by_key[(method_id, protocol)]
            rows.append(table_row_from_summary(method_id, variant, src))
        uncal = by_key[(method_id, PROTOCOL_UNCALIBRATED)]
        ts = by_key[(method_id, PROTOCOL_FULL_VAL)]
        delta = {key: float(ts[key]) - float(uncal[key]) for key in ts if key.endswith(("macro_ece", "macro_brier", "macro_nll"))}
        rows.append(table_row_from_summary(method_id, "delta_after_full_val_ts", delta))
    return rows


def table_row_from_summary(method_id: str, variant: str, src: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "table_variant": variant,
        "full_ECE": src["full_macro_ece"],
        "hard_structured_ECE": src["hard_structured_avg_macro_ece"],
        "hard_overall_ECE": src["hard_overall_avg_macro_ece"],
        "full_Brier": src["full_macro_brier"],
        "hard_structured_Brier": src["hard_structured_avg_macro_brier"],
        "hard_overall_Brier": src["hard_overall_avg_macro_brier"],
        "full_NLL": src["full_macro_nll"],
        "hard_structured_NLL": src["hard_structured_avg_macro_nll"],
        "hard_overall_NLL": src["hard_overall_avg_macro_nll"],
    }


def write_table_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "table_calibration_main.csv", rows)
    columns = [(key, key) for key in rows[0]]
    (out_dir / "table_calibration_main.md").write_text(markdown_table(rows, columns), encoding="utf-8")
    (out_dir / "table_calibration_main.tex").write_text(latex_table(rows, columns), encoding="utf-8")


def write_reliability_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "reliability_curve_data.csv", rows)
    write_json(out_dir / "reliability_curve_data.json", {"rows": rows})
    figure_rows = [
        row
        for row in rows
        if row["method_id"] in FIGURE_METHODS
        and row["pattern"] in FIGURE_PATTERNS
        and row["class"] in FIGURE_CLASSES
        and row["calibration_protocol"] in FIGURE_PROTOCOLS
    ]
    write_csv(out_dir / "figure_calibration_reliability_data.csv", figure_rows)
    write_json(out_dir / "figure_calibration_reliability_data.json", {"rows": figure_rows})


def make_decision(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(row["method_id"], row["calibration_protocol"]): row for row in summary_rows}

    def lower(method_a: str, method_b: str, protocol: str, metric: str, prefix: str = "hard_overall_avg") -> float:
        return float(by_key[(method_a, protocol)][f"{prefix}_macro_{metric}"]) - float(
            by_key[(method_b, protocol)][f"{prefix}_macro_{metric}"]
        )

    a4a_vs_a1 = {
        metric: lower("A4a_subclass_auxiliary", "A1_random_dropout", PROTOCOL_FULL_VAL, metric)
        for metric in ("ece", "brier", "nll")
    }
    a4a_ts_delta = {
        metric: float(by_key[("A4a_subclass_auxiliary", PROTOCOL_FULL_VAL)][f"hard_overall_avg_macro_{metric}"])
        - float(by_key[("A4a_subclass_auxiliary", PROTOCOL_UNCALIBRATED)][f"hard_overall_avg_macro_{metric}"])
        for metric in ("ece", "brier", "nll")
    }
    a4a_pattern_vs_full = {
        metric: float(by_key[("A4a_subclass_auxiliary", PROTOCOL_PATTERN_WISE)][f"hard_overall_avg_macro_{metric}"])
        - float(by_key[("A4a_subclass_auxiliary", PROTOCOL_FULL_VAL)][f"hard_overall_avg_macro_{metric}"])
        for metric in ("ece", "brier", "nll")
    }
    a5_vs_a4a_full = {
        metric: lower(
            "A5_lite_confidence_consistency_0p05",
            "A4a_subclass_auxiliary",
            PROTOCOL_FULL_VAL,
            metric,
            prefix="full",
        )
        for metric in ("ece", "brier", "nll")
    }
    a4a_better_than_a1 = sum(value < 0 for value in a4a_vs_a1.values())
    ts_improves_a4a = sum(value < 0 for value in a4a_ts_delta.values())
    pattern_wise_better = sum(value < 0 for value in a4a_pattern_vs_full.values())
    if a4a_better_than_a1 >= 2 and ts_improves_a4a >= 2:
        role = "appendix_or_short_main_auxiliary"
        claim = "A4a improves probability quality under hard missing shift after validation-only temperature scaling."
    elif a4a_better_than_a1 >= 2:
        role = "appendix_auxiliary"
        claim = "A4a calibration is better than A1 under hard missing shift for most metrics, but temperature scaling effects are mixed."
    else:
        role = "neutral_audit"
        claim = "Calibration analysis does not uniformly favor A4a; report it as an auxiliary audit."
    return {
        "a4a_vs_a1_hard_overall_full_val_ts": a4a_vs_a1,
        "a4a_full_val_ts_delta_vs_uncalibrated_hard_overall": a4a_ts_delta,
        "a4a_pattern_wise_ts_delta_vs_full_val_ts_hard_overall": a4a_pattern_vs_full,
        "a5_lite_vs_a4a_full_full_val_ts": a5_vs_a4a_full,
        "a4a_calibration_better_than_a1_metric_count": a4a_better_than_a1,
        "full_val_ts_improves_a4a_metric_count": ts_improves_a4a,
        "pattern_wise_ts_better_than_full_val_metric_count": pattern_wise_better,
        "calibration_supports_main_claim": bool(role == "appendix_or_short_main_auxiliary"),
        "recommended_calibration_placement": role,
        "recommended_statement": claim,
        "avoid_abstract_calibration_claim": bool(role != "appendix_or_short_main_auxiliary"),
        "records500_used": False,
    }


def write_decision_outputs(out_dir: Path, decision: Mapping[str, Any]) -> None:
    write_json(out_dir / "calibration_decision.json", dict(decision))
    lines = ["# Calibration Decision", ""]
    lines.append(f"- Recommended placement: `{decision['recommended_calibration_placement']}`")
    lines.append(f"- Recommended statement: {decision['recommended_statement']}")
    lines.append(f"- Avoid abstract calibration claim: `{decision['avoid_abstract_calibration_claim']}`")
    lines.append("")
    lines.append("## Key Deltas")
    lines.append("")
    for key in (
        "a4a_vs_a1_hard_overall_full_val_ts",
        "a4a_full_val_ts_delta_vs_uncalibrated_hard_overall",
        "a4a_pattern_wise_ts_delta_vs_full_val_ts_hard_overall",
        "a5_lite_vs_a4a_full_full_val_ts",
    ):
        if key in decision:
            lines.append(f"- `{key}`: `{decision[key]}`")
    (out_dir / "calibration_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if Path("data/ptb-xl/records500").exists():
        raise RuntimeError("records500 exists; refusing calibration audit")
    methods = tuple(args.methods)
    patterns = tuple(PATTERNS if args.patterns == ["all"] else args.patterns)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    val_data = load_prediction_data(
        args.predictions_dir,
        methods=methods,
        patterns=patterns,
        split="val",
        fill_mode=args.fill_mode,
    )
    test_data = load_prediction_data(
        args.predictions_dir,
        methods=methods,
        patterns=patterns,
        split="test",
        fill_mode=args.fill_mode,
    )
    temps, temp_rows = fit_temperature_protocols(val_data, methods=methods, patterns=patterns)
    metric_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for method_id in methods:
        for protocol in PROTOCOLS:
            for pattern in patterns:
                row, rel, calibrated_probs = metric_row(
                    method_id=method_id,
                    protocol=protocol,
                    pattern=pattern,
                    split="test",
                    fill_mode=args.fill_mode,
                    data=test_data[method_id][pattern],
                    temperatures=temps[(method_id, protocol, pattern)],
                    n_bins=args.n_bins,
                )
                metric_rows.append(row)
                reliability_rows.extend(rel)
                if args.save_calibrated_predictions:
                    save_calibrated_predictions(
                        out_dir=out_dir,
                        predictions_dir=args.predictions_dir,
                        method_id=method_id,
                        protocol=protocol,
                        pattern=pattern,
                        split="test",
                        fill_mode=args.fill_mode,
                        temperatures=temps[(method_id, protocol, pattern)],
                        calibrated_probs=calibrated_probs,
                    )

    summary_rows = build_aggregate_summary(metric_rows, methods=methods)
    table_rows = build_table_rows(summary_rows) if all(m in methods for m in PRIMARY_TABLE_METHODS) else []
    decision = make_decision(summary_rows) if all(m in methods for m in ("A1_random_dropout", "A4a_subclass_auxiliary", "A5_lite_confidence_consistency_0p05")) else {
        "recommended_calibration_placement": "smoke_only",
        "recommended_statement": "Smoke run completed.",
        "avoid_abstract_calibration_claim": True,
        "records500_used": False,
    }

    write_temperature_outputs(out_dir, temp_rows)
    write_metric_outputs(out_dir, metric_rows)
    write_aggregate_outputs(out_dir, summary_rows)
    write_reliability_outputs(out_dir, reliability_rows)
    if table_rows:
        write_table_outputs(out_dir, table_rows)
    write_decision_outputs(out_dir, decision)
    config = {
        "predictions_dir": str(args.predictions_dir),
        "out_dir": str(out_dir),
        "fill_mode": args.fill_mode,
        "n_bins": args.n_bins,
        "methods": list(methods),
        "patterns": list(patterns),
        "protocols": list(PROTOCOLS),
        "temperature_fit_split": "val",
        "evaluation_split": "test",
        "label_order": list(LABEL_ORDER),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records500_used": False,
        "smoke_test": bool(args.smoke_test),
        "calibrated_predictions_written": bool(args.save_calibrated_predictions),
    }
    write_json(out_dir / "calibration_run_config.json", config)
    result = {
        **config,
        "temperature_rows": len(temp_rows),
        "metric_rows": len(metric_rows),
        "reliability_rows": len(reliability_rows),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HLM-ECG calibration audit from saved prediction CSVs.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fill-mode", default="mean_fill")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--patterns", nargs="+", default=["all"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-calibrated-predictions", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
