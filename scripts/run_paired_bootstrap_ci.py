#!/usr/bin/env python3
"""Run patient-level paired bootstrap CI from saved HLM-ECG predictions."""

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

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.statistics.bootstrap import (
    AGGREGATES,
    ALL_MISSING_PATTERNS,
    COMPARISONS,
    HARD_OVERALL_PATTERNS,
    HARD_STRUCTURED_PATTERNS,
    METHODS,
    PATTERNS,
    PRIMARY_COMPARISONS,
    REPORT_PATTERNS,
    generate_patient_bootstrap_samples,
    load_prediction_data,
    metric_value,
    paired_delta_summary,
    patient_groups,
    sampled_indices_from_patients,
    summarize_distribution,
)

MACRO_METRICS = ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll", "macro_brier")
PER_CLASS_METRICS = tuple(
    f"per_class_{label}_{metric}" for label in LABEL_ORDER for metric in ("auroc", "auprc", "f1")
)
ALL_METRICS = (*MACRO_METRICS, *PER_CLASS_METRICS)

TABLE1_METHODS = (
    "A0_full_no_masking",
    "A1_random_dropout",
    "A2_structured_masking",
    "A4a_subclass_auxiliary",
    "A5_lite_confidence_consistency_0p05",
)
FIGURE2_METHODS = (
    "A0_full_no_masking",
    "A1_random_dropout",
    "A4a_subclass_auxiliary",
    "A5_lite_confidence_consistency_0p05",
)
FIGURE2_PATTERNS = ("full", "random-1", "random-3", "random-6")


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


def fmt(value: float | int | str | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def tex_fmt(value: float | int | str | None, digits: int = 4) -> str:
    return fmt(value, digits=digits).replace("_", "\\_")


def ci_text(observed: float, ci_low: float, ci_high: float) -> str:
    return f"{observed:.4f} [{ci_low:.4f}, {ci_high:.4f}]"


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


def aggregate_metric_dict(metric_dicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in MACRO_METRICS:
        values = np.asarray([float(d[metric]) for d in metric_dicts], dtype=np.float64)
        out[metric] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    for label in LABEL_ORDER:
        for metric_name in ("auroc", "auprc", "f1"):
            key = f"per_class_{metric_name}"
            values = np.asarray([float(d[key][label]) for d in metric_dicts], dtype=np.float64)
            out.setdefault(key, {})[label] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    out["n_valid_auroc_labels"] = int(min(int(d.get("n_valid_auroc_labels", 0)) for d in metric_dicts))
    out["n_valid_auprc_labels"] = int(min(int(d.get("n_valid_auprc_labels", 0)) for d in metric_dicts))
    out["invalid_macro_auroc"] = bool(any(bool(d.get("invalid_macro_auroc", False)) for d in metric_dicts))
    out["invalid_macro_auprc"] = bool(any(bool(d.get("invalid_macro_auprc", False)) for d in metric_dicts))
    out["warnings"] = [warning for d in metric_dicts for warning in d.get("warnings", [])]
    return out


def observed_metrics_for_data(data_by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    from hlm_ecg.statistics.bootstrap import compute_metric_bundle

    observed: dict[str, dict[str, dict[str, Any]]] = {}
    for method_id, pattern_data in data_by_method.items():
        observed[method_id] = {}
        for pattern, data in pattern_data.items():
            observed[method_id][pattern] = compute_metric_bundle(data)
        for aggregate, members in AGGREGATES.items():
            observed[method_id][aggregate] = aggregate_metric_dict([observed[method_id][pattern] for pattern in members])
    return observed


def bootstrap_metrics(
    data_by_method: Mapping[str, Mapping[str, Any]],
    *,
    samples: Sequence[np.ndarray],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    from hlm_ecg.statistics.bootstrap import compute_metric_bundle

    groups = {
        method_id: {pattern: patient_groups(data.patient_ids) for pattern, data in pattern_data.items()}
        for method_id, pattern_data in data_by_method.items()
    }
    values: dict[str, dict[str, dict[str, list[float]]]] = {
        method_id: {
            pattern: {metric: [] for metric in ALL_METRICS}
            for pattern in (*PATTERNS, *AGGREGATES.keys())
        }
        for method_id in data_by_method
    }
    for sample_idx, sampled_patients in enumerate(samples, start=1):
        if sample_idx == 1 or sample_idx % 100 == 0:
            print(f"bootstrap replicate {sample_idx}/{len(samples)}", flush=True)
        replicate_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        for method_id, pattern_data in data_by_method.items():
            replicate_metrics[method_id] = {}
            for pattern, data in pattern_data.items():
                indices = sampled_indices_from_patients(groups[method_id][pattern], sampled_patients)
                replicate_metrics[method_id][pattern] = compute_metric_bundle(data.subset(indices))
            for aggregate, members in AGGREGATES.items():
                replicate_metrics[method_id][aggregate] = aggregate_metric_dict(
                    [replicate_metrics[method_id][pattern] for pattern in members]
                )
            for pattern in values[method_id]:
                for metric in ALL_METRICS:
                    values[method_id][pattern][metric].append(metric_value(replicate_metrics[method_id][pattern], metric))
    return {
        method_id: {
            pattern: {metric: np.asarray(metric_values, dtype=np.float64) for metric, metric_values in metric_map.items()}
            for pattern, metric_map in pattern_map.items()
        }
        for method_id, pattern_map in values.items()
    }


def build_method_ci_rows(
    observed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    boot: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    n_bootstrap: int,
    seed: int,
    sampling_unit: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in METHODS:
        for pattern in REPORT_PATTERNS:
            for metric in ALL_METRICS:
                dist = summarize_distribution(boot[method_id][pattern][metric])
                rows.append(
                    {
                        "method_id": method_id,
                        "pattern": pattern,
                        "metric": metric,
                        "observed": metric_value(observed[method_id][pattern], metric),
                        "ci_low": dist["ci_low"],
                        "ci_high": dist["ci_high"],
                        "bootstrap_mean": dist["mean"],
                        "n_bootstrap_valid": dist["n_bootstrap_valid"],
                        "invalid_replicates": dist["invalid_replicates"],
                        "sampling_unit": sampling_unit,
                        "n_bootstrap": n_bootstrap,
                        "seed": seed,
                    }
                )
    return rows


def interpretation(ci_low: float, ci_high: float, observed_delta: float) -> str:
    if ci_low > 0:
        return "paired bootstrap supports a positive delta"
    if ci_high < 0:
        return "paired bootstrap supports a negative delta"
    if observed_delta > 0:
        return "point estimate favors method_a, but CI overlaps zero"
    if observed_delta < 0:
        return "point estimate favors method_b, but CI overlaps zero"
    return "point estimate is zero and CI overlaps zero"


def build_paired_delta_rows(
    observed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    boot: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison_id, method_a, method_b in COMPARISONS:
        for pattern in REPORT_PATTERNS:
            for metric in ALL_METRICS:
                observed_delta = metric_value(observed[method_a][pattern], metric) - metric_value(observed[method_b][pattern], metric)
                delta_dist = boot[method_a][pattern][metric] - boot[method_b][pattern][metric]
                summary = paired_delta_summary(delta_dist, observed_delta)
                rows.append(
                    {
                        "comparison_id": comparison_id,
                        "method_a": method_a,
                        "method_b": method_b,
                        "pattern_or_aggregate": pattern,
                        "metric": metric,
                        **summary,
                        "interpretation": interpretation(float(summary["ci_low"]), float(summary["ci_high"]), observed_delta),
                    }
                )
    return rows


def find_row(rows: Sequence[Mapping[str, Any]], **criteria: str) -> Mapping[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(f"Row not found: {criteria}")


def write_method_ci_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "bootstrap_method_ci.csv", rows)
    write_json(out_dir / "bootstrap_method_ci.json", {"rows": rows})
    md_rows = [r for r in rows if r["metric"] in {"macro_auprc", "macro_auroc", "macro_f1"}]
    columns = [
        ("method", "method_id"),
        ("pattern", "pattern"),
        ("metric", "metric"),
        ("observed", "observed"),
        ("ci_low", "ci_low"),
        ("ci_high", "ci_high"),
        ("valid", "n_bootstrap_valid"),
    ]
    (out_dir / "bootstrap_method_ci.md").write_text("# Bootstrap Method CI\n\n" + markdown_table(md_rows, columns), encoding="utf-8")


def write_paired_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "paired_delta_ci.csv", rows)
    write_json(out_dir / "paired_delta_ci.json", {"rows": rows})
    md_rows = [r for r in rows if r["metric"] == "macro_auprc"]
    columns = [
        ("comparison", "comparison_id"),
        ("pattern", "pattern_or_aggregate"),
        ("metric", "metric"),
        ("delta", "observed_delta"),
        ("ci_low", "ci_low"),
        ("ci_high", "ci_high"),
        ("P(delta>0)", "probability_delta_gt_0"),
        ("p_boot", "p_two_sided"),
        ("interpretation", "interpretation"),
    ]
    (out_dir / "paired_delta_ci.md").write_text("# Paired Delta CI\n\n" + markdown_table(md_rows, columns), encoding="utf-8")


def write_table1(out_dir: Path, method_rows: list[dict[str, Any]]) -> None:
    rows = []
    for method_id in TABLE1_METHODS:
        row = {"method_id": method_id}
        for pattern in (
            "full",
            "random-3",
            "random-6",
            "limb-only / precordial-missing",
            "precordial-only / limb-missing",
            "V1-V3 missing",
            "V4-V6 missing",
            "hard_structured_avg",
            "hard_overall_avg",
        ):
            ci_row = find_row(method_rows, method_id=method_id, pattern=pattern, metric="macro_auprc")
            key = pattern.replace(" / ", "_").replace("-", "_").replace(" ", "_")
            row[key] = ci_text(float(ci_row["observed"]), float(ci_row["ci_low"]), float(ci_row["ci_high"]))
        rows.append(row)
    columns = [(key, key) for key in rows[0]]
    write_csv(out_dir / "table1_main_robustness_with_ci.csv", rows)
    (out_dir / "table1_main_robustness_with_ci.md").write_text(markdown_table(rows, columns), encoding="utf-8")
    (out_dir / "table1_main_robustness_with_ci.tex").write_text(latex_table(rows, columns), encoding="utf-8")


def write_appendix_delta(out_dir: Path, delta_rows: list[dict[str, Any]]) -> None:
    rows = [
        row for row in delta_rows
        if row["metric"] in {"macro_auprc", "macro_auroc", "macro_f1"}
    ]
    write_csv(out_dir / "appendix_paired_delta_table.csv", rows)
    columns = [
        ("comparison", "comparison_id"),
        ("pattern", "pattern_or_aggregate"),
        ("metric", "metric"),
        ("delta", "observed_delta"),
        ("ci_low", "ci_low"),
        ("ci_high", "ci_high"),
        ("P(delta>0)", "probability_delta_gt_0"),
    ]
    (out_dir / "appendix_paired_delta_table.md").write_text(markdown_table(rows, columns), encoding="utf-8")
    (out_dir / "appendix_paired_delta_table.tex").write_text(latex_table(rows, columns), encoding="utf-8")


def write_figure_data(out_dir: Path, method_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> None:
    fig2 = []
    for method_id in FIGURE2_METHODS:
        for order, pattern in enumerate(FIGURE2_PATTERNS):
            for metric in ("macro_auprc", "macro_auroc", "macro_f1"):
                row = find_row(method_rows, method_id=method_id, pattern=pattern, metric=metric)
                fig2.append({"method_id": method_id, "pattern": pattern, "pattern_order": order, "metric": metric, **{k: row[k] for k in ("observed", "ci_low", "ci_high")}})
    write_csv(out_dir / "figure2_degradation_curve_ci_data.csv", fig2)
    write_json(out_dir / "figure2_degradation_curve_ci_data.json", {"rows": fig2})

    heat = []
    for pattern in HARD_OVERALL_PATTERNS:
        for label in LABEL_ORDER:
            row = find_row(
                delta_rows,
                comparison_id="A4a_vs_A1",
                pattern_or_aggregate=pattern,
                metric=f"per_class_{label}_auprc",
            )
            heat.append(
                {
                    "comparison_id": "A4a_vs_A1",
                    "method_a": "A4a_subclass_auxiliary",
                    "method_b": "A1_random_dropout",
                    "pattern": pattern,
                    "label": label,
                    "metric": "per_class_auprc",
                    "observed_delta": row["observed_delta"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "probability_delta_gt_0": row["probability_delta_gt_0"],
                }
            )
    write_csv(out_dir / "figure3_heatmap_delta_ci_data.csv", heat)
    write_json(out_dir / "figure3_heatmap_delta_ci_data.json", {"rows": heat})


def write_summary_for_paper(out_dir: Path, delta_rows: list[dict[str, Any]]) -> None:
    key_specs = [
        ("A4a_vs_A1", "hard_structured_avg", "macro_auprc"),
        ("A4a_vs_A1", "hard_overall_avg", "macro_auprc"),
        ("A4a_vs_A2", "hard_overall_avg", "macro_auprc"),
        ("A5_lite_vs_A4a", "full", "macro_auprc"),
        ("A5_lite_vs_A4a", "hard_overall_avg", "macro_auprc"),
        ("A4a_vs_A0", "hard_overall_avg", "macro_auprc"),
    ]
    rows = [find_row(delta_rows, comparison_id=c, pattern_or_aggregate=p, metric=m) for c, p, m in key_specs]
    lines = ["# Bootstrap Summary for Paper", ""]
    for row in rows:
        lines.append(
            f"- {row['comparison_id']} / {row['pattern_or_aggregate']} / {row['metric']}: "
            f"delta `{float(row['observed_delta']):.6f}`, 95% CI "
            f"`[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]`, "
            f"P(delta>0)=`{float(row['probability_delta_gt_0']):.4f}`."
        )
    lines.append("")
    a4a_a1 = rows[1]
    if float(a4a_a1["ci_low"]) > 0:
        lines.append("Paired patient-level bootstrap supported a positive improvement of A4a over random lead dropout under hard missing patterns.")
    else:
        lines.append("The point estimate favored A4a, but the paired bootstrap interval overlapped zero; report the gain as a trend.")
    a5_full = rows[3]
    a5_hard = rows[4]
    if float(a5_full["ci_low"]) > 0 and float(a5_hard["ci_high"]) < 0:
        lines.append("Consistency improved full-lead performance but traded off hard missing robustness.")
    (out_dir / "bootstrap_summary_for_paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    columns = [
        ("comparison", "comparison_id"),
        ("pattern", "pattern_or_aggregate"),
        ("metric", "metric"),
        ("delta", "observed_delta"),
        ("ci_low", "ci_low"),
        ("ci_high", "ci_high"),
        ("P(delta>0)", "probability_delta_gt_0"),
    ]
    (out_dir / "bootstrap_summary_for_paper.tex").write_text(latex_table(rows, columns), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sampling_unit != "patient":
        raise ValueError("Only patient-level primary bootstrap is supported in this script")
    if Path("data/ptb-xl/records500").exists():
        raise RuntimeError("records500 exists; refusing bootstrap")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_prediction_data(
        args.predictions_dir,
        methods=METHODS,
        patterns=PATTERNS,
        split=args.split,
        fill_mode=args.fill_mode,
    )
    base = data["A0_full_no_masking"]["full"]
    samples = generate_patient_bootstrap_samples(base.patient_ids, n_bootstrap=args.n_bootstrap, seed=args.seed)
    observed = observed_metrics_for_data(data)
    boot = bootstrap_metrics(data, samples=samples)
    method_rows = build_method_ci_rows(observed, boot, n_bootstrap=args.n_bootstrap, seed=args.seed, sampling_unit=args.sampling_unit)
    delta_rows = build_paired_delta_rows(observed, boot)
    write_method_ci_outputs(out_dir, method_rows)
    write_paired_outputs(out_dir, delta_rows)
    write_table1(out_dir, method_rows)
    write_appendix_delta(out_dir, delta_rows)
    write_figure_data(out_dir, method_rows, delta_rows)
    write_summary_for_paper(out_dir, delta_rows)
    config = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "sampling_unit": args.sampling_unit,
        "split": args.split,
        "fill_mode": args.fill_mode,
        "methods": list(METHODS),
        "patterns": list(PATTERNS),
        "label_order": list(LABEL_ORDER),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records500_used": False,
        "smoke_test": bool(args.smoke_test),
    }
    write_json(out_dir / "bootstrap_run_config.json", config)
    result = {
        "out_dir": str(out_dir),
        "method_ci_rows": len(method_rows),
        "paired_delta_rows": len(delta_rows),
        **config,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patient-level paired bootstrap CI for HLM-ECG.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fill-mode", default="mean_fill")
    parser.add_argument("--sampling-unit", default="patient")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
