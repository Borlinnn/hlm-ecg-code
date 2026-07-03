#!/usr/bin/env python3
"""Build the three-strategy reviewer-defense benchmark tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_reviewer_defense_results import HARD_STRUCTURED
from hlm_ecg.evaluation.week6_defense import (
    limb_reconstruction_applicability,
    selected_patterns,
)

CHALLENGE_PATTERNS = (
    "challenge_6_limb",
    "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2",
    "challenge_2_I_II",
)
TABLE_PATTERNS = (*HARD_STRUCTURED, *CHALLENGE_PATTERNS)
RECONSTRUCTION_AUDIT_PATTERNS = ("full", *TABLE_PATTERNS)
COMMON_SEEDS = (7, 42, 123)
ALL_SEEDS = (7, 42, 123, 2024, 2025)
BACKBONE = "xresnet1d101_like"
METRICS = ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll")
RECONSTRUCTION_SCOPE_NOTE = (
    "I/II limb reconstruction analytically fills only missing III/aVR/aVL/aVF "
    "when I and II are measured; it does not synthesize precordial leads."
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt_mean_sd(mean: float | None, sd: float | None) -> str:
    if mean is None or sd is None or not np.isfinite(mean):
        return ""
    if not np.isfinite(sd):
        sd = 0.0
    return f"{mean:.4f} +/- {sd:.4f}"


def fmt_delta(delta: float | None) -> str:
    if delta is None or not np.isfinite(delta):
        return ""
    return f"{delta:+.4f}"


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text: Any) -> str:
    value = str(text)
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def latex_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], *, caption: str, label: str) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        " & ".join(latex_escape(col) for col in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(col, "")) for col in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def strategy_table_notes() -> str:
    return (
        "\nNotes: `I/II limb reconstruction` uses analytic I/II-derived limb-lead relationships only "
        "(`III`, `aVR`, `aVL`, `aVF`) and does not synthesize precordial leads. "
        "`limb-only / precordial-missing` and `challenge_6_limb` have the same visible lead set; "
        "both are kept to connect the internal and Challenge-style naming conventions.\n"
    )


def reconstruction_applicability_rows() -> list[dict[str, Any]]:
    patterns = selected_patterns(RECONSTRUCTION_AUDIT_PATTERNS)
    rows: list[dict[str, Any]] = []
    first_visible_set: dict[str, str] = {}
    for pattern_name, pattern in patterns.items():
        row = limb_reconstruction_applicability(pattern_name, pattern)
        visible_key = str(row["available_leads"])
        row["visible_set_duplicate_of"] = first_visible_set.get(visible_key, "")
        first_visible_set.setdefault(visible_key, pattern_name)
        rows.append(row)
    return rows


def read_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_imputation_rows(imputation_dir: Path) -> pd.DataFrame:
    combined = imputation_dir / "imputation_metric_rows.csv"
    if combined.exists():
        return pd.read_csv(combined)
    paths = sorted((imputation_dir / "metric_rows").glob("*.csv"))
    return read_csvs(paths)


def load_primary_rows(primary_dir: Path) -> pd.DataFrame:
    path = primary_dir / "aggregate_metric_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[
        (frame["group"] == "primary")
        & (frame["backbone"] == BACKBONE)
        & (frame["fill_mode"] == "mean_fill")
        & (frame["pattern_or_aggregate"].isin(HARD_STRUCTURED))
        & (frame["method"].isin(["M1_random_dropout", "M2_structured_masking", "M6_structured_plus_availability_plus_subclass"]))
    ].copy()
    frame = frame.rename(columns={"method": "method_id", "pattern_or_aggregate": "pattern"})
    frame["source"] = "primary_final_analysis"
    return frame


def load_specialist_rows(specialist_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(specialist_dir.glob("xresnet1d101_like/*/seed*/test_full_metrics.json")):
        run_dir = metrics_path.parent
        config = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8")) or {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pattern = str(config.get("week6_specialist", {}).get("pattern", "unknown"))
        seed = int(config.get("seed", run_dir.name.replace("seed", "")))
        row: dict[str, Any] = {
            "strategy": "fixed_pattern_specialist",
            "method_id": "SPECIALIST_fixed_pattern",
            "backbone": BACKBONE,
            "seed": seed,
            "pattern": pattern,
            "fill_mode": "specialist",
            "run_dir": str(run_dir),
            "records500_used": False,
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric)
        per_class = metrics.get("per_class_auprc", {})
        if isinstance(per_class, dict):
            for label, value in per_class.items():
                row[f"auprc_{label}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_imputation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "backbone" not in out.columns:
        out["backbone"] = BACKBONE
    out["strategy"] = np.where(
        out["fill_mode"].eq("physiology_limb_reconstruction_fill"),
        "physiology_reconstruction_then_classify",
        "single_model_imputation_eval",
    )
    out["records500_used"] = False
    return out


def robust_rows_from_sources(primary_rows: pd.DataFrame, imputation_rows: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not imputation_rows.empty:
        robust = imputation_rows[
            (imputation_rows["backbone"] == BACKBONE)
            & (imputation_rows["fill_mode"] == "mean_fill")
            & (imputation_rows["method_id"].isin(["M1_random_dropout", "M2_structured_masking", "M6_structured_plus_availability_plus_subclass"]))
        ].copy()
        robust["strategy"] = "single_robust_model"
        robust["source"] = "imputation_eval_mean_fill"
        frames.append(robust)
    if not primary_rows.empty:
        fallback = primary_rows.copy()
        fallback["strategy"] = "single_robust_model"
        frames.append(fallback)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["source_rank"] = combined["source"].map({"imputation_eval_mean_fill": 0, "primary_final_analysis": 1}).fillna(9)
    combined = combined.sort_values("source_rank")
    return combined.drop_duplicates(["method_id", "backbone", "seed", "pattern"], keep="first").drop(columns=["source_rank"])


def mean_sd(frame: pd.DataFrame, *, method_id: str, pattern: str, seeds: Sequence[int], fill_mode: str | None = None) -> tuple[float | None, float | None, int]:
    if frame.empty:
        return None, None, 0
    sub = frame[(frame["method_id"] == method_id) & (frame["pattern"] == pattern) & (frame["seed"].isin(seeds))]
    if fill_mode is not None:
        sub = sub[sub["fill_mode"] == fill_mode]
    values = sub["macro_auprc"].dropna().astype(float)
    if values.empty:
        return None, None, 0
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0, int(values.shape[0])


def values_by_seed(frame: pd.DataFrame, *, method_id: str, pattern: str, seeds: Sequence[int], fill_mode: str | None = None) -> dict[int, float]:
    if frame.empty:
        return {}
    sub = frame[(frame["method_id"] == method_id) & (frame["pattern"] == pattern) & (frame["seed"].isin(seeds))]
    if fill_mode is not None:
        sub = sub[sub["fill_mode"] == fill_mode]
    return {int(row["seed"]): float(row["macro_auprc"]) for row in sub.to_dict("records")}


def t_critical_975(n: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    return table.get(n, 1.96)


def delta_summary(values_a: Mapping[int, float], values_b: Mapping[int, float]) -> dict[str, Any]:
    common = sorted(set(values_a).intersection(values_b))
    deltas = np.asarray([values_a[seed] - values_b[seed] for seed in common], dtype=np.float64)
    if deltas.size == 0:
        return {
            "n_seeds": 0,
            "delta_mean": "",
            "delta_sd": "",
            "delta_ci95_low_seed_t": "",
            "delta_ci95_high_seed_t": "",
            "sign_count_positive": "",
            "all_seed_deltas": "",
        }
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0
    half = t_critical_975(int(deltas.size)) * sd / math.sqrt(int(deltas.size)) if deltas.size > 1 else 0.0
    return {
        "n_seeds": int(deltas.size),
        "delta_mean": mean,
        "delta_sd": sd,
        "delta_ci95_low_seed_t": mean - half,
        "delta_ci95_high_seed_t": mean + half,
        "sign_count_positive": int(np.sum(deltas > 0)),
        "all_seed_deltas": ";".join(f"{x:.6f}" for x in deltas),
    }


def build_seed_delta_rows(robust: pd.DataFrame, imputation: pd.DataFrame, specialist: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("M6_minus_M1", robust, "M6_structured_plus_availability_plus_subclass", None, robust, "M1_random_dropout", None),
        ("M6_recon_minus_M6", imputation, "M6_structured_plus_availability_plus_subclass", "physiology_limb_reconstruction_fill", robust, "M6_structured_plus_availability_plus_subclass", None),
        ("M0_recon_minus_M6", imputation, "M0_full_no_masking", "physiology_limb_reconstruction_fill", robust, "M6_structured_plus_availability_plus_subclass", None),
        ("specialist_minus_M6", specialist, "SPECIALIST_fixed_pattern", "specialist", robust, "M6_structured_plus_availability_plus_subclass", None),
    ]
    for pattern in TABLE_PATTERNS:
        for comparison_id, frame_a, method_a, fill_a, frame_b, method_b, fill_b in comparisons:
            values_a = values_by_seed(frame_a, method_id=method_a, pattern=pattern, seeds=COMMON_SEEDS, fill_mode=fill_a)
            values_b = values_by_seed(frame_b, method_id=method_b, pattern=pattern, seeds=COMMON_SEEDS, fill_mode=fill_b)
            row = {
                "comparison_id": comparison_id,
                "pattern": pattern,
                "method_a": method_a,
                "method_b": method_b,
                "metric": "macro_auprc",
                **delta_summary(values_a, values_b),
            }
            rows.append(row)
    return rows


def build_main_table(robust: pd.DataFrame, imputation: pd.DataFrame, specialist: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []
    delta_rows = build_seed_delta_rows(robust, imputation, specialist)
    delta_lookup = {(row["comparison_id"], row["pattern"]): row for row in delta_rows}
    columns = [
        ("M1 random dropout", robust, "M1_random_dropout", None),
        ("M2 structured masking", robust, "M2_structured_masking", None),
        ("M6 HLM-lite", robust, "M6_structured_plus_availability_plus_subclass", None),
        ("M0 + I/II limb reconstruction", imputation, "M0_full_no_masking", "physiology_limb_reconstruction_fill"),
        ("M6 + I/II limb reconstruction", imputation, "M6_structured_plus_availability_plus_subclass", "physiology_limb_reconstruction_fill"),
        ("fixed-pattern specialist", specialist, "SPECIALIST_fixed_pattern", "specialist"),
    ]
    for pattern in TABLE_PATTERNS:
        raw: dict[str, Any] = {"pattern": pattern, "seeds": "7;42;123", "metric": "macro_auprc"}
        display: dict[str, Any] = {"pattern": pattern}
        for label, frame, method_id, fill_mode in columns:
            mean, sd, n = mean_sd(frame, method_id=method_id, pattern=pattern, seeds=COMMON_SEEDS, fill_mode=fill_mode)
            raw[f"{label}_mean"] = mean if mean is not None else ""
            raw[f"{label}_sd"] = sd if sd is not None else ""
            raw[f"{label}_n"] = n
            display[label] = fmt_mean_sd(mean, sd)
        for comp_id, label in (
            ("M6_minus_M1", "M6 - M1"),
            ("M6_recon_minus_M6", "M6+I/II recon - M6"),
            ("specialist_minus_M6", "specialist - M6"),
        ):
            delta = delta_lookup.get((comp_id, pattern), {})
            raw[label] = delta.get("delta_mean", "")
            raw[f"{label}_sign_count"] = delta.get("sign_count_positive", "")
            display[label] = fmt_delta(delta.get("delta_mean") if isinstance(delta.get("delta_mean"), float) else None)
        raw_rows.append(raw)
        display_rows.append(display)
    return raw_rows, display_rows


def build_summary_rows(frame: pd.DataFrame, *, seeds: Sequence[int], label: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    for keys, group in frame[frame["seed"].isin(seeds)].groupby(["strategy", "method_id", "pattern", "fill_mode"], dropna=False):
        strategy, method_id, pattern, fill_mode = keys
        row: dict[str, Any] = {
            "summary": label,
            "strategy": strategy,
            "method_id": method_id,
            "pattern": pattern,
            "fill_mode": fill_mode,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else ""
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else ""
        out.append(row)
    return out


def build_claim_evidence(output_dir: Path, missing_cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "claim": "Three strategy benchmark uses common seeds 7,42,123.",
            "evidence_file": str(output_dir / "table_strategy_benchmark_common3.csv"),
            "status": "ready" if not missing_cells else "has_missing_cells",
        },
        {
            "claim": "Imputation/reconstruction is evaluation-only and uses validation thresholds.",
            "evidence_file": str(output_dir / "imputation_metric_rows.csv"),
            "status": "ready",
        },
        {
            "claim": "Fixed-pattern specialists are an upper-bound style baseline requiring pattern-specific training.",
            "evidence_file": str(output_dir / "specialist_metric_rows.csv"),
            "status": "ready",
        },
        {
            "claim": "I/II-derived limb reconstruction is complementary in low-lead I/II settings, not a precordial synthesis method.",
            "evidence_file": str(output_dir / "strategy_claim_language.md"),
            "status": "drafted",
        },
        {
            "claim": "No-op reconstruction rows are expected when missing leads are precordial or I/II is unavailable.",
            "evidence_file": str(output_dir / "reconstruction_applicability.csv"),
            "status": "ready",
        },
    ]


def detect_missing_cells(display_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for row in display_rows:
        for key, value in row.items():
            if key == "pattern":
                continue
            if value == "":
                missing.append({"pattern": row["pattern"], "column": key})
    return missing


def write_claim_language(path: Path) -> None:
    text = """# Strategy Benchmark Claim Language

- Reconstruction/imputation is complementary rather than a substitute for pattern-aware robust training.
- I/II-derived limb reconstruction is complementary in low-lead I/II settings.
- It is intentionally not a precordial synthesis method; therefore 6-limb chest-missing settings are no-op under this audit.
- Reconstruction-only should not be framed as useless; report where it helps and where single robust training remains stronger.
- Do not claim SOTA superiority. Claim a controlled strategy-level comparison under one dataset, split, backbone family, and threshold protocol.
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    imputation = normalize_imputation_rows(load_imputation_rows(args.imputation_dir))
    primary = load_primary_rows(args.primary_dir)
    robust = robust_rows_from_sources(primary, imputation)
    specialist = load_specialist_rows(args.specialist_dir)
    combined_sources = pd.concat(
        [df for df in (robust, imputation, specialist) if not df.empty],
        ignore_index=True,
        sort=False,
    ) if any(not df.empty for df in (robust, imputation, specialist)) else pd.DataFrame()

    if not imputation.empty:
        write_csv(output_dir / "imputation_metric_rows.csv", imputation.to_dict("records"))
        write_csv(output_dir / "imputation_mean_sd.csv", build_summary_rows(imputation, seeds=ALL_SEEDS, label="all5"))
    write_csv(output_dir / "specialist_metric_rows.csv", specialist.to_dict("records") if not specialist.empty else [])
    write_csv(output_dir / "single_robust_metric_rows_for_strategy.csv", robust.to_dict("records") if not robust.empty else [])
    write_csv(output_dir / "appendix_strategy_all5_summary.csv", build_summary_rows(combined_sources, seeds=ALL_SEEDS, label="all5"))
    write_csv(output_dir / "appendix_strategy_common3_summary.csv", build_summary_rows(combined_sources, seeds=COMMON_SEEDS, label="common3"))

    raw_rows, display_rows = build_main_table(robust, imputation, specialist)
    missing_cells = detect_missing_cells(display_rows)
    if missing_cells and args.strict:
        write_csv(output_dir / "table_strategy_benchmark_missing_cells.csv", missing_cells)
        raise RuntimeError(f"Strategy table has {len(missing_cells)} missing cells")
    table_columns = [
        "pattern",
        "M1 random dropout",
        "M2 structured masking",
        "M6 HLM-lite",
        "M0 + I/II limb reconstruction",
        "M6 + I/II limb reconstruction",
        "fixed-pattern specialist",
        "M6 - M1",
        "M6+I/II recon - M6",
        "specialist - M6",
    ]
    write_csv(output_dir / "table_strategy_benchmark_common3.csv", raw_rows)
    (output_dir / "table_strategy_benchmark_common3.md").write_text(
        markdown_table(display_rows, table_columns) + strategy_table_notes(),
        encoding="utf-8",
    )
    (output_dir / "table_strategy_benchmark_common3.tex").write_text(
        latex_table(
            display_rows,
            table_columns,
            caption=f"Strategy-level benchmark under simulated missing-lead ECG classification. {RECONSTRUCTION_SCOPE_NOTE}",
            label="tab:strategy_benchmark",
        ),
        encoding="utf-8",
    )

    applicability_rows = reconstruction_applicability_rows()
    write_csv(output_dir / "reconstruction_applicability.csv", applicability_rows)
    applicability_columns = [
        "pattern",
        "available_leads",
        "missing_leads",
        "reconstructable_missing_limb_leads",
        "n_reconstructed_leads",
        "no_op_reason",
        "visible_set_duplicate_of",
    ]
    (output_dir / "reconstruction_applicability.md").write_text(
        markdown_table(applicability_rows, applicability_columns) + f"\nNotes: {RECONSTRUCTION_SCOPE_NOTE}\n",
        encoding="utf-8",
    )

    seed_delta_rows = build_seed_delta_rows(robust, imputation, specialist)
    write_csv(output_dir / "strategy_seed_paired_deltas.csv", seed_delta_rows)
    challenge_rows = [row for row in raw_rows if str(row["pattern"]).startswith("challenge_")]
    write_csv(output_dir / "appendix_challenge_strategy_common3.csv", challenge_rows)
    if missing_cells:
        write_csv(output_dir / "table_strategy_benchmark_missing_cells.csv", missing_cells)
    write_claim_language(output_dir / "strategy_claim_language.md")
    evidence = build_claim_evidence(output_dir, missing_cells)
    write_csv(output_dir / "claim_evidence_matrix.csv", evidence)
    summary = {
        "output_dir": str(output_dir),
        "n_imputation_rows": int(len(imputation)),
        "n_specialist_rows": int(len(specialist)),
        "n_robust_rows": int(len(robust)),
        "n_table_rows": len(raw_rows),
        "n_missing_cells": len(missing_cells),
        "n_reconstruction_applicability_rows": len(applicability_rows),
        "common_seeds": list(COMMON_SEEDS),
        "all_seeds": list(ALL_SEEDS),
        "records500_used": False,
        "filename_hr_used": False,
    }
    write_json(output_dir / "three_strategy_benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, default=Path("results/reviewer_defense_20260701/final_analysis"))
    parser.add_argument("--imputation-dir", type=Path, default=Path("results/reviewer_defense_20260701/strategy_benchmark"))
    parser.add_argument("--specialist-dir", type=Path, default=Path("outputs/reviewer_defense_20260701/specialist_upper_bound"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_defense_20260701/strategy_benchmark"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
