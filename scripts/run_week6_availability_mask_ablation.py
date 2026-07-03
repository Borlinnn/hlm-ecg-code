#!/usr/bin/env python3
"""Run Week 6 availability-mask mechanism ablation for A4a."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from hlm_ecg.evaluation.supplemental_analysis import (
    assert_no_records500_in_runs,
    base_metadata,
    markdown_report,
    summarize_multiseed,
    write_csv,
    write_json,
    write_markdown_table,
)
from hlm_ecg.evaluation.supplemental_patterns import pattern_metadata
from hlm_ecg.evaluation.week6_defense import (
    AVAILABILITY_VARIANTS,
    BOUNDARY_PATTERNS,
    HARD_STRUCTURED_PATTERNS,
    K_BOUNDARY_PATTERNS,
    ROOT,
    WEEK6_DIR,
    WEEK6_PATTERN_SEED,
    evaluate_week6_pattern,
    method_runs_for_week6,
    selected_patterns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 availability-mask mechanism ablation.")
    parser.add_argument("--patterns", nargs="+", default=list(BOUNDARY_PATTERNS))
    parser.add_argument("--variants", nargs="+", default=list(AVAILABILITY_VARIANTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=WEEK6_DIR / "availability_mask_ablation")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _filter_runs(runs, seeds: list[int] | None, *, smoke_test: bool):
    runs = [run for run in runs if run.method_id == "A4a_subclass_auxiliary"]
    if smoke_test:
        seeds = [42]
    if seeds:
        runs = [run for run in runs if run.seed in set(seeds)]
    return runs


def _variant_delta_rows(rows):
    by_key = {
        (row["seed"], row["pattern"], row["availability_variant"]): row
        for row in rows
    }
    out = []
    for row in rows:
        if row["availability_variant"] == "correct":
            continue
        correct = by_key.get((row["seed"], row["pattern"], "correct"))
        if correct is None:
            continue
        out.append(
            {
                "seed": row["seed"],
                "pattern": row["pattern"],
                "variant": row["availability_variant"],
                "delta_macro_auprc_vs_correct": float(row["macro_auprc"]) - float(correct["macro_auprc"]),
                "delta_macro_auroc_vs_correct": float(row["macro_auroc"]) - float(correct["macro_auroc"]),
                "delta_macro_f1_vs_correct": float(row["macro_f1"]) - float(correct["macro_f1"]),
                "delta_bce_nll_vs_correct": float(row["bce_nll"]) - float(correct["bce_nll"]),
            }
        )
    return out


def _guard_summary(delta_rows):
    severe = set(HARD_STRUCTURED_PATTERNS).union(K_BOUNDARY_PATTERNS)
    rows = [row for row in delta_rows if row["pattern"] in severe]
    out = []
    for variant in ("all_ones", "shuffled"):
        values = [float(row["delta_macro_auprc_vs_correct"]) for row in rows if row["variant"] == variant]
        if not values:
            continue
        mean_delta = float(np.mean(values))
        out.append(
            {
                "variant": variant,
                "n_rows": len(values),
                "mean_delta_macro_auprc_vs_correct": mean_delta,
                "mask_signal_used_interpretation": (
                    "corrupting availability hurts or changes predictions"
                    if mean_delta < -0.001
                    else "weak or pattern-dependent mask-use evidence"
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_names = list(args.patterns[:2] if args.smoke_test else args.patterns)
    variants = list(args.variants[:2] if args.smoke_test else args.variants)
    patterns = selected_patterns(pattern_names)
    runs = _filter_runs(method_runs_for_week6(["A4a_subclass_auxiliary"]), args.seeds, smoke_test=args.smoke_test)
    assert_no_records500_in_runs(runs)

    metadata = base_metadata(runs, fill_mode="mean_fill", pattern_seed=WEEK6_PATTERN_SEED)
    metadata.update(
        {
            "analysis": "week6_availability_mask_ablation",
            "patterns": pattern_metadata(patterns),
            "variants": variants,
            "smoke_test": bool(args.smoke_test),
            "records500_used": False,
            "input_signal_semantics": "ECG input uses the correct measured-lead mask; only model availability metadata is changed.",
        }
    )
    write_json(output_dir / "availability_mask_ablation_patterns.json", metadata)

    rows = []
    for run in runs:
        for variant in variants:
            for pattern_name, pattern in patterns.items():
                row, _ = evaluate_week6_pattern(
                    run=run,
                    pattern_name=pattern_name,
                    pattern=pattern,
                    imputation_strategy="mean_fill",
                    split="test",
                    smoke_test=args.smoke_test,
                    availability_variant=variant,
                )
                rows.append(row)

    metric_columns = [
        "method_id",
        "seed",
        "method_run_id",
        "pattern",
        "availability_variant",
        "fill_mode",
        "n",
        "macro_auroc",
        "macro_auprc",
        "macro_f1",
        "bce_nll",
        "auprc_NORM",
        "auprc_MI",
        "auprc_STTC",
        "auprc_CD",
        "auprc_HYP",
        "thresholds_source_split",
        "records500_used",
    ]
    write_csv(output_dir / "availability_mask_ablation_metrics.csv", rows, metric_columns)
    write_json(output_dir / "availability_mask_ablation_metrics.json", {**metadata, "rows": rows})
    write_markdown_table(output_dir / "availability_mask_ablation_metrics.md", rows, metric_columns[:16])

    deltas = _variant_delta_rows(rows)
    delta_cols = [
        "seed",
        "pattern",
        "variant",
        "delta_macro_auprc_vs_correct",
        "delta_macro_auroc_vs_correct",
        "delta_macro_f1_vs_correct",
        "delta_bce_nll_vs_correct",
    ]
    write_csv(output_dir / "availability_mask_ablation_delta_vs_correct.csv", deltas, delta_cols)
    write_markdown_table(output_dir / "availability_mask_ablation_delta_vs_correct.md", deltas, delta_cols)

    summary = summarize_multiseed(rows, group_cols=["pattern", "availability_variant"])
    write_csv(output_dir / "availability_mask_ablation_multiseed_summary.csv", summary)
    write_markdown_table(
        output_dir / "availability_mask_ablation_multiseed_summary.md",
        summary,
        list(summary[0]) if summary else [],
    )

    guard = _guard_summary(deltas)
    write_csv(output_dir / "lead_availability_reliability_guard.csv", guard)
    write_json(
        output_dir / "lead_availability_reliability_guard.json",
        {
            "records500_used": False,
            "rule": "negative mean delta after corrupting masks is evidence the model uses availability metadata",
            "rows": guard,
        },
    )
    write_markdown_table(
        output_dir / "lead_availability_reliability_guard.md",
        guard,
        list(guard[0]) if guard else ["variant", "n_rows", "mean_delta_macro_auprc_vs_correct", "mask_signal_used_interpretation"],
    )

    lines = [
        "- Evaluated A4a with unchanged masked ECG signals and corrupted availability metadata.",
        f"- Seeds: `{sorted({run.seed for run in runs})}`.",
        f"- Variants: `{variants}`.",
        f"- Patterns: `{pattern_names}`.",
        "- `all_ones` asks whether the model fails when it is told all leads are available.",
        "- `shuffled` is most informative for per-sample random k-visible patterns; fixed masks may be unchanged by shuffling.",
    ]
    markdown_report(output_dir / "availability_mask_ablation_report.md", "Week 6 Availability-mask Mechanism Ablation", lines)


if __name__ == "__main__":
    main()
