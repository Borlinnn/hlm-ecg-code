#!/usr/bin/env python3
"""Run Week 6 reconstruction / imputation reviewer-defense audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.evaluation.supplemental_analysis import (
    assert_no_records500_in_runs,
    base_metadata,
    markdown_report,
    paired_bootstrap_prediction_delta,
    summarize_multiseed,
    write_csv,
    write_json,
    write_markdown_table,
)
from hlm_ecg.evaluation.supplemental_patterns import pattern_metadata
from hlm_ecg.evaluation.week6_defense import (
    CHALLENGE_RECON_PATTERNS,
    DEFAULT_NO_TRAIN_METHODS,
    IMPUTATION_STRATEGIES,
    K_BOUNDARY_PATTERNS,
    ROOT,
    WEEK6_DIR,
    WEEK6_PATTERN_SEED,
    delta_vs_baseline_with_fields,
    evaluate_week6_pattern,
    method_runs_for_week6,
    selected_patterns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 reconstruction / imputation audit.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_NO_TRAIN_METHODS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--strategies", nargs="+", default=list(IMPUTATION_STRATEGIES))
    parser.add_argument("--patterns", nargs="+", default=list(CHALLENGE_RECON_PATTERNS))
    parser.add_argument("--include-k-boundary", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=WEEK6_DIR / "reconstruction_imputation")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=WEEK6_PATTERN_SEED)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _filter_runs(runs, seeds: list[int] | None, *, smoke_test: bool):
    if smoke_test:
        seeds = [42]
    if seeds:
        runs = [run for run in runs if run.seed in set(seeds)]
    if smoke_test:
        keep = {"A1_random_dropout", "A4a_subclass_auxiliary"}
        runs = [run for run in runs if run.method_id in keep]
    return runs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_names = list(args.patterns)
    if args.include_k_boundary:
        pattern_names.extend(name for name in K_BOUNDARY_PATTERNS if name not in pattern_names)
    if args.smoke_test:
        pattern_names = pattern_names[:2]
        strategies = list(args.strategies)[:2]
    else:
        strategies = list(args.strategies)
    patterns = selected_patterns(pattern_names)
    runs = _filter_runs(method_runs_for_week6(args.methods), args.seeds, smoke_test=args.smoke_test)
    assert_no_records500_in_runs(runs)
    save_predictions = bool(args.save_predictions or args.bootstrap)
    predictions_dir = output_dir / "predictions" if save_predictions else None

    metadata = base_metadata(runs, fill_mode="multiple_imputation_strategies", pattern_seed=WEEK6_PATTERN_SEED)
    metadata.update(
        {
            "analysis": "week6_reconstruction_imputation",
            "strategies": strategies,
            "patterns": pattern_metadata(patterns),
            "smoke_test": bool(args.smoke_test),
            "records500_used": False,
            "availability_mask_semantics": "original measured-lead mask is passed even when a lead is imputed or reconstructed",
        }
    )
    write_json(output_dir / "reconstruction_imputation_patterns.json", metadata)

    rows = []
    prediction_files = []
    for run in runs:
        for strategy in strategies:
            for pattern_name, pattern in patterns.items():
                row, pred_info = evaluate_week6_pattern(
                    run=run,
                    pattern_name=pattern_name,
                    pattern=pattern,
                    imputation_strategy=strategy,
                    split="test",
                    smoke_test=args.smoke_test,
                    save_predictions=save_predictions,
                    predictions_dir=predictions_dir,
                    prediction_method_id=run.method_run_id,
                )
                rows.append(row)
                if pred_info:
                    prediction_files.append(pred_info)

    metric_columns = [
        "method_id",
        "seed",
        "method_run_id",
        "pattern",
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
        "output_dir",
    ]
    write_csv(output_dir / "reconstruction_imputation_metrics.csv", rows, metric_columns)
    write_json(
        output_dir / "reconstruction_imputation_metrics.json",
        {**metadata, "rows": rows, "prediction_files": prediction_files},
    )
    write_markdown_table(output_dir / "reconstruction_imputation_metrics.md", rows, metric_columns[:15])

    deltas = delta_vs_baseline_with_fields(rows, group_fields=("seed", "pattern", "fill_mode"))
    delta_columns = [
        "method_id",
        "baseline_method",
        "seed",
        "pattern",
        "fill_mode",
        "delta_macro_auprc",
        "delta_macro_auroc",
        "delta_macro_f1",
        "delta_bce_nll",
    ]
    write_csv(output_dir / "reconstruction_imputation_delta_vs_A1.csv", deltas, delta_columns)
    write_markdown_table(output_dir / "reconstruction_imputation_delta_vs_A1.md", deltas, delta_columns)

    summary = summarize_multiseed(rows, group_cols=["method_id", "pattern", "fill_mode"])
    write_csv(output_dir / "reconstruction_imputation_multiseed_summary.csv", summary)
    write_markdown_table(
        output_dir / "reconstruction_imputation_multiseed_summary.md",
        summary,
        list(summary[0]) if summary else [],
    )

    bootstrap_rows = []
    if args.bootstrap:
        run_by_key = {(run.method_id, run.seed): run for run in runs}
        for run in runs:
            if run.method_id != "A4a_subclass_auxiliary":
                continue
            base = run_by_key.get(("A1_random_dropout", run.seed))
            if base is None:
                continue
            for strategy in strategies:
                bootstrap_rows.extend(
                    paired_bootstrap_prediction_delta(
                        predictions_dir=predictions_dir,
                        method_a_run_id=run.method_run_id,
                        method_b_run_id=base.method_run_id,
                        method_a="A4a_subclass_auxiliary",
                        method_b="A1_random_dropout",
                        seed=run.seed,
                        patterns=pattern_names,
                        fill_mode=strategy,
                        split="test",
                        n_bootstrap=20 if args.smoke_test else args.n_bootstrap,
                        bootstrap_seed=args.bootstrap_seed,
                    )
                )
        if bootstrap_rows:
            boot_cols = list(bootstrap_rows[0])
            write_csv(output_dir / "reconstruction_imputation_bootstrap_delta_ci.csv", bootstrap_rows, boot_cols)
            write_markdown_table(output_dir / "reconstruction_imputation_bootstrap_delta_ci.md", bootstrap_rows, boot_cols)
            write_json(
                output_dir / "reconstruction_imputation_bootstrap_run_config.json",
                {
                    "n_bootstrap": 20 if args.smoke_test else args.n_bootstrap,
                    "bootstrap_seed": args.bootstrap_seed,
                    "sampling_unit": "patient",
                    "comparison": "A4a_subclass_auxiliary vs A1_random_dropout",
                    "patterns": pattern_names,
                    "strategies": strategies,
                    "records500_used": False,
                },
            )

    report = [
        f"- Methods: {', '.join(args.methods)}",
        f"- Seeds used: {sorted({run.seed for run in runs})}",
        f"- Strategies: {', '.join(strategies)}",
        f"- Patterns: {', '.join(pattern_names)}",
        f"- Predictions saved: `{save_predictions}`",
        f"- Bootstrap rows: `{len(bootstrap_rows)}`",
        "- Physiology limb reconstruction uses raw-unit I/II formulas when I and II are available.",
        "- The original measured-lead availability mask is still passed to availability-aware models.",
        "- This is a supplementary reviewer-defense audit, not a new final method.",
    ]
    markdown_report(output_dir / "reconstruction_imputation_report.md", "Week 6 Reconstruction / Imputation Audit", report)


if __name__ == "__main__":
    main()
