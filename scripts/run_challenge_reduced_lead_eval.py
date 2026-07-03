#!/usr/bin/env python3
"""Run challenge-style reduced-lead evaluation from existing checkpoints only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.evaluation.evaluate_patterns import evaluate_missing_patterns
from hlm_ecg.evaluation.supplemental_analysis import (
    DEFAULT_METHODS,
    ROOT,
    add_method_run_fields,
    assert_no_records500_in_runs,
    base_metadata,
    delta_vs_baseline,
    discover_method_runs,
    figure_macro_auprc,
    load_yaml,
    markdown_report,
    paired_bootstrap_prediction_delta,
    summarize_multiseed,
    write_csv,
    write_json,
    write_markdown_table,
)
from hlm_ecg.evaluation.supplemental_patterns import (
    CHALLENGE_PATTERN_ORDER,
    SUPPLEMENTAL_PATTERN_SEED,
    challenge_reduced_lead_patterns,
    pattern_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing HLM-ECG checkpoints on challenge-style reduced leads.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--fill-mode", default="mean_fill", choices=["mean_fill", "zero_fill"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/week5_bibm_stabilization/challenge_reduced_leads"))
    parser.add_argument("--pattern-seed", type=int, default=SUPPLEMENTAL_PATTERN_SEED)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=SUPPLEMENTAL_PATTERN_SEED)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = challenge_reduced_lead_patterns(seed=args.pattern_seed)
    selected_patterns = list(CHALLENGE_PATTERN_ORDER)
    runs = discover_method_runs(args.methods)
    assert_no_records500_in_runs(runs)
    save_predictions = bool(args.save_predictions or args.bootstrap)
    predictions_dir = output_dir / "predictions" if save_predictions else None

    metadata = base_metadata(runs, fill_mode=args.fill_mode, pattern_seed=args.pattern_seed)
    metadata["patterns"] = pattern_metadata(patterns)
    metadata["analysis"] = "challenge_reduced_leads"
    write_json(output_dir / "challenge_patterns.json", metadata)

    all_rows = []
    prediction_files = []
    for run in runs:
        config = load_yaml(run.config_path)
        result = evaluate_missing_patterns(
            checkpoint_path=run.checkpoint_path,
            config=config,
            fill_mode=args.fill_mode,
            split="test",
            patterns=selected_patterns,
            pattern_registry=patterns,
            method_id=run.method_run_id,
            save_predictions=save_predictions,
            predictions_dir=predictions_dir,
            write_metrics=False,
            smoke_test=args.smoke_test,
        )
        all_rows.extend(add_method_run_fields(result["rows"], run))
        prediction_files.extend(result.get("prediction_files", []))

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
        "thresholds_source_split",
        "records500_used",
        "output_dir",
    ]
    write_csv(output_dir / "challenge_reduced_lead_metrics.csv", all_rows, metric_columns)
    write_json(
        output_dir / "challenge_reduced_lead_metrics.json",
        {
            **metadata,
            "rows": all_rows,
            "prediction_files": prediction_files,
            "smoke_test": bool(args.smoke_test),
        },
    )
    write_markdown_table(output_dir / "challenge_reduced_lead_metrics.md", all_rows, metric_columns[:10])

    deltas = delta_vs_baseline(all_rows, baseline_method="A1_random_dropout")
    delta_columns = [
        "method_id",
        "baseline_method",
        "seed",
        "pattern",
        "delta_macro_auprc",
        "delta_macro_auroc",
        "delta_macro_f1",
        "delta_bce_nll",
    ]
    write_csv(output_dir / "challenge_reduced_lead_delta_vs_A1.csv", deltas, delta_columns)
    write_markdown_table(output_dir / "challenge_reduced_lead_delta_vs_A1.md", deltas, delta_columns)

    summary = summarize_multiseed(all_rows, group_cols=["method_id", "pattern"])
    write_csv(output_dir / "challenge_reduced_lead_multiseed_summary.csv", summary)
    write_markdown_table(output_dir / "challenge_reduced_lead_multiseed_summary.md", summary, list(summary[0]) if summary else [])

    figure_macro_auprc(
        all_rows,
        path_prefix=output_dir / "figure_challenge_reduced_leads_macro_auprc",
        pattern_order=selected_patterns,
        method_order=list(args.methods),
        x_labels=["12", "6 limb", "4", "3", "2"],
        xlabel="Challenge-style available lead set",
        title="Challenge-style reduced-lead evaluation",
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
            bootstrap_rows.extend(
                paired_bootstrap_prediction_delta(
                    predictions_dir=predictions_dir,
                    method_a_run_id=run.method_run_id,
                    method_b_run_id=base.method_run_id,
                    method_a="A4a_subclass_auxiliary",
                    method_b="A1_random_dropout",
                    seed=run.seed,
                    patterns=selected_patterns[1:],
                    fill_mode=args.fill_mode,
                    split="test",
                    n_bootstrap=20 if args.smoke_test else args.n_bootstrap,
                    bootstrap_seed=args.bootstrap_seed,
                )
            )
        if bootstrap_rows:
            boot_columns = list(bootstrap_rows[0])
            write_csv(output_dir / "challenge_reduced_lead_bootstrap_delta_ci.csv", bootstrap_rows, boot_columns)
            write_markdown_table(output_dir / "challenge_reduced_lead_bootstrap_delta_ci.md", bootstrap_rows, boot_columns)
            write_json(
                output_dir / "challenge_reduced_lead_bootstrap_run_config.json",
                {
                    "n_bootstrap": 20 if args.smoke_test else args.n_bootstrap,
                    "bootstrap_seed": args.bootstrap_seed,
                    "sampling_unit": "patient",
                    "comparison": "A4a_subclass_auxiliary vs A1_random_dropout",
                    "patterns": selected_patterns[1:],
                    "records500_used": False,
                },
            )

    report_lines = [
        f"- Methods: {', '.join(args.methods)}",
        f"- Fill mode: `{args.fill_mode}`",
        f"- Smoke test: `{bool(args.smoke_test)}`",
        f"- Predictions saved: `{save_predictions}`",
        f"- Bootstrap rows: `{len(bootstrap_rows)}`",
        "- Outputs are supplementary and do not overwrite locked results.",
    ]
    markdown_report(output_dir / "challenge_reduced_lead_report.md", "Challenge Reduced-lead Evaluation", report_lines)


if __name__ == "__main__":
    main()
