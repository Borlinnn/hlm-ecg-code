#!/usr/bin/env python3
"""Audit full-lead preservation with patient-level paired bootstrap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.evaluation.supplemental_analysis import markdown_report, write_csv, write_json, write_markdown_table
from hlm_ecg.evaluation.week6_defense import (
    NONINFERIORITY_MARGIN_AUPRC,
    ROOT,
    WEEK6_DIR,
    WEEK6_PATTERN_SEED,
    noninferiority_decision,
)
from hlm_ecg.statistics.bootstrap import load_prediction_csv, paired_delta_summary, patient_groups, generate_patient_bootstrap_samples, sampled_indices_from_patients
from hlm_ecg.evaluation.supplemental_analysis import macro_auprc_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 full-lead preservation non-inferiority audit.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/week5_bibm_stabilization/challenge_reduced_leads/predictions"),
    )
    parser.add_argument("--output-dir", type=Path, default=WEEK6_DIR / "full_lead_preservation")
    parser.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 123])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=WEEK6_PATTERN_SEED)
    parser.add_argument("--margin", type=float, default=NONINFERIORITY_MARGIN_AUPRC)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _prediction_path(predictions_dir: Path, run_id: str) -> Path:
    return predictions_dir / run_id / "mean_fill" / "test" / "challenge_12_all.csv"


def main() -> None:
    args = parse_args()
    predictions_dir = args.predictions_dir if args.predictions_dir.is_absolute() else ROOT / args.predictions_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [42] if args.smoke_test else list(args.seeds)
    n_bootstrap = 20 if args.smoke_test else int(args.n_bootstrap)
    rows = []
    for seed in seeds:
        a_run = f"A4a_subclass_auxiliary_seed{seed}"
        b_run = f"A1_random_dropout_seed{seed}"
        path_a = _prediction_path(predictions_dir, a_run)
        path_b = _prediction_path(predictions_dir, b_run)
        if not path_a.exists() or not path_b.exists():
            rows.append(
                {
                    "seed": seed,
                    "status": "missing_predictions",
                    "method_a_run_id": a_run,
                    "method_b_run_id": b_run,
                    "path_a": str(path_a),
                    "path_b": str(path_b),
                }
            )
            continue
        data_a = load_prediction_csv(path_a, method_id=a_run, pattern="challenge_12_all", split="test", fill_mode="mean_fill")
        data_b = load_prediction_csv(path_b, method_id=b_run, pattern="challenge_12_all", split="test", fill_mode="mean_fill")
        if not (data_a.ecg_ids == data_b.ecg_ids).all():
            raise RuntimeError(f"ECG IDs differ for seed {seed}")
        if not (data_a.targets == data_b.targets).all():
            raise RuntimeError(f"Targets differ for seed {seed}")
        observed_delta = macro_auprc_from_logits(data_a.logits, data_a.targets) - macro_auprc_from_logits(
            data_b.logits, data_b.targets
        )
        groups = patient_groups(data_a.patient_ids)
        samples = generate_patient_bootstrap_samples(data_a.patient_ids, n_bootstrap=n_bootstrap, seed=args.bootstrap_seed)
        deltas = []
        for sampled_patients in samples:
            indices = sampled_indices_from_patients(groups, sampled_patients)
            deltas.append(
                macro_auprc_from_logits(data_a.logits[indices], data_a.targets[indices])
                - macro_auprc_from_logits(data_b.logits[indices], data_b.targets[indices])
            )
        summary = paired_delta_summary(deltas, observed_delta)
        rows.append(
            {
                "seed": seed,
                "status": "ok",
                "method_a": "A4a_subclass_auxiliary",
                "method_b": "A1_random_dropout",
                "method_a_run_id": a_run,
                "method_b_run_id": b_run,
                "pattern": "challenge_12_all",
                "metric": "macro_auprc",
                "n_bootstrap": n_bootstrap,
                "bootstrap_seed": int(args.bootstrap_seed),
                "sampling_unit": "patient",
                "noninferiority_margin": float(args.margin),
                "noninferiority_decision": noninferiority_decision(float(summary["ci_low"]), margin=args.margin),
                **summary,
            }
        )

    cols = [
        "seed",
        "status",
        "method_a",
        "method_b",
        "pattern",
        "metric",
        "observed_delta",
        "ci_low",
        "ci_high",
        "noninferiority_margin",
        "noninferiority_decision",
        "bootstrap_mean_delta",
        "probability_delta_gt_0",
        "p_two_sided",
        "n_bootstrap",
        "n_bootstrap_valid",
        "invalid_replicates",
        "sampling_unit",
    ]
    write_csv(output_dir / "full_lead_preservation_noninferiority.csv", rows, cols)
    write_json(
        output_dir / "full_lead_preservation_noninferiority.json",
        {
            "created_from": str(predictions_dir.relative_to(ROOT)) if predictions_dir.is_relative_to(ROOT) else str(predictions_dir),
            "margin": float(args.margin),
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": int(args.bootstrap_seed),
            "records500_used": False,
            "rows": rows,
        },
    )
    write_markdown_table(output_dir / "full_lead_preservation_noninferiority.md", rows, cols)
    ok = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "- Comparison: `A4a_subclass_auxiliary - A1_random_dropout` on full 12-lead (`challenge_12_all`).",
        f"- Non-inferiority margin: `{float(args.margin):.4f}` Macro AUPRC.",
        f"- Sampling unit: patient; n_bootstrap: `{n_bootstrap}`.",
        f"- Seeds evaluated: `{[row.get('seed') for row in ok]}`.",
        "- Decision rule: CI lower bound must be greater than the pre-specified margin.",
    ]
    for row in ok:
        lines.append(
            f"- Seed {row['seed']}: delta={row['observed_delta']:.4f}, "
            f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
            f"decision={row['noninferiority_decision']}."
        )
    markdown_report(output_dir / "full_lead_preservation_report.md", "Week 6 Full-lead Preservation Audit", lines)


if __name__ == "__main__":
    main()
