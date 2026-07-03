#!/usr/bin/env python3
"""Per-class operating-boundary analysis for Week 6 reviewer defense."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.metrics import average_precision_score

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.prediction_artifacts import safe_pattern_name
from hlm_ecg.evaluation.supplemental_analysis import markdown_report, write_csv, write_json, write_markdown_table
from hlm_ecg.evaluation.week6_defense import HARD_STRUCTURED_PATTERNS, ROOT, WEEK6_DIR
from hlm_ecg.statistics.bootstrap import load_prediction_csv

CHALLENGE_PATTERNS = (
    "challenge_6_limb",
    "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2",
    "challenge_2_I_II",
)
K_PATTERNS = (
    "k6_visible_random",
    "k4_visible_random",
    "k3_visible_random",
    "k2_visible_random",
    "k1_visible_random",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 per-class boundary analysis.")
    parser.add_argument("--output-dir", type=Path, default=WEEK6_DIR / "per_class_boundary_analysis")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _per_class_auprc(path: Path, method_id: str, pattern: str) -> dict[str, float]:
    data = load_prediction_csv(path, method_id=method_id, pattern=pattern, split="test", fill_mode="mean_fill")
    out = {}
    for idx, label in enumerate(LABEL_ORDER):
        y_true = data.targets[:, idx]
        if np.unique(y_true).size < 2:
            out[label] = float("nan")
        else:
            out[label] = float(average_precision_score(y_true, data.probs[:, idx]))
    return out


def _rows_from_pair(source: str, predictions_dir: Path, pattern: str, seed: int, a_run: str, b_run: str) -> list[dict[str, object]]:
    path_a = predictions_dir / a_run / "mean_fill" / "test" / f"{safe_pattern_name(pattern)}.csv"
    path_b = predictions_dir / b_run / "mean_fill" / "test" / f"{safe_pattern_name(pattern)}.csv"
    if not path_a.exists() or not path_b.exists():
        return [
            {
                "source": source,
                "seed": seed,
                "pattern": pattern,
                "label": "MISSING",
                "status": "missing_prediction",
                "path_a": str(path_a),
                "path_b": str(path_b),
            }
        ]
    a = _per_class_auprc(path_a, a_run, pattern)
    b = _per_class_auprc(path_b, b_run, pattern)
    rows = []
    for label in LABEL_ORDER:
        rows.append(
            {
                "source": source,
                "seed": seed,
                "pattern": pattern,
                "label": label,
                "status": "ok",
                "A4a_auprc": a[label],
                "A1_auprc": b[label],
                "delta_A4a_minus_A1": a[label] - b[label],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    hard_dir = ROOT / "outputs/week3_results_lock/predictions"
    hard_patterns = list(HARD_STRUCTURED_PATTERNS[:1] if args.smoke_test else HARD_STRUCTURED_PATTERNS)
    for pattern in hard_patterns:
        rows.extend(
            _rows_from_pair(
                "week3_hard_structured",
                hard_dir,
                pattern,
                42,
                "A4a_subclass_auxiliary",
                "A1_random_dropout",
            )
        )

    challenge_dir = ROOT / "outputs/week5_bibm_stabilization/challenge_reduced_leads/predictions"
    k_dir = ROOT / "outputs/week5_bibm_stabilization/k_visible_curve/predictions"
    seeds = [42] if args.smoke_test else [7, 42, 123]
    challenge_patterns = list(CHALLENGE_PATTERNS[:1] if args.smoke_test else CHALLENGE_PATTERNS)
    k_patterns = list(K_PATTERNS[:1] if args.smoke_test else K_PATTERNS)
    for seed in seeds:
        a_run = f"A4a_subclass_auxiliary_seed{seed}"
        b_run = f"A1_random_dropout_seed{seed}"
        for pattern in challenge_patterns:
            rows.extend(_rows_from_pair("week5_challenge", challenge_dir, pattern, seed, a_run, b_run))
        for pattern in k_patterns:
            rows.extend(_rows_from_pair("week5_k_visible", k_dir, pattern, seed, a_run, b_run))

    columns = ["source", "seed", "pattern", "label", "status", "A4a_auprc", "A1_auprc", "delta_A4a_minus_A1"]
    write_csv(output_dir / "per_class_boundary_delta.csv", rows, columns)
    write_json(
        output_dir / "per_class_boundary_delta.json",
        {
            "records500_used": False,
            "comparison": "A4a_subclass_auxiliary - A1_random_dropout",
            "metric": "per_class_auprc",
            "rows": rows,
        },
    )
    write_markdown_table(output_dir / "per_class_boundary_delta.md", rows, columns)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    negative_k1 = [
        row for row in ok_rows if row["pattern"] == "k1_visible_random" and float(row["delta_A4a_minus_A1"]) < 0
    ]
    worst = sorted(negative_k1, key=lambda row: float(row["delta_A4a_minus_A1"]))[:8]
    guard = {
        "records500_used": False,
        "k1_negative_rows": len(negative_k1),
        "worst_k1_class_deltas": worst,
        "operating_boundary": (
            "A4a is not claimed to dominate arbitrary one-visible-lead settings; "
            "k=1 is reported as an operating boundary / stress-test failure mode."
        ),
    }
    write_json(output_dir / "operating_boundary_guard.json", guard)
    if worst:
        write_markdown_table(
            output_dir / "operating_boundary_guard.md",
            worst,
            ["source", "seed", "pattern", "label", "A4a_auprc", "A1_auprc", "delta_A4a_minus_A1"],
        )
    else:
        (output_dir / "operating_boundary_guard.md").write_text("No negative k1 rows found.\n", encoding="utf-8")

    lines = [
        "- This analysis uses saved prediction CSVs only; no training was run.",
        "- Hard structured rows use Week 3 result-lock predictions for seed42.",
        "- Challenge and k-visible rows use Week 5 predictions for seeds 7/42/123.",
        "- `k1_visible_random` is treated as an operating-boundary stress test, not hidden as a success case.",
        f"- Negative k1 per-class rows: `{len(negative_k1)}`.",
    ]
    if worst:
        lines.append(
            "- Worst k1 deltas: "
            + "; ".join(
                f"seed {row['seed']} {row['label']} {float(row['delta_A4a_minus_A1']):+.4f}"
                for row in worst[:5]
            )
        )
    markdown_report(output_dir / "per_class_boundary_report.md", "Week 6 Per-class Boundary Analysis", lines)


if __name__ == "__main__":
    main()
