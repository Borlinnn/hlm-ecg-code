#!/usr/bin/env python3
"""Create gated Week 6 fixed-pattern specialist baseline scaffolds.

By default this script does not train. It writes configs and a report explaining
how specialist baselines would be run if the human explicitly enables
`WEEK6_ALLOW_SPECIALIST_TRAINING=true`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from hlm_ecg.evaluation.supplemental_analysis import markdown_report, write_json
from hlm_ecg.evaluation.prediction_artifacts import safe_pattern_name
from hlm_ecg.evaluation.week6_defense import ROOT, WEEK6_DIR, specialist_training_allowed

SPECIALIST_PATTERNS = (
    "challenge_6_limb",
    "challenge_2_I_II",
    "precordial-only / limb-missing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Week 6 specialist baseline scaffolds.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/week6_specialists"))
    parser.add_argument("--output-dir", type=Path, default=WEEK6_DIR / "fixed_pattern_specialists")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _base_config(pattern: str, seed: int) -> dict[str, object]:
    safe_pattern = safe_pattern_name(pattern)
    return {
        "seed": int(seed),
        "week6_specialist": {
            "pattern": pattern,
            "enabled": False,
            "training_gate": "WEEK6_ALLOW_SPECIALIST_TRAINING=true",
            "note": "Scaffold only; do not train unless explicitly enabled by the human.",
        },
        "paths": {
            "data_root": "data/ptb-xl",
            "day1_index": "outputs/day1_audit/ptbxl_day1_index.csv",
            "norm_stats": "outputs/day1_audit/train_norm_stats.npz",
            "output_dir": f"outputs/week6_reviewer_defense/fixed_pattern_specialists/{safe_pattern}/seed{seed}",
        },
        "model": {
            "in_channels": 12,
            "num_classes": 5,
            "base_channels": 32,
            "layers": [1, 1, 1, 1],
            "kernel_size": 7,
            "use_availability_embedding": False,
            "enable_subclass_auxiliary": False,
        },
        "training": {
            "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
            "val_fold": 9,
            "test_fold": 10,
            "early_stopping_metric": "val_macro_auprc",
            "threshold_source_split": "val",
            "batch_size": 64,
            "num_workers": 0,
            "max_epochs": 30,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "reduce_on_plateau",
            "early_stopping_patience": 8,
            "note": "No hyperparameter tuning on test fold.",
        },
        "evaluation": {
            "batch_size": 64,
            "num_workers": 0,
            "pattern_seed": 20240606,
        },
        "records500_used": False,
    }


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir if args.config_dir.is_absolute() else ROOT / args.config_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = specialist_training_allowed()
    generated = []
    for pattern in SPECIALIST_PATTERNS:
        path = config_dir / f"{safe_pattern_name(pattern)}_seed{args.seed}.yaml"
        config = _base_config(pattern, args.seed)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        generated.append(str(path.relative_to(ROOT)))

    status = {
        "records500_used": False,
        "specialist_training_allowed": allowed,
        "gate": "WEEK6_ALLOW_SPECIALIST_TRAINING=true",
        "generated_configs": generated,
        "patterns": list(SPECIALIST_PATTERNS),
    }
    write_json(output_dir / "specialist_scaffold_status.json", status)
    lines = [
        "- Fixed-pattern specialist baseline training is gated.",
        f"- `WEEK6_ALLOW_SPECIALIST_TRAINING` currently evaluates to `{os.environ.get('WEEK6_ALLOW_SPECIALIST_TRAINING', 'false')}`.",
        f"- Specialist training allowed: `{allowed}`.",
        "- No training command was launched by this scaffold.",
        "- Generated configs:",
        *[f"  - `{path}`" for path in generated],
        "",
        "Dry-run command example:",
        "",
        "```bash",
        "python3 scripts/train_week6_fixed_pattern_specialist.py \\",
        f"  --config {generated[0]} \\",
        "  --dry-run",
        "```",
        "",
        "Future gated training command example, only after explicit human approval:",
        "",
        "```bash",
        "WEEK6_ALLOW_SPECIALIST_TRAINING=true \\",
        "python3 scripts/train_week6_fixed_pattern_specialist.py \\",
        f"  --config {generated[0]}",
        "```",
    ]
    if not allowed:
        lines.append("- To run specialists later, the human must explicitly enable the gate and approve training.")
    markdown_report(output_dir / "specialist_scaffold_report.md", "Week 6 Fixed-pattern Specialist Scaffold", lines)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
