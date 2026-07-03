#!/usr/bin/env python3
"""Evaluate the availability embedding ablation under missing-lead patterns."""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.evaluation.evaluate_patterns import evaluate_missing_patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate availability embedding missing-pattern robustness.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/availability_embedding.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--fill-modes", nargs="+", default=["zero_fill", "mean_fill"])
    parser.add_argument("--patterns", nargs="+", default=["all"])
    parser.add_argument("--method-id", default="A3_availability_embedding")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = str(args.output_dir)

    selected_patterns = None if args.patterns == ["all"] else args.patterns
    results = {}
    for split in args.splits:
        results[split] = {}
        for fill_mode in args.fill_modes:
            results[split][fill_mode] = evaluate_missing_patterns(
                checkpoint_path=args.checkpoint,
                config=config,
                fill_mode=fill_mode,
                split=split,
                patterns=selected_patterns,
                method_id=args.method_id,
                save_predictions=args.save_predictions,
                predictions_dir=args.predictions_dir,
                write_metrics=not args.save_predictions,
                smoke_test=args.smoke_test,
            )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
