#!/usr/bin/env python3
"""Save evaluation-only per-sample prediction artifacts for locked HLM-ECG methods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.evaluation.evaluate_patterns import evaluate_missing_patterns


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    checkpoint: Path
    config: Path
    output_dir: Path


METHOD_REGISTRY = {
    "A0_full_no_masking": MethodSpec(
        "A0_full_no_masking",
        Path("outputs/week1_full_baseline/full_seed42/best_model.pt"),
        Path("configs/full_baseline.yaml"),
        Path("outputs/week1_full_baseline/full_seed42"),
    ),
    "A1_random_dropout": MethodSpec(
        "A1_random_dropout",
        Path("outputs/week1_random_dropout/random_dropout_seed42/best_model.pt"),
        Path("configs/random_dropout.yaml"),
        Path("outputs/week1_random_dropout/random_dropout_seed42"),
    ),
    "A2_structured_masking": MethodSpec(
        "A2_structured_masking",
        Path("outputs/week2_structured_masking/structured_seed42/best_model.pt"),
        Path("configs/structured_masking.yaml"),
        Path("outputs/week2_structured_masking/structured_seed42"),
    ),
    "A4a_subclass_auxiliary": MethodSpec(
        "A4a_subclass_auxiliary",
        Path("outputs/week2_subclass_auxiliary/subclass_aux_seed42/best_model.pt"),
        Path("configs/subclass_auxiliary.yaml"),
        Path("outputs/week2_subclass_auxiliary/subclass_aux_seed42"),
    ),
    "A5_lite_confidence_consistency_0p05": MethodSpec(
        "A5_lite_confidence_consistency_0p05",
        Path("outputs/week2_confidence_consistency_lite/consistency_lite_seed42/best_model.pt"),
        Path("configs/confidence_consistency_lite.yaml"),
        Path("outputs/week2_confidence_consistency_lite/consistency_lite_seed42"),
    ),
}

ALL_PATTERNS = [
    "full",
    "random-1",
    "random-3",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
]


def resolve_method(spec: MethodSpec) -> MethodSpec:
    checkpoint = spec.checkpoint
    config = spec.config
    if not checkpoint.exists():
        candidates = sorted(spec.output_dir.glob("best_model.pt"))
        if len(candidates) == 1:
            checkpoint = candidates[0]
        else:
            raise FileNotFoundError(f"Checkpoint not found for {spec.method_id}: {spec.checkpoint}")
    if not config.exists():
        candidates = sorted(spec.output_dir.glob("config_used.yaml"))
        if len(candidates) == 1:
            config = candidates[0]
        else:
            raise FileNotFoundError(f"Config not found for {spec.method_id}: {spec.config}")
    return MethodSpec(spec.method_id, checkpoint, config, spec.output_dir)


def load_config(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_used = output_dir / "config_used.yaml"
    if config_used.exists():
        config = yaml.safe_load(config_used.read_text(encoding="utf-8"))
    config.setdefault("paths", {})["output_dir"] = str(output_dir)
    return config


def write_manifest(predictions_dir: Path, entries: list[dict[str, Any]]) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    csv_path = predictions_dir / "prediction_manifest.csv"
    fieldnames = [
        "method_id",
        "split",
        "fill_mode",
        "pattern",
        "csv_path",
        "n_rows",
        "n_labels",
        "has_logits",
        "has_probabilities",
        "has_thresholds",
        "threshold_source_split",
        "file_size",
        "sha256",
        "created_at",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: entry.get(key) for key in fieldnames} for entry in entries])
    (predictions_dir / "prediction_manifest.json").write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Prediction Manifest",
        "",
        f"- entries: `{len(entries)}`",
        "",
        "| method | split | fill | pattern | rows | path |",
        "|---|---|---|---|---:|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['method_id']} | {entry['split']} | {entry['fill_mode']} | "
            f"{entry['pattern']} | {entry['n_rows']} | `{entry['csv_path']}` |"
        )
    (predictions_dir / "prediction_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save HLM-ECG prediction artifacts without training.")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--fill-modes", nargs="+", default=["mean_fill"])
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--patterns", nargs="+", default=["all"])
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if Path("data/ptb-xl/records500").exists():
        raise RuntimeError("records500 exists; refusing to proceed")
    patterns = ALL_PATTERNS if args.patterns == ["all"] else args.patterns
    unknown_methods = set(args.methods).difference(METHOD_REGISTRY)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")

    manifest_entries: list[dict[str, Any]] = []
    used_paths = []
    for method_id in args.methods:
        spec = resolve_method(METHOD_REGISTRY[method_id])
        config = load_config(spec.config, spec.output_dir)
        used_paths.append(
            {
                "method_id": method_id,
                "checkpoint": str(spec.checkpoint),
                "config": str(spec.config),
                "output_dir": str(spec.output_dir),
            }
        )
        for split in args.splits:
            for fill_mode in args.fill_modes:
                result = evaluate_missing_patterns(
                    checkpoint_path=spec.checkpoint,
                    config=config,
                    fill_mode=fill_mode,
                    split=split,
                    patterns=patterns,
                    method_id=method_id,
                    save_predictions=True,
                    predictions_dir=args.predictions_dir,
                    write_metrics=False,
                    smoke_test=args.smoke_test,
                )
                manifest_entries.extend(result["prediction_files"])
    write_manifest(args.predictions_dir, manifest_entries)
    print(
        json.dumps(
            {
                "predictions_dir": str(args.predictions_dir),
                "manifest_entries": len(manifest_entries),
                "used_paths": used_paths,
                "smoke_test": bool(args.smoke_test),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
