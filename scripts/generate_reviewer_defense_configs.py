#!/usr/bin/env python3
"""Generate reviewer-defense experiment configs without launching training."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SEEDS = (7, 42, 123, 2024, 2025)
PRIMARY_BACKBONES = ("resnet1d_tiny", "xresnet1d101_like", "inception_time1d")
APPENDIX_BACKBONES = ("resnet1d_tiny", "xresnet1d101_like")
SHORT_SEEDS = (7, 42, 123)
CONFIG_ROOT = Path("configs/reviewer_defense_20260701")
OUTPUT_ROOT = Path("outputs/reviewer_defense_20260701")

PRIMARY_METHODS = (
    "M0_full_no_masking",
    "M1_random_dropout",
    "M2_structured_masking",
    "M3_random_dropout_plus_availability",
    "M4_structured_plus_availability",
    "M6_structured_plus_availability_plus_subclass",
)
APPENDIX_METHODS = (
    "M5_structured_plus_subclass_no_availability",
    "M7_M6_plus_hierarchy",
    "M8_M7_plus_confidence_weighted_consistency",
)
SPECIALIST_PATTERNS = (
    "limb_only__precordial_missing",
    "precordial_only__limb_missing",
    "V1_V3_missing",
    "V4_V6_missing",
    "challenge_6_limb",
    "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2",
    "challenge_2_I_II",
)
SPECIALIST_PATTERN_CANONICAL = {
    "limb_only__precordial_missing": "limb-only / precordial-missing",
    "precordial_only__limb_missing": "precordial-only / limb-missing",
    "V1_V3_missing": "V1-V3 missing",
    "V4_V6_missing": "V4-V6 missing",
    "challenge_6_limb": "challenge_6_limb",
    "challenge_4_I_II_III_V2": "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2": "challenge_3_I_II_V2",
    "challenge_2_I_II": "challenge_2_I_II",
}


def backbone_defaults(backbone: str) -> dict[str, Any]:
    if backbone == "resnet1d_tiny":
        return {"architecture": backbone, "base_channels": 16, "layers": [1, 1, 1, 1], "kernel_size": 7}
    if backbone == "xresnet1d101_like":
        return {"architecture": backbone, "base_channels": 16, "layers": [3, 4, 23, 3], "kernel_size": 7}
    if backbone == "inception_time1d":
        return {
            "architecture": backbone,
            "base_channels": 32,
            "inception_depth": 6,
            "inception_bottleneck_channels": 32,
        }
    raise ValueError(f"Unsupported backbone: {backbone}")


def train_script_for(method_id: str) -> str:
    if method_id == "M0_full_no_masking":
        return "scripts/train_full_baseline.py"
    if method_id in {"M1_random_dropout", "M3_random_dropout_plus_availability"}:
        return "scripts/train_random_dropout.py"
    if method_id == "M2_structured_masking":
        return "scripts/train_structured_masking.py"
    if method_id == "M4_structured_plus_availability":
        return "scripts/train_availability_embedding.py"
    if method_id in {
        "M5_structured_plus_subclass_no_availability",
        "M6_structured_plus_availability_plus_subclass",
    }:
        return "scripts/train_subclass_auxiliary.py"
    if method_id == "M7_M6_plus_hierarchy":
        return "scripts/train_hierarchy_ablation.py"
    if method_id == "M8_M7_plus_confidence_weighted_consistency":
        return "scripts/train_confidence_consistency.py"
    raise ValueError(f"Unsupported method: {method_id}")


def eval_script_for(method_id: str) -> str:
    if method_id == "M0_full_no_masking":
        return "scripts/evaluate_full_baseline_patterns.py"
    if method_id in {"M1_random_dropout", "M3_random_dropout_plus_availability"}:
        return "scripts/evaluate_random_dropout_patterns.py"
    if method_id in {"M2_structured_masking", "M5_structured_plus_subclass_no_availability"}:
        return "scripts/evaluate_structured_masking_patterns.py"
    if method_id == "M4_structured_plus_availability":
        return "scripts/evaluate_availability_embedding_patterns.py"
    if method_id == "M7_M6_plus_hierarchy":
        return "scripts/evaluate_hierarchy_ablation_patterns.py"
    return "scripts/evaluate_subclass_auxiliary_patterns.py"


def safe_name(value: str) -> str:
    return value.replace(" / ", "_").replace("-", "_").replace(" ", "_").replace(".", "p")


def make_row(group: str, method_id: str, backbone: str, seed: int, *, tag: str = "") -> dict[str, Any]:
    suffix = f"_{tag}" if tag else ""
    rel = Path(group) / backbone / f"{method_id}{suffix}_seed{seed}.yaml"
    output_dir = OUTPUT_ROOT / group / backbone / f"{method_id}{suffix}" / f"seed{seed}"
    return {
        "group": group,
        "method_id": method_id,
        "backbone": backbone,
        "seed": int(seed),
        "tag": tag,
        "config_path": str(CONFIG_ROOT / rel),
        "output_dir": str(output_dir),
        "train_script": train_script_for(method_id) if not method_id.startswith("SPECIALIST") else "scripts/train_week6_fixed_pattern_specialist.py",
        "eval_script": eval_script_for(method_id) if not method_id.startswith("SPECIALIST") else "scripts/evaluate_structured_masking_patterns.py",
    }


def build_experiment_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in PRIMARY_METHODS:
        for backbone in PRIMARY_BACKBONES:
            for seed in SEEDS:
                rows.append(make_row("primary", method_id, backbone, seed))
    for method_id in APPENDIX_METHODS:
        for backbone in APPENDIX_BACKBONES:
            for seed in SEEDS:
                rows.append(make_row("appendix", method_id, backbone, seed))
    for prob in (0.25, 0.5, 0.75, 1.0):
        for seed in SHORT_SEEDS:
            rows.append(make_row("sensitivity_mask_mix", "M2_structured_masking", "xresnet1d101_like", seed, tag=f"structured_prob_{prob}"))
    for lambda_sub in (0.0, 0.1, 0.2, 0.5):
        for seed in SHORT_SEEDS:
            rows.append(
                make_row(
                    "sensitivity_subclass_lambda",
                    "M6_structured_plus_availability_plus_subclass",
                    "xresnet1d101_like",
                    seed,
                    tag=f"lambda_sub_{lambda_sub}",
                )
            )
    for fill_mode in ("mean_fill", "zero_fill", "learnable_mask_token"):
        for seed in SHORT_SEEDS:
            rows.append(
                make_row(
                    "sensitivity_fill_representation",
                    "M2_structured_masking",
                    "xresnet1d101_like",
                    seed,
                    tag=fill_mode,
                )
            )
    for pattern in SPECIALIST_PATTERNS:
        for seed in SHORT_SEEDS:
            row = make_row("specialist_upper_bound", "SPECIALIST_fixed_pattern", "xresnet1d101_like", seed, tag=pattern)
            row["specialist_pattern"] = SPECIALIST_PATTERN_CANONICAL[pattern]
            rows.append(row)
    return rows


def base_config(row: dict[str, Any]) -> dict[str, Any]:
    model = {
        "in_channels": 12,
        "num_classes": 5,
        "signal_length": 1000,
        **backbone_defaults(str(row["backbone"])),
    }
    return {
        "seed": int(row["seed"]),
        "device": "auto",
        "paths": {
            "data_root": "data/ptb-xl",
            "day1_index": "outputs/day1_audit/ptbxl_day1_index.csv",
            "norm_stats": "outputs/day1_audit/train_norm_stats.npz",
            "output_dir": row["output_dir"],
        },
        "model": model,
        "training": {
            "batch_size": 32,
            "num_workers": 0,
            "max_epochs": 30,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "reduce_on_plateau",
            "early_stopping_patience": 8,
        },
        "evaluation": {"batch_size": 64, "num_workers": 0, "pattern_seed": 20240604},
        "smoke": {"train_limit": 64, "val_limit": 64, "test_limit": 64},
        "reviewer_defense": {
            "group": row["group"],
            "method_id": row["method_id"],
            "backbone": row["backbone"],
            "seed": int(row["seed"]),
            "tag": row.get("tag", ""),
            "threshold_source_split": "val",
        },
    }


def add_random_dropout(config: dict[str, Any], *, fill_mode: str = "mean_fill") -> None:
    config["train_augmentation"] = {
        "enabled": True,
        "missing_counts": [0, 1, 3, 6],
        "probabilities": [0.25, 0.25, 0.25, 0.25],
        "fill_mode": fill_mode,
        "min_available_leads": 1,
        "seed": int(config["seed"]),
    }


def add_structured_masking(config: dict[str, Any], *, structured_prob: float = 0.5, fill_mode: str = "mean_fill") -> None:
    config["structured_masking"] = {
        "enabled": True,
        "fill_mode": fill_mode,
        "random_missing_counts": [0, 1, 3, 6],
        "random_prob": max(0.0, 1.0 - float(structured_prob)),
        "structured_prob": float(structured_prob),
        "structured_patterns": [
            "limb_only__precordial_missing",
            "precordial_only__limb_missing",
            "V1_V3_missing",
            "V4_V6_missing",
        ],
        "min_available_leads": 1,
        "seed": int(config["seed"]),
    }


def enable_availability(config: dict[str, Any]) -> None:
    config["model"]["use_availability_embedding"] = True
    config["model"]["availability_embedding_dim"] = 32
    config["model"]["mask_mlp_hidden_dim"] = 32


def enable_subclass(config: dict[str, Any], *, lambda_sub: float = 0.2, allow_without_availability: bool = False) -> None:
    config["model"]["enable_subclass_auxiliary"] = True
    config["subclass_auxiliary"] = {
        "enabled": True,
        "min_train_pos": 50,
        "lambda_sub": float(lambda_sub),
        "subclass_loss_ignore_only_dropped": True,
        "allow_without_availability": bool(allow_without_availability),
    }


def enable_hierarchy(config: dict[str, Any]) -> None:
    config["model"]["use_hierarchy_loss"] = True
    config["hierarchy_loss"] = {"enabled": True, "lambda_hier": 0.1, "hierarchy_violation_eps": 0.0}


def enable_consistency(config: dict[str, Any]) -> None:
    config["model"]["use_confidence_weighted_consistency"] = True
    config["model"]["lambda_cons"] = 0.1
    config["confidence_consistency"] = {"enabled": True, "lambda_cons": 0.1, "consistency_gamma": 1.0}


def build_config(row: dict[str, Any]) -> dict[str, Any]:
    config = base_config(row)
    method_id = str(row["method_id"])
    tag = str(row.get("tag", ""))
    fill_mode = "mean_fill"
    structured_prob = 0.5
    lambda_sub = 0.2

    if tag.startswith("structured_prob_"):
        structured_prob = float(tag.replace("structured_prob_", ""))
    if tag.startswith("lambda_sub_"):
        lambda_sub = float(tag.replace("lambda_sub_", ""))
    if tag == "zero_fill":
        fill_mode = "zero_fill"
    if tag == "learnable_mask_token":
        config["model"]["use_learnable_mask_token"] = True

    if method_id == "M0_full_no_masking":
        return config
    if method_id == "M1_random_dropout":
        add_random_dropout(config, fill_mode=fill_mode)
        return config
    if method_id in {"M2_structured_masking", "SPECIALIST_fixed_pattern"}:
        add_structured_masking(config, structured_prob=structured_prob, fill_mode=fill_mode)
        if method_id == "SPECIALIST_fixed_pattern":
            config["week6_specialist"] = {
                "pattern": row["specialist_pattern"],
                "imputation_strategy": "mean_fill",
            }
        return config
    if method_id == "M3_random_dropout_plus_availability":
        add_random_dropout(config, fill_mode=fill_mode)
        enable_availability(config)
        return config
    if method_id == "M4_structured_plus_availability":
        add_structured_masking(config, structured_prob=structured_prob, fill_mode=fill_mode)
        enable_availability(config)
        return config
    if method_id == "M5_structured_plus_subclass_no_availability":
        add_structured_masking(config, structured_prob=structured_prob, fill_mode=fill_mode)
        enable_subclass(config, lambda_sub=lambda_sub, allow_without_availability=True)
        return config
    if method_id == "M6_structured_plus_availability_plus_subclass":
        add_structured_masking(config, structured_prob=structured_prob, fill_mode=fill_mode)
        enable_availability(config)
        enable_subclass(config, lambda_sub=lambda_sub)
        return config
    if method_id == "M7_M6_plus_hierarchy":
        add_structured_masking(config, structured_prob=structured_prob, fill_mode=fill_mode)
        enable_availability(config)
        enable_subclass(config, lambda_sub=lambda_sub)
        enable_hierarchy(config)
        return config
    if method_id == "M8_M7_plus_confidence_weighted_consistency":
        add_structured_masking(config, structured_prob=structured_prob, fill_mode=fill_mode)
        enable_availability(config)
        enable_subclass(config, lambda_sub=lambda_sub)
        enable_hierarchy(config)
        enable_consistency(config)
        return config
    raise ValueError(f"Unsupported method_id: {method_id}")


def write_configs(*, output_dir: Path = CONFIG_ROOT, dry_run: bool = False) -> dict[str, Any]:
    rows = build_experiment_plan()
    written = []
    for row in rows:
        rel_path = Path(row["config_path"]).relative_to(CONFIG_ROOT)
        path = output_dir / rel_path
        if dry_run:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        config = build_config(row)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        row = deepcopy(row)
        row["config_path"] = str(path)
        written.append(row)
    manifest = {
        "dry_run": bool(dry_run),
        "n_configs": len(rows),
        "n_written": len(written),
        "config_root": str(output_dir),
        "output_root": str(OUTPUT_ROOT),
        "seeds": list(SEEDS),
        "primary_backbones": list(PRIMARY_BACKBONES),
        "primary_methods": list(PRIMARY_METHODS),
        "rows": rows if dry_run else written,
    }
    if not dry_run:
        (output_dir / "reviewer_defense_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = write_configs(output_dir=args.output_dir, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
