#!/usr/bin/env python3
"""Run one generated reviewer-defense config through train/eval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_reviewer_defense_configs import eval_script_for, train_script_for


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def method_id_from_config(config: dict[str, Any]) -> str:
    reviewer = dict(config.get("reviewer_defense", {}))
    method_id = str(reviewer.get("method_id", ""))
    if not method_id:
        raise RuntimeError("Config missing reviewer_defense.method_id")
    return method_id


def command_plan(config_path: Path, *, smoke_test: bool, save_predictions: bool) -> dict[str, Any]:
    config = load_config(config_path)
    method_id = method_id_from_config(config)
    reviewer = dict(config.get("reviewer_defense", {}))
    tag = str(reviewer.get("tag", ""))
    method_run_id = f"{method_id}_{reviewer.get('backbone', 'unknown')}_seed{reviewer.get('seed', 'unknown')}"
    if tag:
        method_run_id = f"{method_run_id}_{tag}"
    configured_output_dir = Path(dict(config.get("paths", {})).get("output_dir", ""))
    if not configured_output_dir:
        raise RuntimeError("Config missing paths.output_dir")
    output_dir = configured_output_dir
    if smoke_test:
        group = str(reviewer.get("group", "unknown_group"))
        backbone = str(reviewer.get("backbone", "unknown_backbone"))
        output_dir = Path("outputs/reviewer_defense_20260701/smoke") / group / backbone / method_run_id
    if method_id == "SPECIALIST_fixed_pattern":
        train_script = "scripts/train_week6_fixed_pattern_specialist.py"
        eval_script = ""
    else:
        train_script = train_script_for(method_id)
        eval_script = eval_script_for(method_id)
    train_cmd = [sys.executable, train_script, "--config", str(config_path), "--output-dir", str(output_dir)]
    if smoke_test:
        train_cmd.append("--smoke-test")
    eval_cmd = []
    if method_id != "SPECIALIST_fixed_pattern":
        eval_cmd = [
            sys.executable,
            eval_script,
            "--checkpoint",
            str(output_dir / "best_model.pt"),
            "--config",
            str(output_dir / "config_used.yaml"),
            "--output-dir",
            str(output_dir),
            "--fill-modes",
            "mean_fill",
            "zero_fill",
            "--method-id",
            method_run_id,
        ]
        if smoke_test:
            eval_cmd.append("--smoke-test")
        if save_predictions:
            pred_root = Path("results/reviewer_defense_20260701/predictions")
            if smoke_test:
                pred_root = Path("results/reviewer_defense_20260701/smoke_predictions")
            eval_cmd.extend(["--save-predictions", "--predictions-dir", str(pred_root)])
    train_env = {}
    if method_id == "SPECIALIST_fixed_pattern":
        train_env["WEEK6_ALLOW_SPECIALIST_TRAINING"] = "true"
    return {
        "method_id": method_id,
        "method_run_id": method_run_id,
        "configured_output_dir": str(configured_output_dir),
        "output_dir": str(output_dir),
        "train_script": train_script,
        "eval_script": eval_script,
        "train_cmd": train_cmd,
        "eval_cmd": eval_cmd,
        "train_env": train_env,
        "smoke_test": bool(smoke_test),
        "save_predictions": bool(save_predictions),
    }


def run_config(config_path: Path, *, smoke_test: bool, save_predictions: bool, dry_run: bool) -> dict[str, Any]:
    plan = command_plan(config_path, smoke_test=smoke_test, save_predictions=save_predictions)
    if dry_run:
        return {**plan, "status": "dry_run"}
    output_dir = Path(plan["output_dir"])
    if output_dir.exists() and not smoke_test:
        raise RuntimeError(f"Refusing to overwrite existing output_dir={output_dir}")
    train_env = {**os.environ, **plan.get("train_env", {})}
    subprocess.run(plan["train_cmd"], check=True, env=train_env)
    if plan["eval_cmd"]:
        subprocess.run(plan["eval_cmd"], check=True)
    return {**plan, "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_config(
                args.config,
                smoke_test=args.smoke_test,
                save_predictions=args.save_predictions,
                dry_run=args.dry_run,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
