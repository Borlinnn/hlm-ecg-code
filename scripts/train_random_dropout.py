#!/usr/bin/env python3
"""Train the Week 1 random lead dropout baseline."""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.training.train_baseline import train_full_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PTB-XL random lead dropout baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/random_dropout.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = str(args.output_dir)
    if not bool(config.get("train_augmentation", {}).get("enabled", False)):
        raise SystemExit("configs/random_dropout.yaml must enable train_augmentation")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs/week1_random_dropout/random_dropout_seed42"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = train_full_baseline(config, max_epochs=args.max_epochs, smoke_test=args.smoke_test)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
