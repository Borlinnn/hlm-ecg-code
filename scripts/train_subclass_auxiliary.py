#!/usr/bin/env python3
"""Train the Week 2 A4a subclass auxiliary ablation."""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.training.train_subclass_auxiliary import train_subclass_auxiliary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PTB-XL subclass auxiliary ablation.")
    parser.add_argument("--config", type=Path, default=Path("configs/subclass_auxiliary.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = str(args.output_dir)
    model_cfg = dict(config.get("model", {}))
    sub_cfg = dict(config.get("subclass_auxiliary", {}))
    allow_without_availability = bool(sub_cfg.get("allow_without_availability", False))
    if not bool(model_cfg.get("use_availability_embedding", False)) and not allow_without_availability:
        raise SystemExit("A4a requires model.use_availability_embedding=true")
    if not bool(model_cfg.get("enable_subclass_auxiliary", False)):
        raise SystemExit("A4a requires model.enable_subclass_auxiliary=true")
    if not bool(sub_cfg.get("enabled", False)):
        raise SystemExit("A4a requires subclass_auxiliary.enabled=true")
    for key in ("use_hierarchy_loss", "hierarchy_loss", "use_confidence_weighted_consistency", "confidence_weighted_consistency", "calibration"):
        if bool(model_cfg.get(key, False)):
            raise SystemExit(f"A4a must not enable {key}")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs/week2_subclass_auxiliary/subclass_aux_seed42"))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_subclass_auxiliary(config, max_epochs=args.max_epochs, smoke_test=args.smoke_test)
    final_config = dict(config)
    final_paths = dict(final_config.get("paths", {}))
    final_paths["output_dir"] = str(output_dir)
    final_paths["subclass_index"] = result["subclass_index"]
    final_paths["subclass_vocab"] = result["subclass_vocab"]
    final_paths["subclass_parent_mapping"] = result["subclass_parent_mapping"]
    final_config["paths"] = final_paths
    final_model = dict(final_config.get("model", {}))
    final_model["num_subclasses"] = int(result["num_subclasses"])
    final_config["model"] = final_model
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
