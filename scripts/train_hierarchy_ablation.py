#!/usr/bin/env python3
"""Train the Week 2 A4b hierarchy ablation."""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.losses.hierarchy import load_parent_indices
from hlm_ecg.training.train_subclass_auxiliary import train_subclass_auxiliary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PTB-XL hierarchy ablation.")
    parser.add_argument("--config", type=Path, default=Path("configs/hierarchy_ablation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = str(args.output_dir)
    model_cfg = dict(config.get("model", {}))
    sub_cfg = dict(config.get("subclass_auxiliary", {}))
    hier_cfg = dict(config.get("hierarchy_loss", {}))
    if not bool(model_cfg.get("use_availability_embedding", False)):
        raise SystemExit("A4b requires model.use_availability_embedding=true")
    if not bool(model_cfg.get("enable_subclass_auxiliary", False)):
        raise SystemExit("A4b requires model.enable_subclass_auxiliary=true")
    if not bool(sub_cfg.get("enabled", False)):
        raise SystemExit("A4b requires subclass_auxiliary.enabled=true")
    if not bool(model_cfg.get("use_hierarchy_loss", False)) or not bool(hier_cfg.get("enabled", False)):
        raise SystemExit("A4b requires hierarchy loss enabled in model and hierarchy_loss config")
    if float(hier_cfg.get("lambda_hier", model_cfg.get("lambda_hier", 0.0))) <= 0.0:
        raise SystemExit("A4b requires lambda_hier > 0")
    for key in ("use_confidence_weighted_consistency", "confidence_weighted_consistency", "calibration"):
        if bool(model_cfg.get(key, False)):
            raise SystemExit(f"A4b must not enable {key}")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs/week2_hierarchy_ablation/hierarchy_seed42"))
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
    final_hier = dict(final_config.get("hierarchy_loss", {}))
    final_hier["parent_indices"] = list(
        load_parent_indices(
            vocab_path=final_paths["subclass_vocab"],
            mapping_path=final_paths["subclass_parent_mapping"],
        )
    )
    final_config["hierarchy_loss"] = final_hier
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
