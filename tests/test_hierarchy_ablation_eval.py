import json
from pathlib import Path

import yaml

from hlm_ecg.evaluation.hierarchy_patterns import evaluate_hierarchy_missing_patterns
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.training.train_baseline import save_checkpoint

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def test_evaluation_writes_hierarchy_diagnostics(tmp_path):
    vocab_path = tmp_path / "subclass_vocab.json"
    mapping_path = tmp_path / "subclass_parent_mapping.json"
    vocab_path.write_text(
        json.dumps(
            {
                "subclasses": ["sub_mi", "sub_cd"],
                "subclass_columns": ["y_sub_sub_mi", "y_sub_sub_cd"],
                "num_subclasses": 2,
                "label_order": ["NORM", "MI", "STTC", "CD", "HYP"],
            }
        ),
        encoding="utf-8",
    )
    mapping_path.write_text(
        json.dumps(
            {
                "mapping_unique": True,
                "mapping": [
                    {"diagnostic_subclass": "sub_mi", "parent_superclass": "MI", "parent_valid": True},
                    {"diagnostic_subclass": "sub_cd", "parent_superclass": "CD", "parent_valid": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "seed": 42,
        "device": "cpu",
        "paths": {
            "data_root": str(ROOT),
            "day1_index": str(INDEX),
            "norm_stats": str(NORM),
            "output_dir": str(tmp_path),
            "subclass_vocab": str(vocab_path),
            "subclass_parent_mapping": str(mapping_path),
        },
        "model": {
            "in_channels": 12,
            "num_classes": 5,
            "base_channels": 4,
            "layers": [1, 1, 1, 1],
            "kernel_size": 7,
            "use_availability_embedding": True,
            "availability_embedding_dim": 4,
            "mask_mlp_hidden_dim": 4,
            "enable_subclass_auxiliary": True,
            "num_subclasses": 2,
        },
        "hierarchy_loss": {"enabled": True, "lambda_hier": 0.1, "hierarchy_violation_eps": 0.0},
        "training": {"batch_size": 2, "num_workers": 0},
        "evaluation": {"batch_size": 2, "num_workers": 0, "pattern_seed": 123},
        "smoke": {"test_limit": 1},
    }
    (tmp_path / "thresholds_val.json").write_text(
        json.dumps({"thresholds": {"NORM": 0.5, "MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}}),
        encoding="utf-8",
    )
    model = ResNet1DAvailability(
        base_channels=4,
        layers=(1, 1, 1, 1),
        availability_embedding_dim=4,
        mask_mlp_hidden_dim=4,
        use_subclass_auxiliary=True,
        num_subclasses=2,
    )
    ckpt = tmp_path / "best_model.pt"
    save_checkpoint(ckpt, model=model, config=config, epoch=1, val_macro_auprc=0.1)
    (tmp_path / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = evaluate_hierarchy_missing_patterns(
        checkpoint_path=ckpt,
        config=config,
        fill_mode="mean_fill",
        smoke_test=True,
    )
    assert Path(result["csv"]).exists()
    assert Path(result["json"]).exists()
    full = next(row for row in result["rows"] if row["pattern"] == "full")
    assert "hierarchy_loss" in full
    assert "violation_rate" in full
    assert "mean_violation_margin" in full
    assert "max_violation_margin" in full
