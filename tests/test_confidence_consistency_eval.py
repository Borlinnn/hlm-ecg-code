import json
from pathlib import Path

import numpy as np
import yaml

from hlm_ecg.evaluation.evaluate_patterns import evaluate_missing_patterns
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.training.train_baseline import save_checkpoint

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def test_consistency_evaluation_uses_superclass_metrics_only(tmp_path):
    config = {
        "seed": 42,
        "device": "cpu",
        "paths": {
            "data_root": str(ROOT),
            "day1_index": str(INDEX),
            "norm_stats": str(NORM),
            "output_dir": str(tmp_path),
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

    result = evaluate_missing_patterns(
        checkpoint_path=ckpt,
        config=config,
        fill_mode="mean_fill",
        smoke_test=True,
    )
    assert Path(result["csv"]).exists()
    assert Path(result["json"]).exists()
    full = next(row for row in result["rows"] if row["pattern"] == "full")
    assert {"macro_auroc", "macro_auprc", "macro_f1", "bce_nll"}.issubset(full)
    data = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    assert "per_class_auprc" in data["patterns"]["full"]["metrics"]
    assert "cw_consistency_loss" not in data["patterns"]["full"]["metrics"]


def test_threshold_contract_is_validation_fold_for_a5():
    from hlm_ecg.evaluation.metrics import tune_thresholds_on_validation

    info = tune_thresholds_on_validation(np.zeros((2, 5)), np.zeros((2, 5)))
    assert info["source_split"] == "val"
