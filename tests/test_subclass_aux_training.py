from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.subclass_labels import build_kept_subclass_artifacts
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.evaluation.evaluate_patterns import build_pattern_dataset
from hlm_ecg.evaluation.missing_patterns import required_patterns
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.training.train_baseline import forward_model
from hlm_ecg.training.train_subclass_auxiliary import predict_super_logits, run_subclass_epoch
from hlm_ecg.losses.subclass_auxiliary import SubclassAuxiliaryLoss

ROOT = Path("data/ptb-xl")
INDEX = Path("outputs/day1_audit/ptbxl_day1_index.csv")
NORM = Path("outputs/day1_audit/train_norm_stats.npz")


def test_training_epoch_runs_and_evaluation_uses_superclass_logits(tmp_path):
    paths = _write_artifacts(tmp_path)
    ds = PTBXLDataset(
        root=ROOT,
        index_csv=INDEX,
        norm_stats_path=NORM,
        split="train",
        subclass_index_csv=paths["subclass_index"],
        subclass_vocab_path=paths["subclass_vocab"],
        limit=2,
    )
    loader = DataLoader(ds, batch_size=2)
    model = ResNet1DAvailability(
        base_channels=4,
        layers=(1, 1, 1, 1),
        use_subclass_auxiliary=True,
        num_subclasses=paths["num_subclasses"],
    )
    losses = run_subclass_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        criterion=SubclassAuxiliaryLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    )
    assert losses["loss"] > 0
    logits, targets = predict_super_logits(model, loader, device=torch.device("cpu"))
    assert tuple(logits.shape) == (2, 5)
    assert tuple(targets.shape) == (2, 5)
    batch = next(iter(loader))
    super_logits = forward_model(model, batch, device=torch.device("cpu"))
    assert tuple(super_logits.shape) == (2, 5)


def test_threshold_source_contract_is_validation_fold():
    from hlm_ecg.evaluation.metrics import tune_thresholds_on_validation

    info = tune_thresholds_on_validation(torch.zeros(4, 5).numpy(), torch.zeros(4, 5).numpy())
    assert info["source_split"] == "val"


def test_evaluation_pattern_dataset_does_not_require_y_sub():
    config = {
        "paths": {
            "data_root": str(ROOT),
            "day1_index": str(INDEX),
            "norm_stats": str(NORM),
        },
        "smoke": {"test_limit": 2},
    }
    ds = build_pattern_dataset(config, required_patterns()["random-1"], fill_mode="mean_fill", smoke_test=True)
    sample = ds[0]
    assert "y_sub" not in sample
    assert tuple(sample["y"].shape) == (5,)


def _write_artifacts(tmp_path):
    artifacts = build_kept_subclass_artifacts(root=ROOT, day1_index=INDEX, min_train_pos=50)
    index_path = tmp_path / "subclass_index.csv"
    vocab_path = tmp_path / "subclass_vocab.json"
    artifacts["index"].to_csv(index_path, index=False)
    vocab_path.write_text(__import__("json").dumps(artifacts["vocab"]), encoding="utf-8")
    return {
        "subclass_index": index_path,
        "subclass_vocab": vocab_path,
        "num_subclasses": int(artifacts["vocab"]["num_subclasses"]),
    }
