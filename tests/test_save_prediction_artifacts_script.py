import importlib.util
import sys
from pathlib import Path

import numpy as np

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.prediction_artifacts import save_predictions_csv
from hlm_ecg.evaluation.prediction_artifacts import prediction_rows


def fake_collected(n=3):
    logits = np.linspace(-1.0, 1.0, n * len(LABEL_ORDER)).reshape(n, len(LABEL_ORDER))
    targets = np.zeros((n, len(LABEL_ORDER)), dtype=np.int64)
    targets[:, 0] = 1
    return {
        "logits": logits,
        "targets": targets,
        "ecg_ids": np.arange(1, n + 1),
        "patient_ids": np.arange(101, 101 + n),
        "strat_folds": np.full(n, 10),
        "availability_masks": np.ones((n, 12), dtype=np.float32),
        "splits": ["test"] * n,
    }


def load_save_script():
    path = Path("scripts/save_prediction_artifacts.py")
    spec = importlib.util.spec_from_file_location("save_prediction_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_audit_script():
    path = Path("scripts/audit_prediction_artifacts.py")
    spec = importlib.util.spec_from_file_location("audit_prediction_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_records_method_pattern_split(tmp_path):
    module = load_save_script()
    entries = [
        {
            "method_id": "A4a_subclass_auxiliary",
            "split": "test",
            "fill_mode": "mean_fill",
            "pattern": "random-6",
            "csv_path": str(tmp_path / "random_6.csv"),
            "n_rows": 2,
            "n_labels": 5,
            "has_logits": True,
            "has_probabilities": True,
            "has_thresholds": True,
            "threshold_source_split": "val",
            "file_size": 1,
            "sha256": "abc",
            "created_at": "now",
        }
    ]
    module.write_manifest(tmp_path, entries)
    assert (tmp_path / "prediction_manifest.csv").exists()
    assert "A4a_subclass_auxiliary" in (tmp_path / "prediction_manifest.md").read_text(encoding="utf-8")


def test_audit_finds_missing_and_invalid_counts(tmp_path):
    audit_module = load_audit_script()
    rows = prediction_rows(
        method_id="A4a_subclass_auxiliary",
        pattern="full",
        fill_mode="mean_fill",
        split="test",
        random_seed=20240604,
        threshold_source_split="val",
        thresholds=[0.5] * 5,
        collected=fake_collected(n=2),
    )
    path = tmp_path / "A4a_subclass_auxiliary" / "mean_fill" / "test" / "full.csv"
    save_predictions_csv(path, rows)
    audit = audit_module.audit_predictions(
        tmp_path,
        methods=["A4a_subclass_auxiliary"],
        fill_modes=["mean_fill"],
        splits=["test"],
        patterns=["full"],
    )
    assert audit["missing_count"] == 0
    assert audit["invalid_count"] == 1
    assert audit["invalid"][0]["issues"][0]["row_count"] == 2


def test_save_predictions_script_does_not_call_training():
    text = Path("scripts/save_prediction_artifacts.py").read_text(encoding="utf-8")
    assert "train_full_baseline" not in text
    assert "train_random" not in text
    assert "train_subclass" not in text
    assert "scripts/train_" not in text


def test_records500_is_not_present():
    assert not Path("data/ptb-xl/records500").exists()
