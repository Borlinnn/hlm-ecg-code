import csv
import numpy as np

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.metrics import sigmoid
from hlm_ecg.evaluation.prediction_artifacts import (
    PREDICTION_REQUIRED_COLUMNS,
    prediction_rows,
    safe_pattern_name,
    save_predictions_csv,
    validate_prediction_csv_schema,
    validate_split_row_count,
)


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


def test_safe_pattern_name_handles_required_patterns():
    assert safe_pattern_name("random-6") == "random_6"
    assert safe_pattern_name("limb-only / precordial-missing") == "limb_only_precordial_missing"
    assert safe_pattern_name("V1-V3 missing") == "V1_V3_missing"


def test_prediction_csv_schema_contains_required_columns(tmp_path):
    rows = prediction_rows(
        method_id="A4a_subclass_auxiliary",
        pattern="random-6",
        fill_mode="mean_fill",
        split="test",
        random_seed=20240604,
        threshold_source_split="val",
        thresholds=[0.5] * len(LABEL_ORDER),
        collected=fake_collected(),
    )
    path = tmp_path / "pred.csv"
    save_predictions_csv(path, rows)
    assert validate_prediction_csv_schema(path) == []
    with path.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    for column in PREDICTION_REQUIRED_COLUMNS:
        assert column in header


def test_threshold_source_split_and_probabilities_are_correct(tmp_path):
    collected = fake_collected(n=1)
    rows = prediction_rows(
        method_id="A1_random_dropout",
        pattern="full",
        fill_mode="mean_fill",
        split="val",
        random_seed=20240604,
        threshold_source_split="val",
        thresholds=[0.5] * len(LABEL_ORDER),
        collected=collected,
    )
    row = rows[0]
    assert row["threshold_source_split"] == "val"
    expected = sigmoid(collected["logits"])[0, 0]
    assert abs(row["prob_NORM"] - expected) < 1e-12
    assert row["pred_NORM"] in {0, 1}


def test_split_row_count_checker_detects_missing_rows(tmp_path):
    rows = prediction_rows(
        method_id="A0_full_no_masking",
        pattern="full",
        fill_mode="mean_fill",
        split="test",
        random_seed=20240604,
        threshold_source_split="val",
        thresholds=[0.5] * len(LABEL_ORDER),
        collected=fake_collected(n=2),
    )
    path = tmp_path / "pred.csv"
    save_predictions_csv(path, rows)
    assert validate_split_row_count(path, 2)
    assert not validate_split_row_count(path, 2198)
