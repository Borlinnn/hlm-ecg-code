import pandas as pd
import pytest

from hlm_ecg.data.splits import assign_official_splits, assert_patient_disjoint, split_summary


def test_official_split_assignment_and_summary():
    df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "strat_fold": [1, 9, 10],
            "NORM": [1, 0, 0],
            "MI": [0, 1, 0],
            "STTC": [0, 0, 1],
            "CD": [0, 0, 0],
            "HYP": [0, 0, 0],
        }
    )
    out = assign_official_splits(df)
    assert out["split"].tolist() == ["train", "val", "test"]
    leakage = assert_patient_disjoint(out)
    assert leakage == {"train_val": 0, "train_test": 0, "val_test": 0}
    summary = split_summary(out)
    assert summary["train"]["records"] == 1
    assert summary["val"]["patients"] == 1
    assert summary["test"]["positive_counts"]["STTC"] == 1


def test_patient_leakage_raises():
    df = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "strat_fold": [1, 9],
            "NORM": [1, 0],
            "MI": [0, 1],
            "STTC": [0, 0],
            "CD": [0, 0],
            "HYP": [0, 0],
        }
    )
    out = assign_official_splits(df)
    with pytest.raises(RuntimeError):
        assert_patient_disjoint(out)
