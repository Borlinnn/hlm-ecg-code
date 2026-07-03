from pathlib import Path

import pandas as pd
import pytest

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.data.subclass_labels import (
    add_threshold_flags,
    assert_no_records500_for_subclass_audit,
    build_subclass_label_matrix,
    build_subclass_parent_mapping,
    diagnostic_subclass_code_table,
    subclass_count_frame,
)


def _scp():
    return pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 1.0, 0.0],
            "diagnostic_class": ["MI", "CD", "HYP", "MI"],
            "diagnostic_subclass": ["IMI", "CLBBB", "LVH", "SHOULD_NOT_USE"],
            "description": ["inferior MI", "complete LBBB", "LVH", "non diagnostic"],
        },
        index=["IMI", "CLBBB", "LVH", "XNON"],
    )


def _metadata():
    return pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4],
            "strat_fold": [1, 2, 9, 10],
            "scp_codes": [
                {"IMI": 100.0},
                {"CLBBB": 100.0, "XNON": 100.0},
                {"LVH": 80.0},
                {"IMI": 50.0, "LVH": 50.0},
            ],
        }
    )


def test_subclass_labels_do_not_hardcode_count_and_use_diagnostic_only():
    table = diagnostic_subclass_code_table(_scp())
    assert set(table["diagnostic_subclass"]) == {"IMI", "CLBBB", "LVH"}
    assert "SHOULD_NOT_USE" not in set(table["diagnostic_subclass"])

    mapping = build_subclass_parent_mapping(table)
    labels = build_subclass_label_matrix(_metadata(), mapping, table)
    assert list(labels.columns) == ["CLBBB", "LVH", "IMI"]
    assert int(labels.sum().sum()) == 5


def test_min_train_pos_filtering_works():
    table = diagnostic_subclass_code_table(_scp())
    mapping = build_subclass_parent_mapping(table)
    metadata = _metadata()
    labels = build_subclass_label_matrix(metadata, mapping, table)
    counts = add_threshold_flags(subclass_count_frame(metadata, labels, mapping), thresholds=(1, 2))
    kept_1 = set(counts.loc[counts["kept_min_train_pos_1"], "diagnostic_subclass"])
    kept_2 = set(counts.loc[counts["kept_min_train_pos_2"], "diagnostic_subclass"])
    assert kept_1 == {"IMI", "CLBBB"}
    assert kept_2 == set()


def test_subclass_audit_rejects_records500(tmp_path):
    root = tmp_path / "ptb-xl"
    (root / "records500").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        assert_no_records500_for_subclass_audit(root)


def test_superclass_label_order_is_unchanged():
    assert LABEL_ORDER == ("NORM", "MI", "STTC", "CD", "HYP")
