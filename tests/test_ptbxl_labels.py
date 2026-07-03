import pandas as pd
import pytest

from hlm_ecg.data.ptbxl_labels import (
    EXPECTED_SUPERCLASS_COUNTS,
    LABEL_ORDER,
    aggregate_superclasses,
    assert_expected_counts,
    build_label_frame,
    diagnostic_class_map,
)


def test_superclass_label_order_fixed():
    assert LABEL_ORDER == ("NORM", "MI", "STTC", "CD", "HYP")


def test_diagnostic_only_aggregation_ignores_non_diagnostic():
    scp = pd.DataFrame(
        {
            "diagnostic": [1.0, 0.0, 1.0],
            "diagnostic_class": ["MI", "NORM", "STTC"],
        },
        index=["IMI", "FORMONLY", "NST_"],
    )
    mapping = diagnostic_class_map(scp)
    classes = aggregate_superclasses({"IMI": 0.0, "FORMONLY": 100.0, "NST_": 20.0}, mapping)
    assert classes == ("MI", "STTC")


def test_likelihood_positive_sensitivity_differs_from_official_style():
    scp = pd.DataFrame(
        {"diagnostic": [1.0], "diagnostic_class": ["NORM"]},
        index=["NORM"],
    )
    codes = [{"NORM": 0.0}]
    official = build_label_frame(codes, scp, positive_likelihood_only=False)
    positive = build_label_frame(codes, scp, positive_likelihood_only=True)
    assert int(official["NORM"].iloc[0]) == 1
    assert int(positive["NORM"].iloc[0]) == 0


def test_expected_count_assertion_is_strict():
    counts = dict(EXPECTED_SUPERCLASS_COUNTS)
    counts["NORM"] -= 1
    with pytest.raises(RuntimeError):
        assert_expected_counts(counts)
