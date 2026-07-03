import pandas as pd
import pytest

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.data.subclass_labels import build_subclass_parent_mapping, diagnostic_subclass_code_table


def test_parent_superclass_must_be_fixed_label_order_member():
    scp = pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0],
            "diagnostic_class": ["MI", "NOT_A_PARENT"],
            "diagnostic_subclass": ["IMI", "BAD"],
        },
        index=["IMI", "BAD"],
    )
    table = diagnostic_subclass_code_table(scp)
    mapping = build_subclass_parent_mapping(table)
    assert set(mapping["parent_superclass"]).issubset(set(LABEL_ORDER))
    assert set(mapping["diagnostic_subclass"]) == {"IMI"}


def test_each_kept_subclass_has_unique_parent():
    scp = pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0],
            "diagnostic_class": ["MI", "CD"],
            "diagnostic_subclass": ["SHARED", "SHARED"],
        },
        index=["A", "B"],
    )
    table = diagnostic_subclass_code_table(scp)
    with pytest.raises(RuntimeError, match="conflicts"):
        build_subclass_parent_mapping(table)


def test_valid_mapping_is_unique_and_reliable():
    scp = pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 1.0],
            "diagnostic_class": ["MI", "CD", "HYP"],
            "diagnostic_subclass": ["IMI", "CLBBB", "LVH"],
        },
        index=["IMI", "CLBBB", "LVH"],
    )
    table = diagnostic_subclass_code_table(scp)
    mapping = build_subclass_parent_mapping(table)
    assert len(mapping) == 3
    assert mapping["diagnostic_subclass"].is_unique
    assert set(mapping["parent_superclass"]) == {"MI", "CD", "HYP"}
