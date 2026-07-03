import pytest

from hlm_ecg.data.lead_patterns import build_required_patterns
from hlm_ecg.data.waveforms import (
    CANONICAL_LEADS,
    assert_records100_filename_lr,
    canonicalize_leads,
)


def test_canonicalize_limb_leads():
    raw = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert canonicalize_leads(raw) == CANONICAL_LEADS


def test_required_structured_pattern_masks():
    patterns = build_required_patterns(CANONICAL_LEADS)
    full = patterns["full"]
    assert full["availability_mask"] == [1] * 12
    limb_only = patterns["limb-only / precordial-missing"]
    assert limb_only["available_leads"] == ["I", "II", "III", "aVR", "aVL", "aVF"]
    assert limb_only["available_indices"] == [0, 1, 2, 3, 4, 5]
    assert limb_only["availability_mask"] == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    v1_v3_missing = patterns["V1-V3 missing"]
    assert v1_v3_missing["availability_mask"] == [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]


def test_random_patterns_include_rule_and_example_mask():
    patterns = build_required_patterns(CANONICAL_LEADS)
    random_3 = patterns["random-3"]
    assert random_3["random"] is True
    assert random_3["missing_count"] == 3
    assert sum(random_3["availability_mask"]) == 9
    assert "indices are derived from lead names" in random_3["rule"]


def test_records500_and_hr_are_forbidden():
    assert_records100_filename_lr("records100/00000/00001_lr")
    with pytest.raises(RuntimeError):
        assert_records100_filename_lr("records500/00000/00001_hr")
    with pytest.raises(RuntimeError):
        assert_records100_filename_lr("records100/00000/00001_hr")
