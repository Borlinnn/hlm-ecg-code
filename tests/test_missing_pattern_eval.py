import numpy as np

from hlm_ecg.data.waveforms import CANONICAL_LEADS
from hlm_ecg.evaluation.missing_patterns import required_patterns


def test_missing_pattern_masks_have_shape_and_available_lead():
    patterns = required_patterns(seed=123)
    for pattern in patterns.values():
        mask = pattern.mask_for_index(0)
        assert mask.shape == (12,)
        assert float(mask.sum()) >= 1.0


def test_structured_indices_are_from_canonical_lead_names():
    pattern = required_patterns()["precordial-only / limb-missing"]
    report = pattern.to_dict(CANONICAL_LEADS)
    expected = [CANONICAL_LEADS.index(lead) for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]]
    assert report["available_indices"] == expected
    assert report["availability_mask"] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_random_patterns_are_reproducible_by_seed_and_index():
    p1 = required_patterns(seed=7)["random-3"]
    p2 = required_patterns(seed=7)["random-3"]
    assert np.array_equal(p1.mask_for_index(5), p2.mask_for_index(5))
    assert int(p1.mask_for_index(5).sum()) == 9


def test_full_pattern_no_masking():
    mask = required_patterns()["full"].mask_for_index(99)
    assert mask.tolist() == [1.0] * 12
