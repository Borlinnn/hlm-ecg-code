import numpy as np

from hlm_ecg.data.waveforms import CANONICAL_LEADS
from hlm_ecg.evaluation.supplemental_patterns import (
    CHALLENGE_PATTERN_ORDER,
    K_VISIBLE_PATTERN_ORDER,
    challenge_reduced_lead_patterns,
    k_visible_random_patterns,
    pattern_metadata,
)


def test_challenge_patterns_use_expected_leads():
    patterns = challenge_reduced_lead_patterns(seed=123)
    assert list(patterns) == list(CHALLENGE_PATTERN_ORDER)
    expected = {
        "challenge_12_all": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "challenge_6_limb": ["I", "II", "III", "aVR", "aVL", "aVF"],
        "challenge_4_I_II_III_V2": ["I", "II", "III", "V2"],
        "challenge_3_I_II_V2": ["I", "II", "V2"],
        "challenge_2_I_II": ["I", "II"],
    }
    for name, leads in expected.items():
        mask = patterns[name].mask_for_index(17)
        expected_mask = [1.0 if lead in leads else 0.0 for lead in CANONICAL_LEADS]
        assert mask.tolist() == expected_mask
        assert int(mask.sum()) == len(leads)


def test_k_visible_patterns_have_exactly_k_available_leads():
    patterns = k_visible_random_patterns(seed=20240606)
    assert list(patterns) == list(K_VISIBLE_PATTERN_ORDER)
    expected_counts = {
        "k12_visible": 12,
        "k8_visible_random": 8,
        "k6_visible_random": 6,
        "k4_visible_random": 4,
        "k3_visible_random": 3,
        "k2_visible_random": 2,
        "k1_visible_random": 1,
    }
    for name, k in expected_counts.items():
        mask = patterns[name].mask_for_index(5)
        assert mask.shape == (12,)
        assert int(mask.sum()) == k
        assert set(mask.tolist()).issubset({0.0, 1.0})


def test_k_visible_masks_are_deterministic_and_paired_by_index():
    p1 = k_visible_random_patterns(seed=7)["k3_visible_random"]
    p2 = k_visible_random_patterns(seed=7)["k3_visible_random"]
    assert np.array_equal(p1.mask_for_index(99), p2.mask_for_index(99))
    assert not np.array_equal(p1.mask_for_index(98), p1.mask_for_index(99))


def test_k_visible_never_returns_all_zero_mask():
    patterns = k_visible_random_patterns(seed=11)
    for pattern in patterns.values():
        for idx in range(10):
            assert float(pattern.mask_for_index(idx).sum()) >= 1.0


def test_pattern_metadata_serializes_masks_from_names():
    metadata = pattern_metadata(challenge_reduced_lead_patterns())
    assert metadata["challenge_2_I_II"]["available_indices"] == [0, 1]
    assert metadata["challenge_2_I_II"]["available_leads"] == ["I", "II"]
