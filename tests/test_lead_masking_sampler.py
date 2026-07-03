import numpy as np

from hlm_ecg.data.lead_masking import (
    StructuredLeadMasking,
    mask_for_missing_count,
    structured_mask_for_pattern,
    structured_pattern_report,
)
from hlm_ecg.data.waveforms import CANONICAL_LEADS


def test_structured_sampler_outputs_mask_shape_and_available_lead():
    sampler = StructuredLeadMasking(seed=123)
    for _ in range(50):
        mask = sampler.sample()
        assert mask.shape == (12,)
        assert float(mask.sum()) >= 1.0


def test_random_missing_counts_are_correct():
    for missing_count in (0, 1, 3, 6):
        mask = mask_for_missing_count(missing_count, rng=np.random.default_rng(7))
        assert mask.shape == (12,)
        assert int((mask == 0).sum()) == missing_count
        assert float(mask.sum()) >= 1.0
    assert mask_for_missing_count(0).tolist() == [1.0] * 12


def test_limb_only_pattern_available_leads_are_from_names():
    report = structured_pattern_report("limb_only__precordial_missing", CANONICAL_LEADS)
    assert report["available_leads"] == ["I", "II", "III", "aVR", "aVL", "aVF"]
    assert report["available_indices"] == [CANONICAL_LEADS.index(lead) for lead in report["available_leads"]]
    assert report["availability_mask"] == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_precordial_only_pattern_available_leads_are_from_names():
    report = structured_pattern_report("precordial_only__limb_missing", CANONICAL_LEADS)
    assert report["available_leads"] == ["V1", "V2", "V3", "V4", "V5", "V6"]
    assert report["available_indices"] == [CANONICAL_LEADS.index(lead) for lead in report["available_leads"]]
    assert report["availability_mask"] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_v1_v3_missing_pattern_available_leads_are_from_names():
    mask = structured_mask_for_pattern("V1_V3_missing", CANONICAL_LEADS)
    available = [lead for lead, value in zip(CANONICAL_LEADS, mask) if value == 1]
    assert available == ["I", "II", "III", "aVR", "aVL", "aVF", "V4", "V5", "V6"]
    assert mask.tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]


def test_v4_v6_missing_pattern_available_leads_are_from_names():
    mask = structured_mask_for_pattern("V4_V6_missing", CANONICAL_LEADS)
    available = [lead for lead, value in zip(CANONICAL_LEADS, mask) if value == 1]
    assert available == ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3"]
    assert mask.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


def test_structured_sampler_fixed_seed_reproducible():
    a = StructuredLeadMasking(seed=42)
    b = StructuredLeadMasking(seed=42)
    seq_a = [a.sample().tolist() for _ in range(20)]
    seq_b = [b.sample().tolist() for _ in range(20)]
    assert seq_a == seq_b
