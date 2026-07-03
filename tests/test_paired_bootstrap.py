import numpy as np

from hlm_ecg.statistics.bootstrap import (
    HARD_OVERALL_PATTERNS,
    HARD_STRUCTURED_PATTERNS,
    generate_patient_bootstrap_samples,
    paired_delta_summary,
    patient_groups,
    sampled_indices_from_patients,
)


def test_patient_level_resampling_uses_patient_id_groups():
    patient_ids = np.asarray([10, 10, 20, 30, 30, 30], dtype=np.int64)
    groups = patient_groups(patient_ids)
    samples = generate_patient_bootstrap_samples(patient_ids, n_bootstrap=2, seed=42)

    assert set(groups) == {10, 20, 30}
    assert all(sample.shape == (3,) for sample in samples)
    indices = sampled_indices_from_patients(groups, np.asarray([30, 10], dtype=np.int64))
    assert indices.tolist() == [3, 4, 5, 0, 1]


def test_paired_bootstrap_can_reuse_the_same_patient_sample_for_two_methods():
    patient_ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    sample = generate_patient_bootstrap_samples(patient_ids, n_bootstrap=1, seed=7)[0]
    assert sample.tolist() == sample.tolist()

    groups_a = patient_groups(patient_ids)
    groups_b = patient_groups(patient_ids.copy())
    idx_a = sampled_indices_from_patients(groups_a, sample)
    idx_b = sampled_indices_from_patients(groups_b, sample)
    assert idx_a.tolist() == idx_b.tolist()


def test_delta_summary_uses_method_a_minus_method_b_and_valid_probability():
    deltas = np.asarray([0.1, 0.2, -0.1, 0.3], dtype=np.float64)
    summary = paired_delta_summary(deltas, observed_delta=0.125)
    assert summary["observed_delta"] == 0.125
    assert 0.0 <= summary["p_two_sided"] <= 1.0
    assert summary["probability_delta_gt_0"] == 0.75


def test_hard_pattern_sets_are_stable():
    assert HARD_STRUCTURED_PATTERNS == (
        "limb-only / precordial-missing",
        "precordial-only / limb-missing",
        "V1-V3 missing",
        "V4-V6 missing",
    )
    assert HARD_OVERALL_PATTERNS == (
        "random-6",
        "limb-only / precordial-missing",
        "precordial-only / limb-missing",
        "V1-V3 missing",
        "V4-V6 missing",
    )
