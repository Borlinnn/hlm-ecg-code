import os
from pathlib import Path

import numpy as np
import torch

from hlm_ecg.evaluation.week6_defense import (
    CHALLENGE_RECON_PATTERNS,
    K_BOUNDARY_PATTERNS,
    NONINFERIORITY_MARGIN_AUPRC,
    limb_reconstruction_applicability,
    reconstruct_limb_leads_from_i_ii,
    noninferiority_decision,
    selected_patterns,
    specialist_training_allowed,
    transform_availability_mask,
)


def test_week6_patterns_are_available_and_nonempty():
    names = [*CHALLENGE_RECON_PATTERNS, *K_BOUNDARY_PATTERNS]
    patterns = selected_patterns(names)
    assert set(patterns) == set(names)
    for pattern in patterns.values():
        mask = pattern.mask_for_index(0)
        assert mask.shape == (12,)
        assert mask.sum() >= 1


def test_limb_reconstruction_formula_raw_units():
    raw_i = np.asarray([1.0, 2.0, -1.0], dtype=np.float32)
    raw_ii = np.asarray([3.0, -2.0, 4.0], dtype=np.float32)
    out = reconstruct_limb_leads_from_i_ii(raw_i, raw_ii)
    np.testing.assert_allclose(out["III"], raw_ii - raw_i)
    np.testing.assert_allclose(out["aVR"], -0.5 * (raw_i + raw_ii))
    np.testing.assert_allclose(out["aVL"], raw_i - 0.5 * raw_ii)
    np.testing.assert_allclose(out["aVF"], raw_ii - 0.5 * raw_i)


def test_limb_reconstruction_applicability_is_name_derived():
    names = [
        "limb-only / precordial-missing",
        "precordial-only / limb-missing",
        "V1-V3 missing",
        "V4-V6 missing",
        "challenge_6_limb",
        "challenge_4_I_II_III_V2",
        "challenge_3_I_II_V2",
        "challenge_2_I_II",
    ]
    patterns = selected_patterns(names)
    rows = {name: limb_reconstruction_applicability(name, patterns[name]) for name in names}

    assert rows["limb-only / precordial-missing"]["n_reconstructed_leads"] == 0
    assert rows["limb-only / precordial-missing"]["no_op_reason"] == "missing_chest_precordial_leads_not_synthesized"
    assert rows["challenge_6_limb"]["n_reconstructed_leads"] == 0
    assert rows["challenge_6_limb"]["no_op_reason"] == "missing_chest_precordial_leads_not_synthesized"
    assert rows["V1-V3 missing"]["n_reconstructed_leads"] == 0
    assert rows["V1-V3 missing"]["no_op_reason"] == "missing_chest_precordial_leads_not_synthesized"
    assert rows["V4-V6 missing"]["n_reconstructed_leads"] == 0
    assert rows["V4-V6 missing"]["no_op_reason"] == "missing_chest_precordial_leads_not_synthesized"
    assert rows["precordial-only / limb-missing"]["n_reconstructed_leads"] == 0
    assert rows["precordial-only / limb-missing"]["no_op_reason"] == "i_or_ii_unavailable"

    assert rows["challenge_2_I_II"]["reconstructable_missing_limb_leads"] == "III,aVR,aVL,aVF"
    assert rows["challenge_3_I_II_V2"]["reconstructable_missing_limb_leads"] == "III,aVR,aVL,aVF"
    assert rows["challenge_4_I_II_III_V2"]["reconstructable_missing_limb_leads"] == "aVR,aVL,aVF"


def test_availability_mask_variants_only_change_metadata_tensor():
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    assert torch.equal(transform_availability_mask(mask, "correct"), mask)
    assert torch.equal(transform_availability_mask(mask, "all_ones"), torch.ones_like(mask))
    shuffled = transform_availability_mask(mask, "shuffled")
    assert shuffled.shape == mask.shape
    assert torch.equal(shuffled[0], mask[1])


def test_noninferiority_margin_rule():
    assert noninferiority_decision(-0.009, margin=NONINFERIORITY_MARGIN_AUPRC) == "noninferior_with_margin"
    assert noninferiority_decision(-0.011, margin=NONINFERIORITY_MARGIN_AUPRC) == "not_established"


def test_specialist_training_is_gated_by_default(monkeypatch):
    monkeypatch.delenv("WEEK6_ALLOW_SPECIALIST_TRAINING", raising=False)
    assert specialist_training_allowed() is False
    monkeypatch.setenv("WEEK6_ALLOW_SPECIALIST_TRAINING", "true")
    assert specialist_training_allowed() is True


def test_week6_scripts_exist():
    scripts = [
        "scripts/run_week6_reconstruction_imputation_eval.py",
        "scripts/run_week6_full_lead_preservation.py",
        "scripts/run_week6_availability_mask_ablation.py",
        "scripts/run_week6_boundary_analysis.py",
        "scripts/setup_week6_specialist_baselines.py",
        "scripts/train_week6_fixed_pattern_specialist.py",
        "scripts/build_week6_reviewer_defense_report.py",
    ]
    for script in scripts:
        assert Path(script).exists(), script
