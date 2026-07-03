"""Supplementary missing-lead patterns for BIBM stabilization analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from hlm_ecg.data.waveforms import CANONICAL_LEADS, canonicalize_leads
from hlm_ecg.evaluation.missing_patterns import MissingPattern, structured_pattern

SUPPLEMENTAL_PATTERN_SEED = 20240606
CHALLENGE_PATTERN_ORDER = (
    "challenge_12_all",
    "challenge_6_limb",
    "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2",
    "challenge_2_I_II",
)
K_VISIBLE_PATTERN_ORDER = (
    "k12_visible",
    "k8_visible_random",
    "k6_visible_random",
    "k4_visible_random",
    "k3_visible_random",
    "k2_visible_random",
    "k1_visible_random",
)


@dataclass(frozen=True)
class KVisibleRandomPattern:
    """Per-sample deterministic random visible-lead pattern.

    The sampled subset depends only on `seed` and sample index, not on method ID.
    Mask convention is `1 = available`, `0 = missing`.
    """

    name: str
    visible_count: int
    seed: int = SUPPLEMENTAL_PATTERN_SEED

    def __post_init__(self) -> None:
        if not 1 <= int(self.visible_count) <= 12:
            raise ValueError(f"visible_count must be in [1, 12], got {self.visible_count}")

    def mask_for_index(self, sample_index: int, lead_names: Sequence[object] = CANONICAL_LEADS) -> np.ndarray:
        canonical = canonicalize_leads(lead_names)
        if canonical != CANONICAL_LEADS:
            raise RuntimeError(f"Unexpected lead order: {canonical}")
        if self.visible_count == 12:
            return np.ones(12, dtype=np.float32)
        rng = np.random.default_rng(int(self.seed) + int(sample_index))
        available_indices = set(rng.choice(np.arange(12), size=int(self.visible_count), replace=False).tolist())
        mask = np.asarray([1.0 if idx in available_indices else 0.0 for idx in range(12)], dtype=np.float32)
        if mask.shape != (12,) or int(mask.sum()) != int(self.visible_count):
            raise RuntimeError(f"Invalid k-visible mask for {self.name}: {mask}")
        return mask

    def example_mask(self, lead_names: Sequence[object] = CANONICAL_LEADS) -> np.ndarray:
        return self.mask_for_index(0, lead_names)

    def to_dict(self, lead_names: Sequence[object] = CANONICAL_LEADS) -> Dict[str, object]:
        canonical = canonicalize_leads(lead_names)
        mask = self.example_mask(canonical)
        available_leads = [lead for lead, value in zip(canonical, mask) if value == 1]
        available_indices = [idx for idx, value in enumerate(mask.tolist()) if value == 1]
        return {
            "name": self.name,
            "available_leads": available_leads,
            "available_indices": available_indices,
            "availability_mask": [int(x) for x in mask.tolist()],
            "random": True,
            "visible_count": int(self.visible_count),
            "missing_count": int(12 - self.visible_count),
            "seed": int(self.seed),
            "rule": "per-sample deterministic random visible leads from canonical lead names",
        }


def challenge_reduced_lead_patterns(seed: int = SUPPLEMENTAL_PATTERN_SEED) -> Dict[str, MissingPattern]:
    """Return PhysioNet/CinC-style fixed reduced-lead configurations."""

    all_leads = tuple(CANONICAL_LEADS)
    patterns = {
        "challenge_12_all": structured_pattern("challenge_12_all", all_leads),
        "challenge_6_limb": structured_pattern("challenge_6_limb", ["I", "II", "III", "aVR", "aVL", "aVF"]),
        "challenge_4_I_II_III_V2": structured_pattern("challenge_4_I_II_III_V2", ["I", "II", "III", "V2"]),
        "challenge_3_I_II_V2": structured_pattern("challenge_3_I_II_V2", ["I", "II", "V2"]),
        "challenge_2_I_II": structured_pattern("challenge_2_I_II", ["I", "II"]),
    }
    # Preserve a seed field in metadata for reproducibility, even fixed masks do not use it.
    return {
        name: MissingPattern(
            name=pattern.name,
            random_missing_count=pattern.random_missing_count,
            available_leads=pattern.available_leads,
            seed=seed,
        )
        for name, pattern in patterns.items()
    }


def k_visible_random_patterns(seed: int = SUPPLEMENTAL_PATTERN_SEED) -> Dict[str, KVisibleRandomPattern]:
    """Return deterministic per-sample random k-visible-lead patterns."""

    counts = {
        "k12_visible": 12,
        "k8_visible_random": 8,
        "k6_visible_random": 6,
        "k4_visible_random": 4,
        "k3_visible_random": 3,
        "k2_visible_random": 2,
        "k1_visible_random": 1,
    }
    return {name: KVisibleRandomPattern(name=name, visible_count=count, seed=seed) for name, count in counts.items()}


def pattern_metadata(patterns: Mapping[str, object], lead_names: Sequence[object] = CANONICAL_LEADS) -> Dict[str, object]:
    """Serialize a pattern registry for report/config artifacts."""

    return {
        name: pattern.to_dict(lead_names)  # type: ignore[attr-defined]
        for name, pattern in patterns.items()
    }
