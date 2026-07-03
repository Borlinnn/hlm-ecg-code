"""Missing-lead evaluation pattern utilities."""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from hlm_ecg.data.lead_patterns import LIMB_LEADS, PRECORDIAL_LEADS
from hlm_ecg.data.waveforms import CANONICAL_LEADS, canonicalize_leads


@dataclass(frozen=True)
class MissingPattern:
    name: str
    random_missing_count: int | None
    available_leads: tuple[str, ...] | None
    seed: int = 20240604

    def mask_for_index(self, sample_index: int, lead_names: Sequence[object] = CANONICAL_LEADS) -> np.ndarray:
        canonical = canonicalize_leads(lead_names)
        if canonical != CANONICAL_LEADS:
            raise RuntimeError(f"Unexpected lead order: {canonical}")
        if self.name == "full":
            return np.ones(12, dtype=np.float32)
        if self.random_missing_count is not None:
            rng = np.random.default_rng(self.seed + int(sample_index))
            missing_indices = set(rng.choice(np.arange(12), size=self.random_missing_count, replace=False).tolist())
            mask = np.array([0.0 if idx in missing_indices else 1.0 for idx in range(12)], dtype=np.float32)
        else:
            available = set(self.available_leads or ())
            mask = np.array([1.0 if lead in available else 0.0 for lead in canonical], dtype=np.float32)
        if mask.shape != (12,) or float(mask.sum()) < 1.0:
            raise RuntimeError(f"Invalid mask for pattern {self.name}: {mask}")
        return mask

    def example_mask(self, lead_names: Sequence[object] = CANONICAL_LEADS) -> np.ndarray:
        return self.mask_for_index(0, lead_names)

    def to_dict(self, lead_names: Sequence[object] = CANONICAL_LEADS) -> Dict[str, object]:
        canonical = canonicalize_leads(lead_names)
        mask = self.example_mask(canonical)
        available_leads = [lead for lead, value in zip(canonical, mask) if value == 1]
        available_indices = [idx for idx, value in enumerate(mask.tolist()) if value == 1]
        out: Dict[str, object] = {
            "name": self.name,
            "available_leads": available_leads,
            "available_indices": available_indices,
            "availability_mask": [int(x) for x in mask.tolist()],
            "random": self.random_missing_count is not None,
        }
        if self.random_missing_count is not None:
            out["missing_count"] = int(self.random_missing_count)
            out["seed"] = int(self.seed)
            out["rule"] = "per-sample deterministic random missing leads from canonical lead names"
        return out


def structured_pattern(name: str, available_leads: Iterable[str]) -> MissingPattern:
    available = tuple(available_leads)
    unknown = set(available).difference(CANONICAL_LEADS)
    if unknown:
        raise RuntimeError(f"Unknown leads in {name}: {sorted(unknown)}")
    return MissingPattern(name=name, random_missing_count=None, available_leads=available)


def required_patterns(seed: int = 20240604) -> Dict[str, MissingPattern]:
    all_leads = tuple(CANONICAL_LEADS)
    patterns = {
        "full": MissingPattern("full", random_missing_count=None, available_leads=all_leads, seed=seed),
        "random-1": MissingPattern("random-1", random_missing_count=1, available_leads=None, seed=seed),
        "random-3": MissingPattern("random-3", random_missing_count=3, available_leads=None, seed=seed),
        "random-6": MissingPattern("random-6", random_missing_count=6, available_leads=None, seed=seed),
        "limb-only / precordial-missing": structured_pattern(
            "limb-only / precordial-missing", LIMB_LEADS
        ),
        "precordial-only / limb-missing": structured_pattern(
            "precordial-only / limb-missing", PRECORDIAL_LEADS
        ),
        "V1-V3 missing": structured_pattern(
            "V1-V3 missing", [lead for lead in all_leads if lead not in {"V1", "V2", "V3"}]
        ),
        "V4-V6 missing": structured_pattern(
            "V4-V6 missing", [lead for lead in all_leads if lead not in {"V4", "V5", "V6"}]
        ),
    }
    return patterns


def pattern_report(seed: int = 20240604) -> Dict[str, Mapping[str, object]]:
    return {name: pattern.to_dict() for name, pattern in required_patterns(seed).items()}
