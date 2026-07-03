"""Class-wise temperature scaling fitted on validation predictions only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.calibration.calibration_metrics import binary_nll
from hlm_ecg.evaluation.metrics import sigmoid


@dataclass(frozen=True)
class TemperatureFitResult:
    class_name: str
    temperature: float
    val_nll_before: float
    val_nll_after: float
    converged: bool
    n_val_samples: int
    label_prevalence: float


def apply_temperatures(logits: np.ndarray, temperatures: Sequence[float]) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    temps = np.asarray(temperatures, dtype=np.float64)
    if temps.shape != (logits.shape[1],):
        raise ValueError(f"temperatures must have shape ({logits.shape[1]},), got {temps.shape}")
    if np.any(temps <= 0):
        raise ValueError("temperatures must be positive")
    return logits / temps.reshape(1, -1)


def _inverse_sigmoid(x: float) -> float:
    x = min(max(float(x), 1e-8), 1.0 - 1e-8)
    return float(np.log(x / (1.0 - x)))


def fit_binary_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    class_name: str,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    max_iter: int = 100,
) -> TemperatureFitResult:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    unique = np.unique(targets.astype(np.int64))
    if unique.size < 2:
        raise RuntimeError(f"Cannot fit temperature for {class_name}: validation labels contain one class")
    if min_temperature <= 0 or max_temperature <= min_temperature:
        raise ValueError("Invalid temperature bounds")

    val_nll_before = binary_nll(sigmoid(logits), targets)
    init_ratio = (1.0 - min_temperature) / (max_temperature - min_temperature)
    param = torch.tensor([_inverse_sigmoid(init_ratio)], dtype=torch.float64, requires_grad=True)
    logits_t = torch.tensor(logits, dtype=torch.float64)
    targets_t = torch.tensor(targets, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([param], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def temperature() -> torch.Tensor:
        return min_temperature + (max_temperature - min_temperature) * torch.sigmoid(param)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits_t / temperature(), targets_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        converged = True
    except RuntimeError:
        converged = False
    temp = float(temperature().detach().cpu().item())
    calibrated_probs = sigmoid(logits / temp)
    val_nll_after = binary_nll(calibrated_probs, targets)
    return TemperatureFitResult(
        class_name=class_name,
        temperature=temp,
        val_nll_before=val_nll_before,
        val_nll_after=val_nll_after,
        converged=bool(converged and np.isfinite(val_nll_after)),
        n_val_samples=int(targets.shape[0]),
        label_prevalence=float(targets.mean()),
    )


def fit_classwise_temperatures(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    label_order: Sequence[str] = LABEL_ORDER,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    max_iter: int = 100,
) -> tuple[np.ndarray, list[TemperatureFitResult]]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: {logits.shape}/{targets.shape}")
    if logits.shape[1] != len(label_order):
        raise ValueError(f"Expected {len(label_order)} labels, got {logits.shape[1]}")
    results: list[TemperatureFitResult] = []
    temps = []
    for idx, label in enumerate(label_order):
        result = fit_binary_temperature(
            logits[:, idx],
            targets[:, idx],
            class_name=label,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            max_iter=max_iter,
        )
        results.append(result)
        temps.append(result.temperature)
    return np.asarray(temps, dtype=np.float64), results

