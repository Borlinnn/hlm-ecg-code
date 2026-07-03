import numpy as np

from hlm_ecg.calibration.calibration_metrics import binary_brier, binary_ece, binary_nll, compute_calibration_metrics
from hlm_ecg.data.ptbxl_labels import LABEL_ORDER


def test_binary_ece_is_zero_for_perfect_extreme_predictions():
    probs = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    ece, mce, bins = binary_ece(probs, targets, n_bins=5)
    assert ece == 0.0
    assert mce == 0.0
    assert sum(row["n_bin"] for row in bins) == 4


def test_binary_ece_is_positive_for_overconfident_predictions():
    probs = np.asarray([0.9, 0.9, 0.1, 0.1], dtype=np.float64)
    targets = np.asarray([1, 0, 0, 1], dtype=np.int64)
    ece, _, _ = binary_ece(probs, targets, n_bins=5)
    assert ece > 0.35


def test_brier_and_nll_are_correct_for_small_example():
    probs = np.asarray([0.25, 0.75], dtype=np.float64)
    targets = np.asarray([0, 1], dtype=np.int64)
    assert abs(binary_brier(probs, targets) - 0.0625) < 1e-12
    expected_nll = -0.5 * (np.log(0.75) + np.log(0.75))
    assert abs(binary_nll(probs, targets) - expected_nll) < 1e-12


def test_macro_ece_is_unweighted_mean_over_five_classes():
    targets = np.asarray(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    probs = np.where(targets == 1, 0.8, 0.2).astype(np.float64)
    metrics = compute_calibration_metrics(targets=targets, probs=probs, n_bins=5)
    expected = np.mean([metrics["per_class_ece"][label] for label in LABEL_ORDER])
    assert abs(metrics["macro_ece"] - expected) < 1e-12
    assert metrics["n_samples"] == 4

