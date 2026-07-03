import numpy as np

from hlm_ecg.calibration.temperature_scaling import apply_temperatures, fit_classwise_temperatures
from hlm_ecg.data.ptbxl_labels import LABEL_ORDER


def test_apply_temperature_one_keeps_logits_unchanged():
    logits = np.arange(10, dtype=np.float64).reshape(2, 5)
    calibrated = apply_temperatures(logits, np.ones(5, dtype=np.float64))
    np.testing.assert_allclose(calibrated, logits)


def test_classwise_temperature_scaling_returns_positive_temperatures():
    logits = np.asarray(
        [
            [2.0, -2.0, 1.0, -1.0, 0.5],
            [-2.0, 2.0, -1.0, 1.0, -0.5],
            [1.5, -1.5, 0.8, -0.8, 1.0],
            [-1.5, 1.5, -0.8, 0.8, -1.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    temps, results = fit_classwise_temperatures(logits, targets, max_iter=20)
    assert temps.shape == (len(LABEL_ORDER),)
    assert np.all(temps > 0)
    assert len(results) == len(LABEL_ORDER)
    assert all(result.n_val_samples == 4 for result in results)


def test_temperature_scaling_rejects_single_class_validation_label():
    logits = np.zeros((4, len(LABEL_ORDER)), dtype=np.float64)
    targets = np.zeros((4, len(LABEL_ORDER)), dtype=np.int64)
    targets[:, 1:] = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
    try:
        fit_classwise_temperatures(logits, targets, max_iter=5)
    except RuntimeError as exc:
        assert "validation labels contain one class" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for single-class validation label")

