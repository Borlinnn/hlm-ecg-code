import numpy as np

from hlm_ecg.evaluation.metrics import compute_multilabel_metrics, tune_thresholds_on_validation


def test_metrics_run_on_multilabel_inputs():
    logits = np.array(
        [
            [2.0, -1.0, 0.5, -0.2, 1.2],
            [-1.0, 2.0, -0.5, 0.3, -0.7],
            [0.2, -0.1, 1.5, 2.0, -1.2],
            [-0.5, 0.4, -0.2, -1.0, 1.0],
        ]
    )
    targets = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
        ]
    )
    thresholds = tune_thresholds_on_validation(logits, targets)
    metrics = compute_multilabel_metrics(logits, targets, thresholds=thresholds["threshold_array"])
    assert metrics["macro_auprc"] is not None
    assert metrics["macro_f1"] >= 0.0
    assert metrics["bce_nll"] > 0.0


def test_threshold_tuning_uses_only_passed_validation_arrays():
    val_logits = np.zeros((4, 5), dtype=float)
    val_targets = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
        ]
    )
    test_targets_a = np.zeros((4, 5), dtype=int)
    test_targets_b = np.ones((4, 5), dtype=int)
    thresholds_a = tune_thresholds_on_validation(val_logits, val_targets)["threshold_array"]
    thresholds_b = tune_thresholds_on_validation(val_logits, val_targets)["threshold_array"]
    assert thresholds_a == thresholds_b
    assert not np.array_equal(test_targets_a, test_targets_b)
