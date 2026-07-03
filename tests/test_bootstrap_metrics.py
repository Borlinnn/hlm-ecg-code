import numpy as np
from sklearn.metrics import average_precision_score

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.statistics.bootstrap_metrics import compute_bootstrap_metrics


def test_macro_auprc_matches_sklearn_average():
    targets = np.asarray(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    probs = np.asarray(
        [
            [0.9, 0.1, 0.8, 0.2, 0.3],
            [0.2, 0.8, 0.7, 0.1, 0.9],
            [0.7, 0.6, 0.4, 0.8, 0.2],
            [0.1, 0.3, 0.2, 0.9, 0.7],
        ],
        dtype=np.float64,
    )
    logits = np.log(probs / (1.0 - probs))
    preds = (probs >= 0.5).astype(np.int64)

    metrics = compute_bootstrap_metrics(logits=logits, targets=targets, preds=preds, probs=probs)
    expected = np.mean(
        [average_precision_score(targets[:, idx], probs[:, idx]) for idx in range(len(LABEL_ORDER))]
    )
    assert abs(metrics["macro_auprc"] - expected) < 1e-12
    assert metrics["n_valid_auprc_labels"] == len(LABEL_ORDER)


def test_macro_auroc_safely_handles_single_class_label():
    targets = np.asarray(
        [
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    probs = np.clip(np.linspace(0.1, 0.9, targets.size).reshape(targets.shape), 1e-4, 1 - 1e-4)
    logits = np.log(probs / (1.0 - probs))
    preds = (probs >= 0.5).astype(np.int64)

    metrics = compute_bootstrap_metrics(logits=logits, targets=targets, preds=preds, probs=probs)
    assert np.isnan(metrics["per_class_auroc"]["NORM"])
    assert metrics["n_valid_auroc_labels"] == 4
    assert metrics["invalid_macro_auroc"] is False
    assert any("AUROC undefined for NORM" in warning for warning in metrics["warnings"])


def test_macro_auroc_invalid_when_too_few_labels_valid():
    targets = np.zeros((4, len(LABEL_ORDER)), dtype=np.int64)
    targets[:, 0] = [0, 1, 0, 1]
    targets[:, 1] = [1, 0, 1, 0]
    probs = np.clip(np.linspace(0.1, 0.9, targets.size).reshape(targets.shape), 1e-4, 1 - 1e-4)
    logits = np.log(probs / (1.0 - probs))
    preds = (probs >= 0.5).astype(np.int64)

    metrics = compute_bootstrap_metrics(logits=logits, targets=targets, preds=preds, probs=probs)
    assert metrics["n_valid_auroc_labels"] == 2
    assert metrics["invalid_macro_auroc"] is True
    assert np.isnan(metrics["macro_auroc"])
