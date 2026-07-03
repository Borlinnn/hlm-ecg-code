import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.prediction_artifacts import build_prediction_output_path, prediction_rows, save_predictions_csv
from hlm_ecg.statistics.bootstrap import METHODS, PATTERNS


def load_script_module():
    path = Path("scripts/run_calibration_audit.py")
    spec = importlib.util.spec_from_file_location("run_calibration_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fake_prediction_tree(root: Path, *, n_rows: int = 20) -> None:
    targets = np.asarray(
        [[(row_idx + label_idx) % 2 for label_idx in range(len(LABEL_ORDER))] for row_idx in range(n_rows)],
        dtype=np.int64,
    )
    base_logits = np.linspace(-2.0, 2.0, n_rows * len(LABEL_ORDER)).reshape(n_rows, len(LABEL_ORDER))
    for method_idx, method_id in enumerate(METHODS):
        for split in ("val", "test"):
            strat_fold = 9 if split == "val" else 10
            for pattern_idx, pattern in enumerate(PATTERNS):
                logits = base_logits + 0.05 * method_idx - 0.01 * pattern_idx
                collected = {
                    "logits": logits,
                    "targets": targets,
                    "ecg_ids": np.arange(1, n_rows + 1),
                    "patient_ids": np.arange(1001, 1001 + n_rows),
                    "strat_folds": np.full(n_rows, strat_fold),
                    "availability_masks": np.ones((n_rows, 12), dtype=np.float32),
                    "splits": [split] * n_rows,
                }
                rows = prediction_rows(
                    method_id=method_id,
                    pattern=pattern,
                    fill_mode="mean_fill",
                    split=split,
                    random_seed=42,
                    threshold_source_split="val",
                    thresholds=[0.5] * len(LABEL_ORDER),
                    collected=collected,
                )
                path = build_prediction_output_path(
                    root,
                    method_id=method_id,
                    fill_mode="mean_fill",
                    split=split,
                    pattern=pattern,
                )
                save_predictions_csv(path, rows)


def test_run_calibration_audit_smoke_outputs_required_files(tmp_path):
    predictions_dir = tmp_path / "predictions"
    out_dir = tmp_path / "calibration"
    make_fake_prediction_tree(predictions_dir)
    module = load_script_module()

    result = module.run(
        Namespace(
            predictions_dir=predictions_dir,
            out_dir=out_dir,
            fill_mode="mean_fill",
            n_bins=5,
            methods=["A1_random_dropout", "A4a_subclass_auxiliary"],
            patterns=["full", "random-6"],
            smoke_test=True,
            save_calibrated_predictions=True,
        )
    )

    assert result["temperature_fit_split"] == "val"
    assert result["evaluation_split"] == "test"
    assert (out_dir / "calibration_temperature_parameters.csv").exists()
    assert (out_dir / "calibration_metrics_by_pattern.csv").exists()
    assert (out_dir / "calibration_aggregate_summary.csv").exists()
    assert (out_dir / "reliability_curve_data.json").exists()
    assert (out_dir / "calibration_decision.json").exists()
    assert (out_dir / "calibrated_predictions" / "full_val_classwise_ts" / "A4a_subclass_auxiliary" / "test" / "random_6.csv").exists()

