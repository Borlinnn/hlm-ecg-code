import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.prediction_artifacts import build_prediction_output_path, prediction_rows, save_predictions_csv
from hlm_ecg.statistics.bootstrap import METHODS, PATTERNS


def load_script_module():
    path = Path("scripts/run_paired_bootstrap_ci.py")
    spec = importlib.util.spec_from_file_location("run_paired_bootstrap_ci", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fake_prediction_tree(root: Path, *, n_rows: int = 12) -> None:
    targets = np.asarray(
        [[(row_idx + label_idx) % 2 for label_idx in range(len(LABEL_ORDER))] for row_idx in range(n_rows)],
        dtype=np.int64,
    )
    base_logits = np.linspace(-1.5, 1.5, n_rows * len(LABEL_ORDER)).reshape(n_rows, len(LABEL_ORDER))
    for method_idx, method_id in enumerate(METHODS):
        for pattern_idx, pattern in enumerate(PATTERNS):
            logits = base_logits + 0.02 * method_idx - 0.01 * pattern_idx
            collected = {
                "logits": logits,
                "targets": targets,
                "ecg_ids": np.arange(1, n_rows + 1),
                "patient_ids": np.arange(1001, 1001 + n_rows),
                "strat_folds": np.full(n_rows, 10),
                "availability_masks": np.ones((n_rows, 12), dtype=np.float32),
                "splits": ["test"] * n_rows,
            }
            rows = prediction_rows(
                method_id=method_id,
                pattern=pattern,
                fill_mode="mean_fill",
                split="test",
                random_seed=42,
                threshold_source_split="val",
                thresholds=[0.5] * len(LABEL_ORDER),
                collected=collected,
            )
            path = build_prediction_output_path(
                root,
                method_id=method_id,
                fill_mode="mean_fill",
                split="test",
                pattern=pattern,
            )
            save_predictions_csv(path, rows)


def test_run_paired_bootstrap_ci_smoke_outputs_required_files(tmp_path):
    predictions_dir = tmp_path / "predictions"
    out_dir = tmp_path / "bootstrap"
    make_fake_prediction_tree(predictions_dir)
    module = load_script_module()

    result = module.run(
        Namespace(
            predictions_dir=predictions_dir,
            out_dir=out_dir,
            n_bootstrap=5,
            seed=42,
            split="test",
            fill_mode="mean_fill",
            sampling_unit="patient",
            smoke_test=True,
        )
    )

    assert result["sampling_unit"] == "patient"
    assert result["split"] == "test"
    assert (out_dir / "bootstrap_method_ci.csv").exists()
    assert (out_dir / "paired_delta_ci.csv").exists()
    assert (out_dir / "table1_main_robustness_with_ci.csv").exists()
    assert (out_dir / "appendix_paired_delta_table.tex").exists()
    assert (out_dir / "figure2_degradation_curve_ci_data.json").exists()
    assert (out_dir / "figure3_heatmap_delta_ci_data.json").exists()
    assert (out_dir / "bootstrap_run_config.json").exists()
