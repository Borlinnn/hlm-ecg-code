import importlib.util
from pathlib import Path

import pandas as pd

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER


def load_module():
    path = Path("scripts/run_reviewer_defense_existing_predictions.py")
    spec = importlib.util.spec_from_file_location("run_reviewer_defense_existing_predictions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_prediction(path: Path, *, method_id: str, pattern: str) -> None:
    rows = []
    targets = [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ]
    for idx, y in enumerate(targets):
        row = {
            "ecg_id": idx,
            "patient_id": idx,
            "split": "test",
            "strat_fold": 10,
            "method_id": method_id,
            "pattern": pattern,
            "fill_mode": "mean_fill",
            "random_seed": 20240604,
            "threshold_source_split": "val",
        }
        for lead_idx in range(12):
            row[f"availability_mask_{lead_idx}"] = 1
        for label_idx, label in enumerate(LABEL_ORDER):
            prob = 0.8 if y[label_idx] else 0.2
            row[f"y_true_{label}"] = y[label_idx]
            row[f"logit_{label}"] = 1.38629436112 if y[label_idx] else -1.38629436112
            row[f"prob_{label}"] = prob
            row[f"pred_{label}"] = int(prob >= 0.5)
            row[f"threshold_{label}"] = 0.5
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_existing_prediction_analysis_writes_expected_outputs(tmp_path):
    module = load_module()
    input_dir = tmp_path / "predictions"
    write_prediction(input_dir / "random_lead_dropout_seed7" / "mean_fill" / "test" / "full.csv", method_id="random_lead_dropout_seed7", pattern="full")
    write_prediction(input_dir / "random_lead_dropout_seed7" / "mean_fill" / "test" / "random_6.csv", method_id="random_lead_dropout_seed7", pattern="random-6")
    output_dir = tmp_path / "analysis"
    summary = module.run_analysis(input_dir=input_dir, output_dir=output_dir)
    assert summary["n_prediction_files"] == 2
    assert (output_dir / "prediction_metric_rows.csv").exists()
    assert (output_dir / "clean_vs_robust_pareto_data.csv").exists()
    assert (output_dir / "class_by_pattern_heatmap_data.csv").exists()
    metrics = pd.read_csv(output_dir / "prediction_metric_rows.csv")
    assert {"macro_auroc", "macro_auprc", "macro_f1", "macro_ece", "macro_brier", "bce_nll"}.issubset(metrics.columns)
