import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path("scripts/lock_final_results.py")
    spec = importlib.util.spec_from_file_location("lock_final_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prediction_artifact_audit_reports_missing_without_fabrication(tmp_path):
    module = load_module()
    audit = module.audit_prediction_artifacts(tmp_path)
    assert audit["artifact_complete"] is False
    assert audit["missing_count"] == len(module.KEY_PREDICTION_METHODS) * len(module.KEY_PREDICTION_PATTERNS)
    assert audit["existing_count"] == 0
    assert audit["all_key_scripts_support_save_predictions"] is True


def test_prediction_artifact_required_columns_include_truth_and_probs(tmp_path):
    module = load_module()
    audit = module.audit_prediction_artifacts(tmp_path)
    required = set(audit["required_columns"])
    for label in module.LABELS:
        assert f"y_true_{label}" in required
        assert f"prob_{label}" in required
    assert "threshold_source_split" in required
    assert "ecg_id" in required
