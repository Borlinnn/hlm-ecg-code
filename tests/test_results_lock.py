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


def test_method_registry_contains_A0_to_A5_lite():
    module = load_module()
    method_ids = [method.method_id for method in module.METHOD_REGISTRY]
    assert method_ids == [
        "A0_full_no_masking",
        "A1_random_dropout",
        "A2_structured_masking",
        "A3_availability_embedding",
        "A4a_subclass_auxiliary",
        "A4b_hierarchy_loss",
        "A5_confidence_consistency_0p10",
        "A5_lite_confidence_consistency_0p05",
    ]


def test_final_decision_locks_A4a_and_val_thresholds():
    module = load_module()
    rows = module.build_summary_rows()
    decision = module.make_decision(rows)
    assert decision["final_robustness_candidate"] == "A4a_subclass_auxiliary"
    assert decision["stop_model_structure_experiments"] is True
    assert all(row["thresholds_source_split"] == "val" for row in rows)
    assert all(row["records500_used"] is False for row in rows)


def test_hard_average_definitions_are_stable():
    module = load_module()
    method = next(m for m in module.METHOD_REGISTRY if m.method_id == "A4a_subclass_auxiliary")
    store = module.read_pattern_metrics(method, "mean_fill")
    hard_structured = module.average_metric(store, module.HARD_STRUCTURED_PATTERNS, "macro_auprc")
    hard_overall = module.average_metric(store, module.HARD_OVERALL_PATTERNS, "macro_auprc")
    assert hard_structured > hard_overall
    assert abs(hard_structured - 0.7764715843007397) < 1e-9
    assert abs(hard_overall - 0.774649434718888) < 1e-9
