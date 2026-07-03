import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    path = Path("scripts/summarize_multiseed_results.py")
    spec = importlib.util.spec_from_file_location("summarize_multiseed_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_seed_output(directory: Path, *, full_auprc: float, hard_base: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "best_model.pt").write_text("mock", encoding="utf-8")
    (directory / "train_log.csv").write_text(
        "epoch,train_loss,val_loss,val_macro_auprc,val_macro_auroc,val_macro_f1\n"
        f"1,0.5,0.4,{full_auprc},0.9,0.7\n",
        encoding="utf-8",
    )
    (directory / "val_metrics.json").write_text(json.dumps({"macro_auprc": full_auprc}), encoding="utf-8")
    full_metrics = {
        "macro_auroc": 0.9,
        "macro_auprc": full_auprc,
        "macro_f1": 0.7,
        "per_class_auprc": {"NORM": 0.9, "MI": hard_base, "STTC": 0.8, "CD": hard_base, "HYP": hard_base},
    }
    (directory / "test_full_metrics.json").write_text(json.dumps(full_metrics), encoding="utf-8")
    (directory / "thresholds_val.json").write_text(json.dumps({"source_split": "val"}), encoding="utf-8")
    patterns = {}
    values = {
        "full": full_auprc,
        "random-1": hard_base + 0.01,
        "random-3": hard_base + 0.02,
        "random-6": hard_base,
        "limb-only / precordial-missing": hard_base,
        "precordial-only / limb-missing": hard_base,
        "V1-V3 missing": hard_base,
        "V4-V6 missing": hard_base,
    }
    for pattern, value in values.items():
        patterns[pattern] = {
            "metrics": {
                "macro_auroc": 0.9,
                "macro_auprc": value,
                "macro_f1": 0.7,
                "per_class_auprc": {"NORM": 0.9, "MI": value, "STTC": 0.8, "CD": value, "HYP": value},
            }
        }
    (directory / "test_missing_patterns_mean_fill.json").write_text(json.dumps({"patterns": patterns}), encoding="utf-8")
    (directory / "test_missing_patterns_mean_fill.csv").write_text("pattern,macro_auprc\nfull,0.8\n", encoding="utf-8")


def test_multiseed_summary_reads_mock_outputs(tmp_path):
    module = load_module()
    registry = []
    for method, full, hard in (
        ("A1_random_dropout", 0.82, 0.75),
        ("A4a_subclass_auxiliary", 0.815, 0.78),
        ("A5_lite_confidence_consistency_0p05", 0.83, 0.77),
    ):
        for seed in (42, 7, 123):
            out = tmp_path / method / f"seed{seed}"
            write_seed_output(out, full_auprc=full + seed * 0.00001, hard_base=hard + seed * 0.00001)
            registry.append(module.SeedRun(method, seed, out))
    result = module.run(tmp_path / "summary", registry=tuple(registry))
    assert result["n_seed_runs"] == 9
    assert result["a4a_stable_final_robustness_candidate"] is True
    assert result["records500_used"] is False
    assert (tmp_path / "summary" / "multiseed_summary_mean_std.csv").exists()
    assert (tmp_path / "summary" / "multiseed_delta_summary.json").exists()


def test_hard_average_pattern_sets_are_stable():
    module = load_module()
    assert module.STRUCTURED_HARD_PATTERNS == (
        "limb-only / precordial-missing",
        "precordial-only / limb-missing",
        "V1-V3 missing",
        "V4-V6 missing",
    )
    assert module.HARD_OVERALL_PATTERNS[0] == "random-6"

