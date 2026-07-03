import importlib.util
from pathlib import Path


def load_generator_module():
    path = Path("scripts/generate_reviewer_defense_configs.py")
    spec = importlib.util.spec_from_file_location("generate_reviewer_defense_configs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_runner_module():
    path = Path("scripts/run_reviewer_defense_config.py")
    spec = importlib.util.spec_from_file_location("run_reviewer_defense_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reviewer_defense_plan_counts_and_scope():
    module = load_generator_module()
    plan = module.build_experiment_plan()
    primary = [row for row in plan if row["group"] == "primary"]
    assert len(primary) == 3 * 5 * 6
    assert {row["backbone"] for row in primary} == {"resnet1d_tiny", "xresnet1d101_like", "inception_time1d"}
    assert {row["seed"] for row in primary} == {7, 42, 123, 2024, 2025}
    assert {row["method_id"] for row in primary} == {
        "M0_full_no_masking",
        "M1_random_dropout",
        "M2_structured_masking",
        "M3_random_dropout_plus_availability",
        "M4_structured_plus_availability",
        "M6_structured_plus_availability_plus_subclass",
    }
    assert all("reviewer_defense_20260701" in row["output_dir"] for row in plan)
    serialized = "\n".join(str(row) for row in plan)
    assert "records500" not in serialized
    assert "filename_hr" not in serialized


def test_config_for_m5_allows_subclass_without_availability():
    module = load_generator_module()
    row = next(
        row
        for row in module.build_experiment_plan()
        if row["method_id"] == "M5_structured_plus_subclass_no_availability"
        and row["backbone"] == "xresnet1d101_like"
        and row["seed"] == 42
    )
    config = module.build_config(row)
    assert config["model"]["enable_subclass_auxiliary"] is True
    assert config["model"].get("use_availability_embedding", False) is False
    assert config["subclass_auxiliary"]["allow_without_availability"] is True
    assert config["structured_masking"]["enabled"] is True


def test_dry_run_does_not_write_configs(tmp_path):
    module = load_generator_module()
    summary = module.write_configs(output_dir=tmp_path, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["n_configs"] > 0
    assert not list(tmp_path.rglob("*.yaml"))


def test_smoke_runner_uses_smoke_only_outputs():
    runner = load_runner_module()
    config_path = Path(
        "configs/reviewer_defense_20260701/primary/xresnet1d101_like/"
        "M1_random_dropout_seed42.yaml"
    )
    plan = runner.command_plan(config_path, smoke_test=True, save_predictions=True)
    assert plan["configured_output_dir"] != plan["output_dir"]
    assert "outputs/reviewer_defense_20260701/smoke/" in plan["output_dir"]
    assert "results/reviewer_defense_20260701/smoke_predictions" in plan["eval_cmd"]
    assert "--output-dir" in plan["train_cmd"]


def test_full_runner_keeps_configured_output_dir():
    runner = load_runner_module()
    config_path = Path(
        "configs/reviewer_defense_20260701/primary/xresnet1d101_like/"
        "M1_random_dropout_seed42.yaml"
    )
    plan = runner.command_plan(config_path, smoke_test=False, save_predictions=True)
    assert plan["configured_output_dir"] == plan["output_dir"]
    assert "outputs/reviewer_defense_20260701/smoke/" not in plan["output_dir"]
    assert "results/reviewer_defense_20260701/predictions" in plan["eval_cmd"]


def test_specialist_runner_uses_gated_specialist_training_script():
    runner = load_runner_module()
    config_path = Path(
        "configs/reviewer_defense_20260701/specialist_upper_bound/xresnet1d101_like/"
        "SPECIALIST_fixed_pattern_V1_V3_missing_seed7.yaml"
    )
    plan = runner.command_plan(config_path, smoke_test=False, save_predictions=True)
    assert plan["train_script"] == "scripts/train_week6_fixed_pattern_specialist.py"
    assert plan["eval_cmd"] == []
    assert plan["train_env"]["WEEK6_ALLOW_SPECIALIST_TRAINING"] == "true"


def test_specialist_config_uses_canonical_pattern_name():
    module = load_generator_module()
    row = next(
        row
        for row in module.build_experiment_plan()
        if row["method_id"] == "SPECIALIST_fixed_pattern"
        and row["tag"] == "V1_V3_missing"
        and row["seed"] == 7
    )
    config = module.build_config(row)
    assert config["week6_specialist"]["pattern"] == "V1-V3 missing"
