from pathlib import Path

import yaml


CONFIG_PAIRS = (
    ("configs/random_dropout.yaml", "configs/seeds/random_dropout_seed7.yaml", 7),
    ("configs/random_dropout.yaml", "configs/seeds/random_dropout_seed123.yaml", 123),
    ("configs/subclass_auxiliary.yaml", "configs/seeds/subclass_auxiliary_seed7.yaml", 7),
    ("configs/subclass_auxiliary.yaml", "configs/seeds/subclass_auxiliary_seed123.yaml", 123),
    ("configs/confidence_consistency_lite.yaml", "configs/seeds/confidence_consistency_lite_seed7.yaml", 7),
    ("configs/confidence_consistency_lite.yaml", "configs/seeds/confidence_consistency_lite_seed123.yaml", 123),
)


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def collect_differences(a, b, prefix=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                diffs.append(child)
            else:
                diffs.extend(collect_differences(a[key], b[key], child))
    else:
        if a != b:
            diffs.append(prefix)
    return diffs


def allowed_difference(path: str) -> bool:
    return path == "paths.output_dir" or path.endswith(".seed") or path == "seed"


def test_seed_configs_only_change_seed_and_output_dir():
    for base_path, seed_path, seed in CONFIG_PAIRS:
        base = load_yaml(base_path)
        seeded = load_yaml(seed_path)
        diffs = collect_differences(base, seeded)
        assert diffs
        assert all(allowed_difference(path) for path in diffs), (base_path, seed_path, diffs)
        assert seeded["seed"] == seed
        assert str(seeded["paths"]["output_dir"]).startswith("outputs/week3_multiseed/")


def test_seed_configs_keep_locked_method_settings():
    a4a = load_yaml("configs/seeds/subclass_auxiliary_seed7.yaml")
    a5 = load_yaml("configs/seeds/confidence_consistency_lite_seed123.yaml")
    assert a4a["model"]["use_hierarchy_loss"] is False
    assert a4a["model"]["use_confidence_weighted_consistency"] is False
    assert a4a["subclass_auxiliary"]["lambda_sub"] == 0.2
    assert a5["model"]["use_confidence_weighted_consistency"] is True
    assert a5["confidence_consistency"]["lambda_cons"] == 0.05
    assert a5["hierarchy_loss"]["enabled"] is False

