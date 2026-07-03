#!/usr/bin/env python3
"""Audit saved HLM-ECG per-sample prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.prediction_artifacts import (
    PREDICTION_REQUIRED_COLUMNS,
    build_prediction_output_path,
    count_csv_rows,
    validate_prediction_csv_schema,
)

KEY_METHODS = [
    "A0_full_no_masking",
    "A1_random_dropout",
    "A2_structured_masking",
    "A4a_subclass_auxiliary",
    "A5_lite_confidence_consistency_0p05",
]
PATTERNS = [
    "full",
    "random-1",
    "random-3",
    "random-6",
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
]
EXPECTED_SPLIT_ROWS = {"val": 2183, "test": 2198}


def unique_column_values(path: Path, column: str) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values.add(str(row.get(column, "")))
    return values


def audit_predictions(
    predictions_dir: Path,
    *,
    methods: list[str] | None = None,
    fill_modes: list[str] | None = None,
    splits: list[str] | None = None,
    patterns: list[str] | None = None,
) -> dict[str, Any]:
    methods = KEY_METHODS if methods is None else methods
    fill_modes = ["mean_fill"] if fill_modes is None else fill_modes
    splits = ["val", "test"] if splits is None else splits
    patterns = PATTERNS if patterns is None else patterns
    existing = []
    missing = []
    invalid = []
    for method_id in methods:
        for fill_mode in fill_modes:
            for split in splits:
                expected_rows = EXPECTED_SPLIT_ROWS.get(split)
                for pattern in patterns:
                    path = build_prediction_output_path(
                        predictions_dir,
                        method_id=method_id,
                        fill_mode=fill_mode,
                        split=split,
                        pattern=pattern,
                    )
                    record = {
                        "method_id": method_id,
                        "fill_mode": fill_mode,
                        "split": split,
                        "pattern": pattern,
                        "csv_path": str(path),
                    }
                    if not path.exists():
                        missing.append(record)
                        continue
                    missing_columns = validate_prediction_csv_schema(path)
                    n_rows = count_csv_rows(path)
                    threshold_sources = unique_column_values(path, "threshold_source_split")
                    split_values = unique_column_values(path, "split")
                    issues = []
                    if missing_columns:
                        issues.append({"missing_columns": missing_columns})
                    if expected_rows is not None and n_rows != expected_rows:
                        issues.append({"row_count": n_rows, "expected_rows": expected_rows})
                    if threshold_sources != {"val"}:
                        issues.append({"threshold_source_split_values": sorted(threshold_sources)})
                    if split_values != {split}:
                        issues.append({"split_values": sorted(split_values)})
                    has_logits = all(not validate_prediction_csv_schema(path).count(f"logit_{label}") for label in LABEL_ORDER)
                    info = {
                        **record,
                        "n_rows": n_rows,
                        "expected_rows": expected_rows,
                        "has_logits": has_logits,
                        "has_probabilities": all(f"prob_{label}" not in missing_columns for label in LABEL_ORDER),
                        "has_thresholds": all(f"threshold_{label}" not in missing_columns for label in LABEL_ORDER),
                        "threshold_source_split_values": sorted(threshold_sources),
                    }
                    if issues:
                        invalid.append({**info, "issues": issues})
                    else:
                        existing.append(info)
    manifest_csv = predictions_dir / "prediction_manifest.csv"
    records500_used = Path("data/ptb-xl/records500").exists()
    complete = not missing and not invalid and not records500_used
    return {
        "predictions_dir": str(predictions_dir),
        "artifact_complete": complete,
        "mean_fill_complete": complete and fill_modes == ["mean_fill"],
        "methods": methods,
        "fill_modes": fill_modes,
        "splits": splits,
        "patterns": patterns,
        "expected_val_rows": EXPECTED_SPLIT_ROWS["val"],
        "expected_test_rows": EXPECTED_SPLIT_ROWS["test"],
        "expected_csv_count": len(methods) * len(fill_modes) * len(splits) * len(patterns),
        "existing_count": len(existing),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "existing": existing,
        "missing": missing,
        "invalid": invalid,
        "manifest_exists": manifest_csv.exists(),
        "manifest_path": str(manifest_csv),
        "required_columns": list(PREDICTION_REQUIRED_COLUMNS),
        "records500_used": records500_used,
        "can_proceed_paired_bootstrap_ci": complete,
        "can_proceed_calibration_audit": complete,
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Prediction Artifact Audit After Save",
        "",
        f"- artifact complete: `{audit['artifact_complete']}`",
        f"- expected CSV count: `{audit['expected_csv_count']}`",
        f"- existing count: `{audit['existing_count']}`",
        f"- missing count: `{audit['missing_count']}`",
        f"- invalid count: `{audit['invalid_count']}`",
        f"- manifest exists: `{audit['manifest_exists']}`",
        f"- records500 used: `{audit['records500_used']}`",
        f"- can proceed paired bootstrap / CI: `{audit['can_proceed_paired_bootstrap_ci']}`",
        f"- can proceed calibration audit: `{audit['can_proceed_calibration_audit']}`",
        "",
    ]
    if audit["missing"]:
        lines.extend(["## Missing", ""])
        for item in audit["missing"][:80]:
            lines.append(f"- {item['method_id']} / {item['fill_mode']} / {item['split']} / {item['pattern']}")
    if audit["invalid"]:
        lines.extend(["", "## Invalid", ""])
        for item in audit["invalid"][:80]:
            lines.append(f"- {item['csv_path']}: {item['issues']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HLM-ECG saved prediction artifacts.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/week3_results_lock"))
    parser.add_argument("--methods", nargs="+", default=KEY_METHODS)
    parser.add_argument("--fill-modes", nargs="+", default=["mean_fill"])
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--patterns", nargs="+", default=["all"])
    args = parser.parse_args()

    patterns = PATTERNS if args.patterns == ["all"] else args.patterns
    audit = audit_predictions(
        args.predictions_dir,
        methods=args.methods,
        fill_modes=args.fill_modes,
        splits=args.splits,
        patterns=patterns,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "prediction_artifact_audit_after_save.json"
    md_path = args.out_dir / "prediction_artifact_audit_after_save.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
