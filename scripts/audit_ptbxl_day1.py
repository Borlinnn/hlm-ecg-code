#!/usr/bin/env python3
"""Run Day 1 PTB-XL data audit for HLM-ECG."""

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.data.lead_patterns import build_required_patterns
from hlm_ecg.data.normalization import audit_waveforms_and_compute_train_stats
from hlm_ecg.data.ptbxl_labels import (
    EXPECTED_SUPERCLASS_COUNTS,
    LABEL_ORDER,
    build_labeled_index,
    load_metadata,
    load_scp_statements,
    observed_counts,
)
from hlm_ecg.data.splits import assign_official_splits, assert_patient_disjoint, split_summary
from hlm_ecg.data.waveforms import CANONICAL_LEADS, assert_no_records500


def json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_markdown(report: Mapping[str, object], path: Path) -> None:
    labels = report["labels"]
    splits = report["splits"]
    waveform = report["waveforms"]
    normalization = report["normalization"]

    lines = [
        "# PTB-XL Day 1 Sanity Report",
        "",
        f"Root: `{report['root']}`",
        f"Records500 absent: `{report['records500_absent']}`",
        "",
        "## Labels",
        "",
        "| Class | Observed | Expected |",
        "|---|---:|---:|",
    ]
    for label in LABEL_ORDER:
        lines.append(
            f"| {label} | {labels['observed_counts'][label]} | "
            f"{labels['expected_counts'][label]} |"
        )

    lines.extend(
        [
            "",
            "## Splits",
            "",
            "| Split | Records | Patients |",
            "|---|---:|---:|",
        ]
    )
    for split in ("train", "val", "test"):
        item = splits["summary"][split]
        lines.append(f"| {split} | {item['records']} | {item['patients']} |")

    lines.extend(
        [
            "",
            f"Patient leakage counts: `{splits['patient_leakage_counts']}`",
            "",
            "## Waveforms",
            "",
            f"Records checked: `{waveform['records_checked']}`",
            f"Raw shape: `{waveform['raw_shape']}`",
            f"Model input shape: `{waveform['model_input_shape']}`",
            f"Sampling rate: `{waveform['fs']}`",
            f"Canonical leads: `{waveform['lead_names_canonical']}`",
            "",
            "## Train Normalization",
            "",
            f"Train records used: `{normalization['train_records_used_for_normalization']}`",
            "",
            "| Lead | Mean | Std |",
            "|---|---:|---:|",
        ]
    )
    for lead in CANONICAL_LEADS:
        lines.append(
            f"| {lead} | {normalization['mean_by_lead'][lead]:.8f} | "
            f"{normalization['std_by_lead'][lead]:.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PTB-XL Day 1 data for HLM-ECG.")
    parser.add_argument("--root", type=Path, default=Path("data/ptb-xl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/day1_audit"))
    args = parser.parse_args()

    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    assert_no_records500(root)

    metadata = load_metadata(root)
    scp_statements = load_scp_statements(root)
    index = build_labeled_index(metadata, scp_statements)
    index = assign_official_splits(index)
    leakage = assert_patient_disjoint(index)
    split_stats = split_summary(index)

    label_counts = observed_counts(index)
    sensitivity_counts = {
        label: int(index[f"{label}_likelihood_positive"].sum()) for label in LABEL_ORDER
    }

    patterns = build_required_patterns(CANONICAL_LEADS)
    norm_path = out / "train_norm_stats.npz"
    waveform_report = audit_waveforms_and_compute_train_stats(root, index, norm_path)

    index_columns = [
        "ecg_id",
        "patient_id",
        "strat_fold",
        "split",
        "filename_lr",
        "diagnostic_superclasses",
        "diagnostic_superclasses_likelihood_positive",
    ] + list(LABEL_ORDER) + [f"{label}_likelihood_positive" for label in LABEL_ORDER]
    index[index_columns].to_csv(out / "ptbxl_day1_index.csv", index=False)

    report = {
        "root": str(root),
        "records500_absent": not (root / "records500").exists(),
        "labels": {
            "label_order": list(LABEL_ORDER),
            "observed_counts": label_counts,
            "expected_counts": EXPECTED_SUPERCLASS_COUNTS,
            "counts_match_expected": label_counts == EXPECTED_SUPERCLASS_COUNTS,
            "likelihood_positive_sensitivity_counts": sensitivity_counts,
            "aggregation": "official-style diagnostic SCP code presence",
        },
        "splits": {
            "definition": {
                "train": "strat_fold in 1..8",
                "val": "strat_fold == 9",
                "test": "strat_fold == 10",
            },
            "summary": split_stats,
            "patient_leakage_counts": leakage,
        },
        "waveforms": waveform_report,
        "lead_patterns": patterns,
        "normalization": waveform_report,
        "outputs": {
            "ptbxl_day1_index_csv": str(out / "ptbxl_day1_index.csv"),
            "train_norm_stats_npz": str(norm_path),
            "sanity_report_json": str(out / "sanity_report.json"),
            "sanity_report_md": str(out / "sanity_report.md"),
        },
    }

    if not report["labels"]["counts_match_expected"]:
        raise RuntimeError("Label counts failed after prior assertion")

    json_path = out / "sanity_report.json"
    md_path = out / "sanity_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
