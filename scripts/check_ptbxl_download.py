#!/usr/bin/env python3
"""Verify the PTB-XL v1.0.3 records100 download for HLM-ECG.

This script checks dataset availability only. It does not parse labels for
training, create model inputs, download data, or touch any model code.
"""

import argparse
import json
from pathlib import Path

EXPECTED_ROWS = 21799
EXPECTED_LR_HEA = 21799
EXPECTED_LR_DAT = 21799
REQUIRED_FILES = ("ptbxl_database.csv", "scp_statements.csv", "records100")
OPTIONAL_FILES = ("LICENSE.txt", "RECORDS", "SHA256SUMS.txt", "example_physionet.py")
REQUIRED_COLUMNS = ("filename_lr", "strat_fold", "patient_id", "scp_codes")


def canonical_lead_name(name):
    mapping = {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
        "avr": "aVR",
        "avl": "aVL",
        "avf": "aVF",
    }
    text = str(name).strip()
    return mapping.get(text, text)


def init_report(root):
    return {
        "dataset_root": str(root),
        "required_files": {
            "ptbxl_database.csv": False,
            "scp_statements.csv": False,
            "records100": False,
        },
        "optional_files": {
            "LICENSE.txt": False,
            "RECORDS": False,
            "SHA256SUMS.txt": False,
            "example_physionet.py": False,
        },
        "metadata": {
            "n_rows": None,
            "expected_rows": EXPECTED_ROWS,
            "required_columns_present": False,
            "missing_columns": list(REQUIRED_COLUMNS),
            "read_error": "",
        },
        "records100": {
            "n_lr_hea": 0,
            "n_lr_dat": 0,
            "expected_lr_hea": EXPECTED_LR_HEA,
            "expected_lr_dat": EXPECTED_LR_DAT,
            "missing_waveform_pairs": None,
            "missing_examples": [],
        },
        "waveform_read_check": {
            "wfdb_available": False,
            "example_filename_lr": "",
            "shape": None,
            "fs": None,
            "lead_names": [],
            "units": [],
            "pass": False,
            "error": "",
        },
        "records500": {
            "exists": False,
            "required": False,
            "note": "records500 is not needed for HLM-ECG and must not be used for this task.",
        },
        "dependencies": {
            "pandas_available": False,
            "numpy_available": False,
            "wfdb_available": False,
            "missing_packages": [],
        },
        "overall_pass": False,
    }


def write_reports(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "download_check.json"
    md_path = out_dir / "download_check.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")


def status_text(value):
    return "PASS" if value else "FAIL"


def markdown_report(report):
    required = report["required_files"]
    optional = report["optional_files"]
    metadata = report["metadata"]
    records = report["records100"]
    waveform = report["waveform_read_check"]
    dependencies = report["dependencies"]

    next_actions = []
    if dependencies["missing_packages"]:
        next_actions.append(
            "Install missing Python dependencies in the intended project environment: "
            + ", ".join(dependencies["missing_packages"])
        )
    if not all(required.values()):
        next_actions.append("Download only the missing PTB-XL v1.0.3 records100 files from official PhysioNet.")
    if metadata["n_rows"] != EXPECTED_ROWS:
        next_actions.append("Verify `ptbxl_database.csv`; expected exactly 21,799 rows.")
    if not metadata["required_columns_present"]:
        next_actions.append("Verify required metadata columns: filename_lr, strat_fold, patient_id, scp_codes.")
    if records["n_lr_hea"] != EXPECTED_LR_HEA or records["n_lr_dat"] != EXPECTED_LR_DAT:
        next_actions.append("Complete the `records100` low-resolution WFDB files; do not download records500.")
    if records["missing_waveform_pairs"] not in (0, None):
        next_actions.append("Repair missing `.hea` / `.dat` pairs referenced by `filename_lr`.")
    if not waveform["pass"]:
        next_actions.append("Fix WFDB waveform read check in the project Python environment.")
    if not next_actions:
        next_actions.append("No data repair needed. Proceed to Day 1 audit.")

    lines = [
        "# PTB-XL Download Check",
        "",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Final status: `{status_text(report['overall_pass'])}`",
        f"- records500 exists: `{report['records500']['exists']}`",
        f"- records500 required: `{report['records500']['required']}`",
        f"- records500 note: {report['records500']['note']}",
        "",
        "## Required File Status",
        "",
        "| File/Directory | Exists |",
        "|---|---:|",
    ]
    for name in REQUIRED_FILES:
        lines.append(f"| `{name}` | `{required[name]}` |")
    lines.extend(["", "## Optional File Status", "", "| File | Exists |", "|---|---:|"])
    for name in OPTIONAL_FILES:
        lines.append(f"| `{name}` | `{optional[name]}` |")
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            f"- Row count: `{metadata['n_rows']}`",
            f"- Expected rows: `{metadata['expected_rows']}`",
            f"- Required columns present: `{metadata['required_columns_present']}`",
            f"- Missing columns: `{metadata['missing_columns']}`",
            f"- Metadata read error: `{metadata['read_error']}`",
            "",
            "## records100 Counts",
            "",
            f"- `*_lr.hea`: `{records['n_lr_hea']}` / `{records['expected_lr_hea']}`",
            f"- `*_lr.dat`: `{records['n_lr_dat']}` / `{records['expected_lr_dat']}`",
            f"- Missing waveform pairs from `filename_lr`: `{records['missing_waveform_pairs']}`",
            f"- Missing examples: `{records['missing_examples']}`",
            "",
            "## WFDB Waveform Read Result",
            "",
            f"- WFDB available: `{waveform['wfdb_available']}`",
            f"- Example `filename_lr`: `{waveform['example_filename_lr']}`",
            f"- Shape: `{waveform['shape']}`",
            f"- Sampling rate: `{waveform['fs']}`",
            f"- Lead names: `{waveform['lead_names']}`",
            f"- Units: `{waveform['units']}`",
            f"- Pass: `{waveform['pass']}`",
            f"- Error: `{waveform['error']}`",
            "",
            "## Final Decision",
            "",
            f"`{status_text(report['overall_pass'])}`",
            "",
            "## Next Actions",
            "",
        ]
    )
    for action in next_actions:
        lines.append(f"- {action}")
    lines.extend(["", "If this check passes, the next Day 1 audit command is:", "", "```bash", "python scripts/audit_ptbxl_day1.py --root data/ptb-xl --out outputs/day1_audit", "```", ""])
    return "\n".join(lines)


def count_files(records100):
    if not records100.exists():
        return 0, 0
    return len(list(records100.rglob("*_lr.hea"))), len(list(records100.rglob("*_lr.dat")))


def check_filename_pairs(root, db, max_examples=20):
    missing = []
    missing_count = 0
    for filename_lr in db["filename_lr"].astype(str).tolist():
        record = root / filename_lr
        hea_exists = record.with_suffix(".hea").exists()
        dat_exists = record.with_suffix(".dat").exists()
        if not hea_exists or not dat_exists:
            missing_count += 1
            if len(missing) < max_examples:
                missing.append(
                    {
                        "filename_lr": filename_lr,
                        "hea_exists": bool(hea_exists),
                        "dat_exists": bool(dat_exists),
                    }
                )
    return missing_count, missing


def run_check(root, out_dir):
    report = init_report(root)
    report["records500"]["exists"] = (root / "records500").exists()

    for name in REQUIRED_FILES:
        report["required_files"][name] = (root / name).exists()
    for name in OPTIONAL_FILES:
        report["optional_files"][name] = (root / name).exists()

    records100 = root / "records100"
    n_hea, n_dat = count_files(records100)
    report["records100"]["n_lr_hea"] = int(n_hea)
    report["records100"]["n_lr_dat"] = int(n_dat)

    db = None
    try:
        import pandas as pd  # type: ignore

        report["dependencies"]["pandas_available"] = True
    except Exception as exc:  # noqa: BLE001
        report["dependencies"]["missing_packages"].append("pandas")
        report["metadata"]["read_error"] = repr(exc)

    try:
        import numpy  # noqa: F401

        report["dependencies"]["numpy_available"] = True
    except Exception as exc:  # noqa: BLE001
        report["dependencies"]["missing_packages"].append("numpy")
        if not report["waveform_read_check"]["error"]:
            report["waveform_read_check"]["error"] = repr(exc)

    if report["dependencies"]["pandas_available"] and (root / "ptbxl_database.csv").exists():
        try:
            import pandas as pd  # type: ignore

            db = pd.read_csv(root / "ptbxl_database.csv")
            report["metadata"]["n_rows"] = int(len(db))
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in db.columns]
            report["metadata"]["missing_columns"] = missing_columns
            report["metadata"]["required_columns_present"] = not missing_columns
            if not missing_columns:
                missing_count, missing_examples = check_filename_pairs(root, db)
                report["records100"]["missing_waveform_pairs"] = int(missing_count)
                report["records100"]["missing_examples"] = missing_examples
        except Exception as exc:  # noqa: BLE001
            report["metadata"]["read_error"] = repr(exc)

    try:
        import wfdb  # type: ignore

        report["dependencies"]["wfdb_available"] = True
        report["waveform_read_check"]["wfdb_available"] = True
    except Exception as exc:  # noqa: BLE001
        report["dependencies"]["missing_packages"].append("wfdb")
        report["waveform_read_check"]["error"] = repr(exc)

    if db is not None and report["waveform_read_check"]["wfdb_available"] and len(db) > 0:
        example = str(db.iloc[0]["filename_lr"])
        report["waveform_read_check"]["example_filename_lr"] = example
        try:
            import wfdb  # type: ignore

            sig, fields = wfdb.rdsamp(str(root / example))
            shape = list(sig.shape)
            fs = int(fields.get("fs", -1))
            lead_names = [canonical_lead_name(x) for x in fields.get("sig_name", [])]
            units = list(fields.get("units", [])) if fields.get("units", []) is not None else []
            report["waveform_read_check"].update(
                {
                    "shape": shape,
                    "fs": fs,
                    "lead_names": lead_names,
                    "units": units,
                    "pass": bool(tuple(shape) == (1000, 12) and fs == 100 and len(lead_names) == 12),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            report["waveform_read_check"]["error"] = repr(exc)

    required_ok = all(report["required_files"].values())
    metadata_ok = (
        report["metadata"]["n_rows"] == EXPECTED_ROWS
        and report["metadata"]["required_columns_present"] is True
    )
    records_ok = (
        report["records100"]["n_lr_hea"] == EXPECTED_LR_HEA
        and report["records100"]["n_lr_dat"] == EXPECTED_LR_DAT
        and report["records100"]["missing_waveform_pairs"] == 0
    )
    waveform_ok = report["waveform_read_check"]["pass"] is True
    report["overall_pass"] = bool(required_ok and metadata_ok and records_ok and waveform_ok)
    write_reports(report, out_dir)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Check local PTB-XL v1.0.3 records100 download.")
    parser.add_argument("--root", type=Path, default=Path("data/ptb-xl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/day1_audit"))
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_check(args.root, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
