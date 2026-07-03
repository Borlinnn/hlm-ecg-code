#!/usr/bin/env python3
"""Build a Week 7 external ECG manifest from real data only.

When external data are absent, this script reports schema/readiness status and
does not create fake manifests.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from check_week7_external_data_readiness import (
    CANONICAL_LEADS,
    EXCLUDED_SOURCES,
    EXPECTED_SOURCES,
    parse_header,
    source_dirs,
)

MANIFEST_COLUMNS = [
    "record_id",
    "source",
    "relative_header_path",
    "relative_waveform_path",
    "sampling_rate",
    "n_samples",
    "n_leads",
    "lead_names",
    "age",
    "sex",
    "snomed_dx_codes",
    "include_external_primary",
    "exclusion_reason",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def rel(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def exclusion_reasons(source, header, min_seconds=10.0):
    reasons = []
    if source in EXCLUDED_SOURCES:
        reasons.append("excluded_ptb_or_ptbxl_source")
    if header.parse_error:
        reasons.append(f"header_parse_error:{header.parse_error}")
    if not header.waveform_mat_exists:
        reasons.append("missing_waveform_mat")
    if tuple(header.lead_names or []) != CANONICAL_LEADS:
        reasons.append("missing_or_noncanonical_12_leads")
    if header.sampling_rate is None:
        reasons.append("missing_sampling_rate")
    if header.n_samples is None:
        reasons.append("missing_n_samples")
    if header.sampling_rate and header.n_samples is not None:
        if float(header.n_samples) < min_seconds * float(header.sampling_rate):
            reasons.append("too_short_for_safe_10_second_crop")
    if not header.has_dx:
        reasons.append("missing_snomed_dx")
    return reasons


def build_rows(root):
    raw_dir = root / "raw"
    rows = []
    source_summary = {}
    for source, source_dir in source_dirs(raw_dir).items():
        headers = sorted(source_dir.rglob("*.hea")) if source_dir.exists() else []
        source_summary[source] = {
            "source": source,
            "expected_source": source in EXPECTED_SOURCES,
            "excluded_source": source in EXCLUDED_SOURCES,
            "source_dir": str(source_dir),
            "dir_exists": source_dir.exists(),
            "n_headers": len(headers),
            "n_included_primary": 0,
            "n_excluded": 0,
            "records500_used": False,
        }
        for header_path in headers:
            header = parse_header(header_path)
            reasons = exclusion_reasons(source, header)
            include = not reasons
            if include:
                source_summary[source]["n_included_primary"] += 1
            else:
                source_summary[source]["n_excluded"] += 1
            rows.append(
                {
                    "record_id": header.record_id or header_path.stem,
                    "source": source,
                    "relative_header_path": rel(header_path, root),
                    "relative_waveform_path": rel(header.waveform_mat_path, root) if header.waveform_mat_path else "",
                    "sampling_rate": "" if header.sampling_rate is None else header.sampling_rate,
                    "n_samples": "" if header.n_samples is None else header.n_samples,
                    "n_leads": "" if header.n_leads is None else header.n_leads,
                    "lead_names": "|".join(header.lead_names or []),
                    "age": header.age,
                    "sex": header.sex,
                    "snomed_dx_codes": ",".join(header.snomed_dx_codes or []),
                    "include_external_primary": str(include).lower(),
                    "exclusion_reason": ";".join(reasons),
                }
            )
    return rows, list(source_summary.values())


def write_audit(path, root, rows, source_rows, wrote_manifest):
    included = sum(1 for row in rows if row["include_external_primary"] == "true")
    lines = [
        "# Week 7 External Preprocessing Audit",
        "",
        f"- Created at: `{utc_now()}`",
        f"- Root: `{root}`",
        f"- Wrote manifest: `{wrote_manifest}`",
        f"- Total records scanned: `{len(rows)}`",
        f"- Primary included records: `{included}`",
        f"- records500 used: `False`",
        f"- Runs model evaluation: `False`",
        "",
        "## Source Summary",
        "",
        "| source | dir_exists | n_headers | n_included_primary | n_excluded | excluded_source |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            "| {source} | {dir_exists} | {n_headers} | {n_included_primary} | {n_excluded} | {excluded_source} |".format(**row)
        )
    if not rows:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "No real external headers were found. No external manifest was generated.",
            ]
        )
    elif not wrote_manifest:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "Dry-run completed. Review rows, label mapping, and exclusions before running with `--write-manifest`.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "Manifest files were written from real external headers. This is still not an evaluation result.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or dry-run a Week 7 external ECG manifest.")
    parser.add_argument("--root", type=Path, default=Path("data/external/challenge2021"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/week7_external_generalization"))
    parser.add_argument("--dry-run", action="store_true", help="Validate schema and write audit only; do not write manifest CSVs.")
    parser.add_argument("--write-manifest", action="store_true", help="Write manifest CSVs when real data are present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, source_rows = build_rows(args.root)
    has_real_headers = bool(rows)
    should_write = bool(args.write_manifest and has_real_headers and not args.dry_run)

    run_config = {
        "created_at": utc_now(),
        "root": str(args.root),
        "out_dir": str(out_dir),
        "dry_run": bool(args.dry_run),
        "write_manifest_requested": bool(args.write_manifest),
        "wrote_manifest": should_write,
        "n_records_scanned": len(rows),
        "manifest_columns": MANIFEST_COLUMNS,
        "records500_used": False,
        "runs_model_evaluation": False,
    }
    (out_dir / "external_manifest_build_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    write_audit(out_dir / "external_preprocessing_audit.md", root=args.root, rows=rows, source_rows=source_rows, wrote_manifest=should_write)

    if should_write:
        write_csv(out_dir / "external_dataset_manifest.csv", rows, MANIFEST_COLUMNS)
        write_csv(
            out_dir / "external_source_summary.csv",
            source_rows,
            [
                "source",
                "expected_source",
                "excluded_source",
                "source_dir",
                "dir_exists",
                "n_headers",
                "n_included_primary",
                "n_excluded",
                "records500_used",
            ],
        )
        print(f"[week7_manifest] wrote manifest files under {out_dir}")
    elif not has_real_headers:
        print("[week7_manifest] no external headers found; no fake manifest generated")
    else:
        print("[week7_manifest] dry-run/schema validation completed; no manifest CSVs written")


if __name__ == "__main__":
    main()
