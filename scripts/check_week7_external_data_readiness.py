#!/usr/bin/env python3
"""Readiness checks for Week 7 external ECG data intake.

This script parses directory structure and WFDB header metadata only. It does
not load checkpoints, run inference, tune thresholds, or compute metrics.
"""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
EXPECTED_SOURCES = (
    "cpsc_2018",
    "cpsc_2018_extra",
    "st_petersburg_incart",
    "georgia",
    "chapman-shaoxing",
    "ningbo",
)
EXCLUDED_SOURCES = ("ptb", "ptb-xl")


class HeaderSummary:
    def __init__(self, header_path):
        self.header_path = str(header_path)
        self.record_id = ""
        self.sampling_rate = None
        self.n_samples = None
        self.n_leads = None
        self.lead_names = []
        self.has_canonical_12_leads = False
        self.age = ""
        self.sex = ""
        self.snomed_dx_codes = []
        self.has_dx = False
        self.waveform_mat_exists = False
        self.waveform_mat_path = ""
        self.parse_error = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_lead_name(name: object) -> str:
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


def parse_number_token(token):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", token)
    if not match:
        return None
    return float(match.group(0))


def parse_int_token(token):
    value = parse_number_token(token)
    if value is None:
        return None
    return int(value)


def parse_comment_value(line, key):
    prefix = f"#{key}:"
    if line.lower().startswith(prefix.lower()):
        return line.split(":", 1)[1].strip()
    return ""


def parse_header(header_path):
    summary = HeaderSummary(header_path=str(header_path))
    mat_path = header_path.with_suffix(".mat")
    summary.waveform_mat_exists = mat_path.exists()
    summary.waveform_mat_path = str(mat_path)

    try:
        lines = header_path.read_text(errors="replace").splitlines()
        if not lines:
            summary.parse_error = "empty_header"
            return summary

        first = lines[0].strip().split()
        if len(first) < 4:
            summary.parse_error = "first_line_has_fewer_than_4_fields"
            return summary

        summary.record_id = first[0]
        summary.n_leads = parse_int_token(first[1])
        summary.sampling_rate = parse_number_token(first[2])
        summary.n_samples = parse_int_token(first[3])
        if summary.n_leads is None:
            summary.parse_error = "cannot_parse_n_leads"
            return summary

        signal_lines = []
        comments = []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comments.append(stripped)
                continue
            if len(signal_lines) < int(summary.n_leads):
                signal_lines.append(stripped)

        lead_names = []
        for line in signal_lines:
            parts = line.split()
            if parts:
                lead_names.append(canonicalize_lead_name(parts[-1]))
        summary.lead_names = lead_names
        summary.has_canonical_12_leads = tuple(lead_names) == CANONICAL_LEADS

        dx = ""
        for line in comments:
            summary.age = parse_comment_value(line, "Age") or summary.age
            summary.sex = parse_comment_value(line, "Sex") or summary.sex
            dx = parse_comment_value(line, "Dx") or dx
        codes = [code.strip() for code in dx.split(",") if code.strip()]
        summary.snomed_dx_codes = codes
        summary.has_dx = bool(codes)
        return summary
    except Exception as exc:  # pragma: no cover - defensive report path
        summary.parse_error = repr(exc)
        return summary


def source_dirs(raw_dir):
    sources = {name: raw_dir / name for name in EXPECTED_SOURCES}
    for name in EXCLUDED_SOURCES:
        sources[name] = raw_dir / name
    if raw_dir.exists():
        for path in sorted(raw_dir.iterdir()):
            if path.is_dir() and path.name not in sources:
                sources[path.name] = path
    return sources


def iter_headers(source_dir):
    if not source_dir.exists():
        return []
    return sorted(source_dir.rglob("*.hea"))


def summarize_source(source, path, max_parse=None):
    headers = list(iter_headers(path))
    parse_headers = headers if max_parse is None else headers[:max_parse]
    parsed = [parse_header(header) for header in parse_headers]
    mat_count = sum(1 for header in headers if header.with_suffix(".mat").exists())
    parse_errors = sum(1 for item in parsed if item.parse_error)
    canonical_count = sum(1 for item in parsed if item.has_canonical_12_leads)
    dx_count = sum(1 for item in parsed if item.has_dx)
    first = parsed[0] if parsed else HeaderSummary(header_path="")
    return {
        "source": source,
        "expected_source": source in EXPECTED_SOURCES,
        "excluded_source": source in EXCLUDED_SOURCES,
        "source_dir": str(path),
        "dir_exists": path.exists(),
        "hea_count": len(headers),
        "mat_count": mat_count,
        "headers_parsed": len(parsed),
        "parse_errors": parse_errors,
        "canonical_12_lead_headers": canonical_count,
        "headers_with_dx": dx_count,
        "first_record_id": first.record_id,
        "first_sampling_rate": first.sampling_rate,
        "first_n_samples": first.n_samples,
        "first_n_leads": first.n_leads,
        "first_lead_names": "|".join(first.lead_names or []),
        "first_has_dx": first.has_dx,
        "first_parse_error": first.parse_error,
    }


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown_report(path, report, rows, columns):
    lines = [
        "# Week 7 External Data Readiness Report",
        "",
        f"- Created at: `{report['created_at']}`",
        f"- Root: `{report['root']}`",
        f"- Raw directory: `{report['raw_dir']}`",
        f"- Root exists: `{report['root_exists']}`",
        f"- Raw directory exists: `{report['raw_dir_exists']}`",
        f"- Readiness status: `{report['readiness_status']}`",
        f"- This script does not load models or compute metrics.",
        "",
        "## Source Counts",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["interpretation"]),
            "",
            "## Next Step",
            "",
            str(report["next_step"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Week 7 external ECG data readiness without running evaluation.")
    parser.add_argument("--root", type=Path, default=Path("data/external/challenge2021"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/week7_external_generalization/data_intake_preparation"))
    parser.add_argument(
        "--max-parse-per-source",
        type=int,
        default=0,
        help="Maximum headers to parse per source; 0 means parse all headers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    raw_dir = root / "raw"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    max_parse = None if args.max_parse_per_source == 0 else max(0, int(args.max_parse_per_source))

    rows = [summarize_source(source, path, max_parse=max_parse) for source, path in source_dirs(raw_dir).items()]
    included_rows = [row for row in rows if not row["excluded_source"]]
    included_hea = sum(int(row["hea_count"]) for row in included_rows)
    included_canonical = sum(int(row["canonical_12_lead_headers"]) for row in included_rows)
    included_dx = sum(int(row["headers_with_dx"]) for row in included_rows)
    missing_expected = [row["source"] for row in included_rows if row["expected_source"] and not row["dir_exists"]]

    if not root.exists() or not raw_dir.exists():
        status = "missing_external_data_root"
        interpretation = "No usable external ECG data are present yet. Do not run external evaluation."
        next_step = "Place PhysioNet/CinC Challenge 2021 public training data under data/external/challenge2021/raw/."
    elif included_hea == 0:
        status = "no_included_headers_found"
        interpretation = "External raw directory exists, but no included-source .hea files were found."
        next_step = "Check source folder names and WFDB files before attempting manifest generation."
    elif included_canonical == 0 or included_dx == 0:
        status = "metadata_incomplete"
        interpretation = "Some external headers exist, but canonical 12-lead or #Dx metadata are not ready for evaluation."
        next_step = "Review source folders, lead names, waveform files, and SNOMED #Dx labels."
    else:
        status = "potentially_ready_for_manifest_review"
        interpretation = "External headers were found. Review source counts, exclusions, and label mapping before evaluation."
        next_step = "Run build_week7_external_manifest.py in dry-run mode and manually review the label mapping."

    report = {
        "created_at": utc_now(),
        "root": str(root),
        "raw_dir": str(raw_dir),
        "root_exists": root.exists(),
        "raw_dir_exists": raw_dir.exists(),
        "expected_sources": list(EXPECTED_SOURCES),
        "excluded_sources": list(EXCLUDED_SOURCES),
        "missing_expected_sources": missing_expected,
        "readiness_status": status,
        "interpretation": interpretation,
        "next_step": next_step,
        "records500_used": False,
        "runs_evaluation": False,
    }
    json_path = out_dir / "external_data_readiness_report.json"
    json_path.write_text(json.dumps({**report, "sources": rows}, indent=2), encoding="utf-8")
    columns = [
        "source",
        "expected_source",
        "excluded_source",
        "dir_exists",
        "hea_count",
        "mat_count",
        "headers_parsed",
        "parse_errors",
        "canonical_12_lead_headers",
        "headers_with_dx",
        "first_sampling_rate",
        "first_n_samples",
        "first_n_leads",
        "first_lead_names",
    ]
    write_csv(out_dir / "external_data_source_counts.csv", rows, columns)
    write_markdown_report(out_dir / "external_data_readiness_report.md", report, rows, columns)
    print(f"[week7_readiness] wrote {json_path}")
    print(f"[week7_readiness] status={status}")


if __name__ == "__main__":
    main()
