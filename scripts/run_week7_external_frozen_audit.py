#!/usr/bin/env python3
"""Frozen external-source audit runner for Challenge 2021 ECG data.

This script is evaluation-only. It refuses to run unless the user explicitly
passes no-training, no-tuning, and no-calibration flags. The external audit uses
reviewed SNOMED mappings and available classes only: MI, STTC, CD, HYP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hlm_ecg.data.waveforms import CANONICAL_LEADS, canonicalize_leads
from hlm_ecg.evaluation.missing_patterns import MissingPattern, required_patterns
from hlm_ecg.evaluation.supplemental_patterns import (
    KVisibleRandomPattern,
    challenge_reduced_lead_patterns,
    k_visible_random_patterns,
)
from hlm_ecg.models.resnet1d import ResNet1D
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability


INTERNAL_LABEL_ORDER = ("NORM", "MI", "STTC", "CD", "HYP")
AVAILABLE_CLASS_ORDER = ("MI", "STTC", "CD", "HYP")
INTERNAL_CLASS_INDEX = {label: idx for idx, label in enumerate(INTERNAL_LABEL_ORDER)}
ALLOWED_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")
FORBIDDEN_SOURCES = {"ptb-xl", "ptb", "st_petersburg_incart"}
NOT_APPROVED_SOURCES = {"chapman_shaoxing", "chapman-shaoxing", "ningbo"}
DEFAULT_METHODS = ("A1_random_dropout", "A4a_subclass_auxiliary", "A2_structured_masking")
DEFAULT_SEEDS = (7, 42, 123)
DEFAULT_PATTERNS = (
    "full_12",
    "challenge_6_limb",
    "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2",
    "challenge_2_I_II",
    "hard_structured_avg",
    "hard_overall_avg",
)
PRIMARY_COMPARISONS = (
    ("A4a_minus_A1", "A4a_subclass_auxiliary", "A1_random_dropout"),
    ("A4a_minus_A2", "A4a_subclass_auxiliary", "A2_structured_masking"),
    ("A5_lite_minus_A4a", "A5_lite_confidence_consistency_0p05", "A4a_subclass_auxiliary"),
)
EXPECTED_OUTPUT_FILES = (
    "external_frozen_manifest_used.csv",
    "external_frozen_results_by_source.csv",
    "external_frozen_results_pooled.csv",
    "external_frozen_delta_by_source.csv",
    "external_frozen_delta_pooled.csv",
    "external_frozen_bootstrap_ci_by_source.csv",
    "external_frozen_bootstrap_ci_pooled.csv",
    "external_frozen_perclass_delta.csv",
    "external_frozen_seed_summary.csv",
    "external_frozen_audit_report.md",
    "external_frozen_audit_report.json",
    "external_frozen_audit_interpretation.md",
)
STRUCTURED_COMPONENTS = (
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
HARD_OVERALL_COMPONENTS = ("random-6", *STRUCTURED_COMPONENTS)
AGGREGATE_PATTERNS = {
    "hard_structured_avg": STRUCTURED_COMPONENTS,
    "hard_overall_avg": HARD_OVERALL_COMPONENTS,
}
FORBIDDEN_PATH_TOKENS = ("records" + "500", "filename" + "_" + "hr")


@dataclass(frozen=True)
class MethodRun:
    method_id: str
    seed: int
    output_dir: Path
    checkpoint_path: Path
    config_path: Path
    thresholds_path: Path

    @property
    def method_run_id(self) -> str:
        return f"{self.method_id}_seed{self.seed}"


@dataclass(frozen=True)
class MappingRow:
    snomed_code: str
    superclass: str
    confidence: str
    include_in_eval: bool
    name: str


@dataclass(frozen=True)
class HeaderInfo:
    header_path: Path
    mat_path: Path
    source: str
    record_id: str
    fs: float | None
    n_samples: int | None
    lead_names: tuple[str, ...]
    dx_codes: tuple[str, ...]
    parse_error: str

    @property
    def has_pair(self) -> bool:
        return self.mat_path.exists()

    @property
    def has_all_standard_leads(self) -> bool:
        return set(CANONICAL_LEADS).issubset(set(self.lead_names))

    @property
    def duration_seconds(self) -> float | None:
        if self.fs is None or self.n_samples is None or self.fs <= 0:
            return None
        return float(self.n_samples) / float(self.fs)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"PyYAML is required only if checkpoint config is missing: {exc}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}")
    return data


def make_method_run(method_id: str, seed: int, output_dir: Path) -> MethodRun | None:
    checkpoint = output_dir / "best_model.pt"
    config = output_dir / "config_used.yaml"
    thresholds = output_dir / "thresholds_val.json"
    if not checkpoint.exists() or not config.exists() or not thresholds.exists():
        return None
    return MethodRun(
        method_id=method_id,
        seed=int(seed),
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        config_path=config,
        thresholds_path=thresholds,
    )


def discover_method_runs(method_ids: Sequence[str]) -> list[MethodRun]:
    requested = set(method_ids)
    runs: list[MethodRun] = []
    seen: set[tuple[str, int]] = set()
    manifest_path = REPO_ROOT / "outputs/week3_multiseed_summary/multiseed_manifest.csv"
    for row in read_csv_rows(manifest_path):
        method_id = row.get("method_id", "")
        if method_id not in requested:
            continue
        run = make_method_run(method_id, int(row["seed"]), REPO_ROOT / row["output_dir"])
        if run is not None:
            runs.append(run)
            seen.add((run.method_id, run.seed))
    lock_path = REPO_ROOT / "outputs/week3_results_lock/all_methods_summary.csv"
    for row in read_csv_rows(lock_path):
        method_id = row.get("method_id", "")
        if method_id not in requested:
            continue
        if (method_id, 42) in seen:
            continue
        run = make_method_run(method_id, 42, REPO_ROOT / row["output_dir"])
        if run is not None:
            runs.append(run)
            seen.add((run.method_id, run.seed))
    runs.sort(key=lambda item: (item.method_id, item.seed))
    required = {"A1_random_dropout", "A2_structured_masking", "A4a_subclass_auxiliary"}.intersection(requested)
    present = {run.method_id for run in runs}
    missing = sorted(required.difference(present))
    if missing:
        raise FileNotFoundError(f"Missing required method checkpoints/configs/thresholds for: {missing}")
    return runs


def assert_no_forbidden_path_tokens_in_runs(runs: Sequence[MethodRun]) -> None:
    bad: list[str] = []
    for run in runs:
        for path in (run.output_dir, run.config_path, run.checkpoint_path, run.thresholds_path):
            if any(token in str(path) for token in FORBIDDEN_PATH_TOKENS):
                bad.append(str(path))
        text = run.config_path.read_text(encoding="utf-8")
        if any(token in text for token in FORBIDDEN_PATH_TOKENS):
            bad.append(str(run.config_path))
    if bad:
        raise RuntimeError(f"Forbidden high-resolution path token detected in method sources: {bad}")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt_float(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.6f}"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr))


def ensure_not_locked_output_dir(path: Path) -> None:
    resolved = path.resolve()
    locked = (REPO_ROOT / "outputs").resolve()
    if resolved == locked or locked in resolved.parents:
        raise RuntimeError(f"Refusing to write audit outputs under locked output tree: {path}")


def ensure_no_forbidden_path_tokens(paths: Iterable[Path | str]) -> None:
    bad: list[str] = []
    for item in paths:
        text = str(item)
        if any(token in text for token in FORBIDDEN_PATH_TOKENS):
            bad.append(text)
    if bad:
        raise RuntimeError(f"Forbidden high-resolution path token detected: {bad}")


def parse_number_token(token: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", token)
    if not match:
        return None
    return float(match.group(0))


def parse_int_token(token: str) -> int | None:
    value = parse_number_token(token)
    if value is None:
        return None
    return int(value)


def parse_comment_value(line: str, key: str) -> str:
    text = line.strip()
    if text.startswith("#"):
        text = text[1:].strip()
    prefix = f"{key}:"
    if text.lower().startswith(prefix.lower()):
        return text.split(":", 1)[1].strip()
    return ""


def parse_header(header_path: Path, source: str) -> HeaderInfo:
    mat_path = header_path.with_suffix(".mat")
    try:
        lines = header_path.read_text(errors="replace").splitlines()
        if not lines:
            return HeaderInfo(header_path, mat_path, source, header_path.stem, None, None, (), (), "empty_header")
        first = lines[0].strip().split()
        if len(first) < 4:
            return HeaderInfo(header_path, mat_path, source, header_path.stem, None, None, (), (), "bad_first_line")
        record_id = first[0]
        n_leads = parse_int_token(first[1])
        fs = parse_number_token(first[2])
        n_samples = parse_int_token(first[3])
        if n_leads is None:
            return HeaderInfo(header_path, mat_path, source, record_id, fs, n_samples, (), (), "bad_lead_count")

        signal_lines: list[str] = []
        comments: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comments.append(stripped)
                continue
            if len(signal_lines) < int(n_leads):
                signal_lines.append(stripped)

        lead_names = []
        for line in signal_lines:
            parts = line.split()
            if parts:
                lead_names.append(parts[-1])
        canonical = canonicalize_leads(lead_names)

        dx = ""
        for line in comments:
            dx = parse_comment_value(line, "Dx") or dx
        codes = tuple(code.strip() for code in dx.split(",") if code.strip())
        return HeaderInfo(header_path, mat_path, source, record_id, fs, n_samples, canonical, codes, "")
    except Exception as exc:  # pragma: no cover - defensive path
        return HeaderInfo(header_path, mat_path, source, header_path.stem, None, None, (), (), repr(exc))


def parse_signal_metadata(header_path: Path) -> dict[str, Any]:
    lines = header_path.read_text(errors="replace").splitlines()
    first = lines[0].strip().split()
    if len(first) < 4:
        raise RuntimeError(f"Bad WFDB first line: {header_path}")
    n_leads = int(parse_int_token(first[1]) or 0)
    fs = float(parse_number_token(first[2]) or 0.0)
    n_samples = int(parse_int_token(first[3]) or 0)
    signal_lines: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(signal_lines) < n_leads:
            signal_lines.append(stripped)
    if len(signal_lines) != n_leads:
        raise RuntimeError(f"Header signal line count mismatch: {header_path}")

    leads: list[str] = []
    gains: list[float] = []
    baselines: list[float] = []
    offsets: list[int] = []
    formats: list[str] = []
    for line in signal_lines:
        parts = line.split()
        if len(parts) < 3:
            raise RuntimeError(f"Bad signal line in {header_path}: {line}")
        fmt = parts[1]
        formats.append(fmt)
        offset_match = re.search(r"\+(\d+)", fmt)
        offsets.append(int(offset_match.group(1)) if offset_match else 0)
        if not fmt.startswith("16"):
            raise RuntimeError(f"Only 16-bit WFDB Challenge .mat fallback is supported, got {fmt}")
        gain_token = parts[2]
        gain = parse_number_token(gain_token)
        if gain is None or gain <= 0:
            raise RuntimeError(f"Cannot parse gain from {gain_token} in {header_path}")
        baseline_match = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", gain_token)
        baseline = float(baseline_match.group(1)) if baseline_match else 0.0
        gains.append(float(gain))
        baselines.append(float(baseline))
        leads.append(parts[-1])
    return {
        "n_leads": n_leads,
        "fs": fs,
        "n_samples": n_samples,
        "lead_names": canonicalize_leads(leads),
        "gains": np.asarray(gains, dtype=np.float32),
        "baselines": np.asarray(baselines, dtype=np.float32),
        "byte_offset": max(offsets) if offsets else 0,
        "formats": formats,
    }


def read_external_waveform(header_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import wfdb  # type: ignore

        sig, fields = wfdb.rdsamp(str(header_path.with_suffix("")))
        return sig.astype(np.float32, copy=False), dict(fields)
    except ModuleNotFoundError:
        pass
    meta = parse_signal_metadata(header_path)
    n_samples = int(meta["n_samples"])
    n_leads = int(meta["n_leads"])
    mat_path = header_path.with_suffix(".mat")
    count = n_samples * n_leads
    with mat_path.open("rb") as f:
        f.seek(int(meta["byte_offset"]))
        data = np.fromfile(f, dtype="<i2", count=count)
    if data.size != count:
        raise RuntimeError(f"WFDB binary size mismatch for {mat_path}: {data.size} != {count}")
    digital = data.reshape(n_samples, n_leads).astype(np.float32)
    signal = (digital - meta["baselines"].reshape(1, n_leads)) / meta["gains"].reshape(1, n_leads)
    fields = {"fs": meta["fs"], "sig_name": list(meta["lead_names"])}
    return signal.astype(np.float32, copy=False), fields


def load_reviewed_mapping(path: Path) -> dict[str, MappingRow]:
    rows = read_csv_rows(path)
    required = {
        "snomed_code",
        "snomed_name",
        "final_ptbxl_superclass",
        "final_mapping_confidence",
        "final_include_in_eval",
    }
    if not rows:
        raise RuntimeError(f"Reviewed mapping is empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Reviewed mapping missing columns: {sorted(missing)}")
    if len(rows) != 90:
        raise RuntimeError(f"Reviewed mapping must have 90 rows, found {len(rows)}")
    allowed_superclasses = {"NORM", "MI", "STTC", "CD", "HYP", "NONE", "UNKNOWN"}
    allowed_conf = {"exact", "clinically_reasonable", "uncertain", "exclude"}
    mapping: dict[str, MappingRow] = {}
    for row in rows:
        code = str(row["snomed_code"]).strip()
        if not code:
            raise RuntimeError("Reviewed mapping has an empty snomed_code")
        if code in mapping:
            raise RuntimeError(f"Duplicate snomed_code in reviewed mapping: {code}")
        superclass = str(row["final_ptbxl_superclass"]).strip()
        confidence = str(row["final_mapping_confidence"]).strip()
        include = truthy(row["final_include_in_eval"])
        if superclass not in allowed_superclasses:
            raise RuntimeError(f"Invalid final_ptbxl_superclass for {code}: {superclass}")
        if confidence not in allowed_conf:
            raise RuntimeError(f"Invalid final_mapping_confidence for {code}: {confidence}")
        if include and (superclass not in AVAILABLE_CLASS_ORDER or confidence not in {"exact", "clinically_reasonable"}):
            raise RuntimeError(
                "final_include_in_eval=true is only allowed for MI/STTC/CD/HYP "
                f"with exact or clinically_reasonable confidence; failed for {code}"
            )
        if include and superclass == "NORM":
            raise RuntimeError(f"NORM include_in_eval is disallowed for external audit; failed for {code}")
        mapping[code] = MappingRow(
            snomed_code=code,
            superclass=superclass,
            confidence=confidence,
            include_in_eval=include,
            name=str(row.get("snomed_name", "")).strip() or "UNKNOWN",
        )
    sinus = mapping.get("426783006")
    if sinus is None:
        raise RuntimeError("Reviewed mapping missing sinus rhythm code 426783006")
    if sinus.include_in_eval:
        raise RuntimeError("sinus rhythm code 426783006 must have final_include_in_eval=false")
    return mapping


def validate_classes(classes: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(item).strip() for item in classes if str(item).strip())
    if not selected:
        raise RuntimeError("At least one external class is required")
    if "NORM" in selected:
        raise RuntimeError("NORM is unavailable in the reviewed mapping and is disallowed for external metrics")
    unknown = [label for label in selected if label not in AVAILABLE_CLASS_ORDER]
    if unknown:
        raise RuntimeError(f"External classes must be a subset of {AVAILABLE_CLASS_ORDER}, got {unknown}")
    if len(set(selected)) != len(selected):
        raise RuntimeError(f"Duplicate external classes requested: {selected}")
    order = [AVAILABLE_CLASS_ORDER.index(label) for label in selected]
    if order != sorted(order):
        raise RuntimeError(f"External classes must follow this order: {AVAILABLE_CLASS_ORDER}")
    return selected


def validate_sources(sources: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(item).strip() for item in sources if str(item).strip())
    if not selected:
        raise RuntimeError("At least one source is required")
    forbidden = sorted(set(selected).intersection(FORBIDDEN_SOURCES))
    if forbidden:
        raise RuntimeError(f"Forbidden external sources requested: {forbidden}")
    not_approved = sorted(set(selected).intersection(NOT_APPROVED_SOURCES))
    if not_approved:
        raise RuntimeError(f"These optional sources are not approved in this task: {not_approved}")
    unknown = sorted(set(selected).difference(ALLOWED_SOURCES))
    if unknown:
        raise RuntimeError(f"Unknown or unapproved sources requested: {unknown}")
    return selected


def validate_required_flags(args: argparse.Namespace) -> None:
    missing = []
    if not args.no_train:
        missing.append("--no-train")
    if not args.no_tune:
        missing.append("--no-tune")
    if not args.no_calibration:
        missing.append("--no-calibration")
    if missing:
        raise RuntimeError(f"Refusing to run without required safety flags: {', '.join(missing)}")
    if args.bootstrap_level != "record":
        raise RuntimeError("Only record-level bootstrap is available for current external sources")
    if args.dry_run and args.smoke_test:
        raise RuntimeError("Use either --dry-run or --smoke-test, not both")
    if not args.dry_run and args.max_records_per_source is not None and not args.smoke_test:
        raise RuntimeError("--max-records-per-source is only allowed with --smoke-test or --dry-run")


def source_dir(root: Path, source: str) -> Path:
    raw = root / "raw"
    if raw.exists():
        return raw / source
    return root / source


def label_record(
    dx_codes: Sequence[str],
    mapping: Mapping[str, MappingRow],
    classes: Sequence[str],
) -> dict[str, Any]:
    y = {label: 0 for label in classes}
    mapped_positive: list[str] = []
    excluded_codes: list[str] = []
    unknown_codes: list[str] = []
    for code in dx_codes:
        item = mapping.get(str(code))
        if item is None:
            unknown_codes.append(str(code))
            continue
        if item.include_in_eval and item.superclass in classes:
            y[item.superclass] = 1
            mapped_positive.append(item.superclass)
        elif item.superclass in {"UNKNOWN"} or item.confidence == "uncertain":
            unknown_codes.append(str(code))
        else:
            excluded_codes.append(str(code))
    positives = sorted(set(mapped_positive), key=lambda label: classes.index(label))
    return {
        "targets": y,
        "mapped_positive_labels": positives,
        "excluded_or_none_codes": sorted(set(excluded_codes)),
        "unknown_or_unmapped_codes": sorted(set(unknown_codes)),
        "has_any_positive": bool(positives),
        "has_unknown_or_unmapped": bool(unknown_codes),
    }


def build_external_manifest(
    *,
    root: Path,
    sources: Sequence[str],
    mapping: Mapping[str, MappingRow],
    classes: Sequence[str],
    max_records_per_source: int | None,
    dry_run: bool,
    smoke_test: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        path = source_dir(root, source)
        if not path.exists():
            raise RuntimeError(f"Source directory not found: {path}")
        selected = 0
        for header_path in sorted(path.rglob("*.hea")):
            info = parse_header(header_path, source)
            labels = label_record(info.dx_codes, mapping, classes)
            eligible = True
            reasons: list[str] = []
            if info.parse_error:
                eligible = False
                reasons.append(f"header_parse_error:{info.parse_error}")
            if not info.has_pair:
                eligible = False
                reasons.append("missing_mat_pair")
            if not info.has_all_standard_leads:
                eligible = False
                reasons.append("missing_standard_lead")
            if not info.dx_codes:
                eligible = False
                reasons.append("missing_dx")
            if not eligible:
                continue
            relative = header_path.relative_to(path)
            record_key = f"{source}/{relative.with_suffix('').as_posix()}"
            rows.append(
                {
                    "source": source,
                    "record_key": record_key,
                    "record_id": info.record_id,
                    "header_path": str(header_path),
                    "waveform_path": str(info.mat_path),
                    "relative_path": str(relative),
                    "fs_original": info.fs,
                    "n_samples_original": info.n_samples,
                    "duration_original": info.duration_seconds,
                    "available_leads": "|".join(info.lead_names),
                    "has_all_12_standard_leads": True,
                    "labels_raw": "|".join(info.dx_codes),
                    "mapped_superclass_labels": "|".join(labels["mapped_positive_labels"]),
                    "no_positive_reviewed_available_class_label": not labels["has_any_positive"],
                    "has_unknown_or_unmapped_label": labels["has_unknown_or_unmapped"],
                    "unknown_or_unmapped_codes": "|".join(labels["unknown_or_unmapped_codes"]),
                    "excluded_or_none_codes": "|".join(labels["excluded_or_none_codes"]),
                    "target_MI": labels["targets"].get("MI", ""),
                    "target_STTC": labels["targets"].get("STTC", ""),
                    "target_CD": labels["targets"].get("CD", ""),
                    "target_HYP": labels["targets"].get("HYP", ""),
                    "patient_id": "",
                    "source_id": source,
                    "eligible_for_future_eval": True,
                    "eligibility_reason": "paired_header_waveform_all12_reviewed_mapping",
                    "dry_run": dry_run,
                    "smoke_test": smoke_test,
                }
            )
            selected += 1
            if max_records_per_source is not None and selected >= int(max_records_per_source):
                break
    if not rows:
        raise RuntimeError("No eligible external records found for requested sources")
    return rows


def build_pattern_registry(include_kvisible: bool) -> dict[str, MissingPattern | KVisibleRandomPattern]:
    registry: dict[str, MissingPattern | KVisibleRandomPattern] = {}
    challenge = challenge_reduced_lead_patterns()
    registry["full_12"] = challenge["challenge_12_all"]
    for name, pattern in challenge.items():
        if name != "challenge_12_all":
            registry[name] = pattern
    base = required_patterns()
    for name in (*STRUCTURED_COMPONENTS, "random-6"):
        registry[name] = base[name]
    if include_kvisible:
        registry.update(k_visible_random_patterns())
    return registry


def expand_patterns(
    requested: Sequence[str],
    registry: Mapping[str, MissingPattern | KVisibleRandomPattern],
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    eval_names: list[str] = []
    aggregates: dict[str, tuple[str, ...]] = {}
    for name in requested:
        if name in AGGREGATE_PATTERNS:
            components = AGGREGATE_PATTERNS[name]
            missing = [component for component in components if component not in registry]
            if missing:
                raise RuntimeError(f"Aggregate pattern {name} missing components: {missing}")
            aggregates[name] = components
            for component in components:
                if component not in eval_names:
                    eval_names.append(component)
        elif name in registry:
            if name not in eval_names:
                eval_names.append(name)
        else:
            raise RuntimeError(f"Unknown pattern requested: {name}")
    return eval_names, aggregates


def load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    leads = tuple(str(x) for x in stats["lead_names"].tolist())
    if leads != CANONICAL_LEADS:
        raise RuntimeError(f"Normalization lead order mismatch: {leads}")
    if mean.shape != (12,) or std.shape != (12,):
        raise RuntimeError("Normalization mean/std must have shape (12,)")
    if np.any(std <= 0):
        raise RuntimeError("Normalization std must be positive")
    return mean, std


def build_model_from_config(config: Mapping[str, Any]) -> torch.nn.Module:
    model_cfg = dict(config.get("model", {}))
    if bool(model_cfg.get("use_availability_embedding", False)):
        return ResNet1DAvailability(
            in_channels=int(model_cfg.get("in_channels", 12)),
            num_classes=int(model_cfg.get("num_classes", 5)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            layers=tuple(model_cfg.get("layers", [1, 1, 1, 1])),
            kernel_size=int(model_cfg.get("kernel_size", 7)),
            availability_embedding_dim=int(model_cfg.get("availability_embedding_dim", 32)),
            mask_mlp_hidden_dim=int(model_cfg.get("mask_mlp_hidden_dim", 32)),
            use_subclass_auxiliary=bool(model_cfg.get("enable_subclass_auxiliary", False)),
            num_subclasses=None if model_cfg.get("num_subclasses") is None else int(model_cfg.get("num_subclasses")),
        )
    return ResNet1D(
        in_channels=int(model_cfg.get("in_channels", 12)),
        num_classes=int(model_cfg.get("num_classes", 5)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        layers=tuple(model_cfg.get("layers", [1, 1, 1, 1])),
        kernel_size=int(model_cfg.get("kernel_size", 7)),
    )


def load_model_from_checkpoint_external(
    checkpoint_path: Path,
    fallback_config: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", fallback_config)
    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError(f"Checkpoint config is not a mapping: {checkpoint_path}")
    model = build_model_from_config(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def resolve_device_external(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def forward_model_external(model: torch.nn.Module, batch: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
    x = batch["x"].to(device=device, dtype=torch.float32)
    if bool(getattr(model, "requires_availability_mask", False)):
        mask = batch.get("availability_mask", batch.get("lead_mask"))
        if mask is None:
            raise RuntimeError("Availability-aware model requires an availability mask")
        output = model(x, availability_mask=mask.to(device=device, dtype=torch.float32))
        if isinstance(output, Mapping):
            return output["logits_super"]
        return output
    return model(x)


def resample_to_100hz(signal: np.ndarray, fs: float) -> np.ndarray:
    if abs(float(fs) - 100.0) < 1e-6:
        return signal.astype(np.float32, copy=False)
    target_len = max(1, int(round(signal.shape[0] * 100.0 / float(fs))))
    try:
        from scipy import signal as scipy_signal  # type: ignore

        if abs(float(fs) - round(float(fs))) < 1e-6:
            fs_i = int(round(float(fs)))
            gcd = math.gcd(fs_i, 100)
            up = 100 // gcd
            down = fs_i // gcd
            return scipy_signal.resample_poly(signal, up=up, down=down, axis=0).astype(np.float32, copy=False)
        return scipy_signal.resample(signal, target_len, axis=0).astype(np.float32, copy=False)
    except Exception:
        old_x = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        channels = [np.interp(new_x, old_x, signal[:, idx]) for idx in range(signal.shape[1])]
        return np.stack(channels, axis=1).astype(np.float32, copy=False)


def crop_or_pad_10s(signal: np.ndarray) -> np.ndarray:
    target = 1000
    if signal.shape[0] >= target:
        return signal[:target, :].astype(np.float32, copy=False)
    out = np.zeros((target, signal.shape[1]), dtype=np.float32)
    out[: signal.shape[0], :] = signal.astype(np.float32, copy=False)
    return out


class ExternalChallengeDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        pattern: MissingPattern | KVisibleRandomPattern,
        classes: Sequence[str],
        norm_stats_path: Path,
    ) -> None:
        self.records = list(records)
        self.pattern = pattern
        self.classes = tuple(classes)
        self.mean, self.std = load_norm_stats(norm_stats_path)
        if not self.records:
            raise RuntimeError("ExternalChallengeDataset received no records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        row = self.records[int(idx)]
        header_path = Path(str(row["header_path"]))
        sig, fields = read_external_waveform(header_path)
        fs = float(fields.get("fs", row.get("fs_original", 0)))
        lead_names = canonicalize_leads(fields.get("sig_name", []))
        if not set(CANONICAL_LEADS).issubset(set(lead_names)):
            raise RuntimeError(f"External record missing standard leads: {row['record_key']} {lead_names}")
        lead_index = {lead: lead_names.index(lead) for lead in CANONICAL_LEADS}
        ordered = sig[:, [lead_index[lead] for lead in CANONICAL_LEADS]].astype(np.float32, copy=False)
        resampled = resample_to_100hz(ordered, fs)
        raw = crop_or_pad_10s(resampled)
        if raw.shape != (1000, 12):
            raise RuntimeError(f"External waveform shape mismatch after crop/pad: {raw.shape}")
        mask = self.pattern.mask_for_index(int(idx), CANONICAL_LEADS).astype(np.float32)
        x = (raw - self.mean.reshape(1, 12)) / self.std.reshape(1, 12)
        if not np.all(mask == 1):
            x = x.copy()
            x[:, mask == 0] = 0.0
        target = np.asarray([int(row[f"target_{label}"]) for label in self.classes], dtype=np.float32)
        return {
            "x": torch.from_numpy(x.T.astype(np.float32, copy=False)),
            "y": torch.from_numpy(target),
            "lead_mask": torch.from_numpy(mask),
            "availability_mask": torch.from_numpy(mask),
            "record_key": str(row["record_key"]),
            "record_id": str(row["record_id"]),
            "source": str(row["source"]),
        }


@torch.no_grad()
def predict_external(
    *,
    run: MethodRun,
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    pattern_name: str,
    pattern: MissingPattern | KVisibleRandomPattern,
    classes: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    norm_stats_path: Path,
) -> dict[str, Any]:
    model = load_model_from_checkpoint_external(run.checkpoint_path, config, device)
    model.eval()
    dataset = ExternalChallengeDataset(records, pattern=pattern, classes=classes, norm_stats_path=norm_stats_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    record_keys: list[str] = []
    sources: list[str] = []
    record_ids: list[str] = []
    select_indices = [INTERNAL_CLASS_INDEX[label] for label in classes]
    for batch in loader:
        logits5 = forward_model_external(model, batch, device=device).detach().cpu().numpy()
        if logits5.shape[1] != len(INTERNAL_LABEL_ORDER):
            raise RuntimeError(f"Expected 5 model outputs in {INTERNAL_LABEL_ORDER}, got {logits5.shape}")
        logits_all.append(logits5[:, select_indices])
        targets_all.append(batch["y"].detach().cpu().numpy())
        record_keys.extend([str(x) for x in batch["record_key"]])
        sources.extend([str(x) for x in batch["source"]])
        record_ids.extend([str(x) for x in batch["record_id"]])
    logits = np.concatenate(logits_all, axis=0)
    targets = np.concatenate(targets_all, axis=0).astype(np.int64)
    probs = sigmoid(logits)
    return {
        "method_id": run.method_id,
        "seed": run.seed,
        "pattern": pattern_name,
        "record_keys": np.asarray(record_keys),
        "record_ids": np.asarray(record_ids),
        "sources": np.asarray(sources),
        "targets": targets,
        "logits": logits,
        "probs": probs,
    }


def compute_auprc_metrics(targets: np.ndarray, probs: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if targets.shape != probs.shape or targets.shape[1] != len(classes):
        raise RuntimeError(f"Metric shape mismatch: {targets.shape} vs {probs.shape} for {classes}")
    per_class: dict[str, float | None] = {}
    warnings: list[str] = []
    for idx, label in enumerate(classes):
        unique = np.unique(targets[:, idx])
        if unique.size < 2:
            per_class[label] = None
            value = int(unique[0]) if unique.size else "empty"
            warnings.append(f"AUPRC undefined for {label}: only {value} present")
        else:
            per_class[label] = average_precision_binary(targets[:, idx], probs[:, idx])
    valid = [value for value in per_class.values() if value is not None]
    macro = float(np.mean(valid)) if valid else None
    return {
        "macro_auprc_available": macro,
        "macro_auprc_n_defined_classes": len(valid),
        "per_class_auprc": per_class,
        "warnings": warnings,
        "positives": {label: int(targets[:, idx].sum()) for idx, label in enumerate(classes)},
    }


def result_row(
    *,
    scope: str,
    source: str,
    run: MethodRun,
    pattern: str,
    metrics: Mapping[str, Any],
    n_records: int,
    classes: Sequence[str],
    dry_run: bool,
    smoke_test: bool,
    aggregate_components: Sequence[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "dry_run" if dry_run else ("smoke_test" if smoke_test else "full_audit"),
        "dry_run": dry_run,
        "smoke_test": smoke_test,
        "scope": scope,
        "source": source,
        "method_id": run.method_id,
        "seed": run.seed,
        "pattern": pattern,
        "n_records": int(n_records),
        "classes": "|".join(classes),
        "macro_auprc_available": metrics.get("macro_auprc_available"),
        "macro_auprc_n_defined_classes": metrics.get("macro_auprc_n_defined_classes"),
        "bootstrap_level": "record",
        "aggregate_components": "|".join(aggregate_components or []),
        "warnings": "|".join(metrics.get("warnings", [])),
    }
    positives = dict(metrics.get("positives", {}))
    per_class = dict(metrics.get("per_class_auprc", {}))
    for label in AVAILABLE_CLASS_ORDER:
        if label in classes:
            row[f"n_positive_{label}"] = positives.get(label)
            row[f"auprc_{label}"] = per_class.get(label)
        else:
            row[f"n_positive_{label}"] = ""
            row[f"auprc_{label}"] = ""
    return row


def subset_prediction(pred: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        **pred,
        "record_keys": np.asarray(pred["record_keys"])[mask],
        "record_ids": np.asarray(pred["record_ids"])[mask],
        "sources": np.asarray(pred["sources"])[mask],
        "targets": np.asarray(pred["targets"])[mask],
        "logits": np.asarray(pred["logits"])[mask],
        "probs": np.asarray(pred["probs"])[mask],
    }


def save_prediction_csv(path: Path, pred: Mapping[str, Any], classes: Sequence[str], run: MethodRun) -> None:
    rows: list[dict[str, Any]] = []
    targets = np.asarray(pred["targets"])
    logits = np.asarray(pred["logits"])
    probs = np.asarray(pred["probs"])
    for idx, record_key in enumerate(np.asarray(pred["record_keys"]).tolist()):
        row: dict[str, Any] = {
            "method_id": run.method_id,
            "seed": run.seed,
            "pattern": pred["pattern"],
            "source": np.asarray(pred["sources"])[idx],
            "record_key": record_key,
            "record_id": np.asarray(pred["record_ids"])[idx],
        }
        for class_idx, label in enumerate(classes):
            row[f"target_{label}"] = int(targets[idx, class_idx])
            row[f"logit_{label}"] = float(logits[idx, class_idx])
            row[f"prob_{label}"] = float(probs[idx, class_idx])
        rows.append(row)
    write_csv(path, rows)


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos <= 0 or n_pos >= y.shape[0]:
        raise ValueError("average precision requires both positive and negative examples")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(y_sorted.shape[0], dtype=np.float64) + 1.0)
    return float(np.sum(precision[y_sorted == 1]) / float(n_pos))


def build_aggregate_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    aggregates: Mapping[str, Sequence[str]],
    classes: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    group_keys = ["status", "dry_run", "smoke_test", "scope", "source", "method_id", "seed"]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    for key, group_rows in grouped.items():
        by_pattern = {str(row["pattern"]): row for row in group_rows}
        base = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        for aggregate_name, components in aggregates.items():
            if not all(component in by_pattern for component in components):
                continue
            component_rows = [by_pattern[component] for component in components]
            values = [row.get("macro_auprc_available") for row in component_rows]
            valid_values = [float(value) for value in values if value not in (None, "")]
            aggregate_row = dict(base)
            aggregate_row.update(
                {
                    "pattern": aggregate_name,
                    "n_records": "",
                    "classes": "|".join(classes),
                    "macro_auprc_available": float(np.mean(valid_values)) if valid_values else None,
                    "macro_auprc_n_defined_classes": "",
                    "bootstrap_level": "record",
                    "aggregate_components": "|".join(components),
                    "warnings": "aggregate_from_component_metric_rows",
                }
            )
            for label in AVAILABLE_CLASS_ORDER:
                per_values = [
                    row.get(f"auprc_{label}")
                    for row in component_rows
                    if row.get(f"auprc_{label}") not in (None, "")
                ]
                aggregate_row[f"auprc_{label}"] = float(np.mean([float(x) for x in per_values])) if per_values else None
                aggregate_row[f"n_positive_{label}"] = ""
            out.append(aggregate_row)
    return out


def align_predictions(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys_a = [str(x) for x in np.asarray(a["record_keys"]).tolist()]
    keys_b = [str(x) for x in np.asarray(b["record_keys"]).tolist()]
    index_b = {key: idx for idx, key in enumerate(keys_b)}
    a_indices: list[int] = []
    b_indices: list[int] = []
    for idx, key in enumerate(keys_a):
        if key in index_b:
            a_indices.append(idx)
            b_indices.append(index_b[key])
    if not a_indices:
        raise RuntimeError("No overlapping record keys for paired comparison")
    targets_a = np.asarray(a["targets"])[a_indices]
    targets_b = np.asarray(b["targets"])[b_indices]
    if not np.array_equal(targets_a, targets_b):
        raise RuntimeError("Targets differ after paired record alignment")
    probs_a = np.asarray(a["probs"])[a_indices]
    probs_b = np.asarray(b["probs"])[b_indices]
    return targets_a, probs_a, probs_b


def delta_for_predictions(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    classes: Sequence[str],
) -> tuple[dict[str, Any], int]:
    targets, probs_a, probs_b = align_predictions(a, b)
    met_a = compute_auprc_metrics(targets, probs_a, classes)
    met_b = compute_auprc_metrics(targets, probs_b, classes)
    per_class = {}
    for label in classes:
        va = met_a["per_class_auprc"].get(label)
        vb = met_b["per_class_auprc"].get(label)
        per_class[label] = None if va is None or vb is None else float(va) - float(vb)
    macro_a = met_a["macro_auprc_available"]
    macro_b = met_b["macro_auprc_available"]
    return {
        "macro_delta": None if macro_a is None or macro_b is None else float(macro_a) - float(macro_b),
        "per_class_delta": per_class,
        "metric_a": met_a,
        "metric_b": met_b,
    }, int(targets.shape[0])


def build_delta_rows(
    predictions: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    *,
    classes: Sequence[str],
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    macro_rows: list[dict[str, Any]] = []
    perclass_rows: list[dict[str, Any]] = []
    for comparison, method_a, method_b in PRIMARY_COMPARISONS:
        keys_a = [key for key in predictions if key[0] == method_a and key[3] == source]
        for key_a in keys_a:
            _, seed, pattern, _source = key_a
            key_b = (method_b, seed, pattern, source)
            if key_b not in predictions:
                continue
            delta, n_aligned = delta_for_predictions(predictions[key_a], predictions[key_b], classes)
            macro_rows.append(
                {
                    "comparison": comparison,
                    "method_a": method_a,
                    "method_b": method_b,
                    "seed": seed,
                    "pattern": pattern,
                    "source": source,
                    "n_aligned_records": n_aligned,
                    "metric": "macro_auprc_available",
                    "delta": delta["macro_delta"],
                }
            )
            for label, value in delta["per_class_delta"].items():
                perclass_rows.append(
                    {
                        "comparison": comparison,
                        "method_a": method_a,
                        "method_b": method_b,
                        "seed": seed,
                        "pattern": pattern,
                        "source": source,
                        "class": label,
                        "metric": "auprc",
                        "delta": value,
                    }
                )
    return macro_rows, perclass_rows


def bootstrap_macro_auprc_from_counts(
    counts: np.ndarray,
    orders: Sequence[np.ndarray],
    y_sorted: Sequence[np.ndarray],
) -> np.ndarray:
    values = np.full((counts.shape[0], len(orders)), np.nan, dtype=np.float64)
    total_draws = counts.sum(axis=1)
    for class_idx, order in enumerate(orders):
        class_counts = counts[:, order].astype(np.float64, copy=False)
        y = y_sorted[class_idx].reshape(1, -1)
        positive_counts = class_counts * y
        total_positive = positive_counts.sum(axis=1)
        valid = (total_positive > 0) & (total_positive < total_draws)
        if not np.any(valid):
            continue
        cumulative_positive = np.cumsum(positive_counts[valid], axis=1)
        cumulative_total = np.cumsum(class_counts[valid], axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        ap = (precision * positive_counts[valid]).sum(axis=1) / total_positive[valid]
        values[valid, class_idx] = ap
    with np.errstate(invalid="ignore"):
        valid_counts = np.sum(np.isfinite(values), axis=1)
        summed = np.nansum(values, axis=1)
        macro = np.divide(summed, valid_counts, out=np.full(counts.shape[0], np.nan), where=valid_counts > 0)
    return macro


def bootstrap_ci(
    *,
    pred_a: Mapping[str, Any],
    pred_b: Mapping[str, Any],
    classes: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    targets, probs_a, probs_b = align_predictions(pred_a, pred_b)
    if targets.shape[0] < 2:
        return {"ci_low": None, "ci_high": None, "bootstrap_mean": None, "n_bootstrap_valid": 0}
    rng = np.random.default_rng(int(seed))
    n = targets.shape[0]

    order_a = [np.argsort(-probs_a[:, idx], kind="mergesort") for idx in range(len(classes))]
    order_b = [np.argsort(-probs_b[:, idx], kind="mergesort") for idx in range(len(classes))]
    y_sorted_a = [targets[order_a[idx], idx].astype(np.float64) for idx in range(len(classes))]
    y_sorted_b = [targets[order_b[idx], idx].astype(np.float64) for idx in range(len(classes))]
    deltas: list[np.ndarray] = []
    chunk_size = 64
    for start in range(0, int(n_bootstrap), chunk_size):
        chunk = min(chunk_size, int(n_bootstrap) - start)
        sampled = rng.integers(0, n, size=(chunk, n), dtype=np.int64)
        counts = np.zeros((chunk, n), dtype=np.float32)
        np.add.at(counts, (np.repeat(np.arange(chunk), n), sampled.reshape(-1)), 1.0)
        macro_a = bootstrap_macro_auprc_from_counts(counts, order_a, y_sorted_a)
        macro_b = bootstrap_macro_auprc_from_counts(counts, order_b, y_sorted_b)
        delta = macro_a - macro_b
        deltas.append(delta[np.isfinite(delta)])
    if not deltas:
        return {"ci_low": None, "ci_high": None, "bootstrap_mean": None, "n_bootstrap_valid": 0}
    arr = np.concatenate(deltas)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"ci_low": None, "ci_high": None, "bootstrap_mean": None, "n_bootstrap_valid": 0}
    return {
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "bootstrap_mean": float(np.mean(arr)),
        "n_bootstrap_valid": int(arr.shape[0]),
    }


def build_bootstrap_rows(
    predictions: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    *,
    classes: Sequence[str],
    source: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison, method_a, method_b in PRIMARY_COMPARISONS:
        keys_a = [key for key in predictions if key[0] == method_a and key[3] == source]
        for key_a in keys_a:
            _, seed, pattern, _source = key_a
            key_b = (method_b, seed, pattern, source)
            if key_b not in predictions:
                continue
            delta, n_aligned = delta_for_predictions(predictions[key_a], predictions[key_b], classes)
            ci = bootstrap_ci(
                pred_a=predictions[key_a],
                pred_b=predictions[key_b],
                classes=classes,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + int(seed),
            )
            rows.append(
                {
                    "comparison": comparison,
                    "method_a": method_a,
                    "method_b": method_b,
                    "seed": seed,
                    "pattern": pattern,
                    "source": source,
                    "metric": "macro_auprc_available",
                    "observed_delta": delta["macro_delta"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "bootstrap_mean": ci["bootstrap_mean"],
                    "n_bootstrap": int(n_bootstrap),
                    "n_bootstrap_valid": ci["n_bootstrap_valid"],
                    "bootstrap_level": "record",
                    "n_aligned_records": n_aligned,
                }
            )
    return rows


def build_seed_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["scope"]), str(row["source"]), str(row["method_id"]), str(row["pattern"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (scope, source, method_id, pattern), group in sorted(groups.items()):
        values = [row.get("macro_auprc_available") for row in group if row.get("macro_auprc_available") not in (None, "")]
        floats = [float(value) for value in values]
        seeds = [str(row.get("seed")) for row in group]
        out.append(
            {
                "scope": scope,
                "source": source,
                "method_id": method_id,
                "pattern": pattern,
                "n_seeds": len(group),
                "seeds": "|".join(seeds),
                "macro_auprc_available_mean": float(np.mean(floats)) if floats else None,
                "macro_auprc_available_sd": float(np.std(floats, ddof=1)) if len(floats) > 1 else 0.0 if floats else None,
                "status": str(group[0].get("status", "")),
            }
        )
    return out


def rows_for_scope(
    pred: Mapping[str, Any],
    *,
    run: MethodRun,
    pattern_name: str,
    classes: Sequence[str],
    dry_run: bool,
    smoke_test: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prediction_map: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    sources = sorted(set(str(x) for x in np.asarray(pred["sources"]).tolist()))
    for source in sources:
        mask = np.asarray(pred["sources"]) == source
        sub = subset_prediction(pred, mask)
        metrics = compute_auprc_metrics(sub["targets"], sub["probs"], classes)
        rows.append(
            result_row(
                scope="by_source",
                source=source,
                run=run,
                pattern=pattern_name,
                metrics=metrics,
                n_records=int(np.asarray(sub["targets"]).shape[0]),
                classes=classes,
                dry_run=dry_run,
                smoke_test=smoke_test,
            )
        )
        prediction_map[(run.method_id, run.seed, pattern_name, source)] = sub
    metrics = compute_auprc_metrics(pred["targets"], pred["probs"], classes)
    rows.append(
        result_row(
            scope="pooled",
            source="pooled",
            run=run,
            pattern=pattern_name,
            metrics=metrics,
            n_records=int(np.asarray(pred["targets"]).shape[0]),
            classes=classes,
            dry_run=dry_run,
            smoke_test=smoke_test,
        )
    )
    prediction_map[(run.method_id, run.seed, pattern_name, "pooled")] = dict(pred)
    return rows, prediction_map


def output_headers() -> dict[str, list[str]]:
    result_cols = [
        "status",
        "dry_run",
        "smoke_test",
        "scope",
        "source",
        "method_id",
        "seed",
        "pattern",
        "n_records",
        "classes",
        "macro_auprc_available",
        "macro_auprc_n_defined_classes",
        "bootstrap_level",
        "aggregate_components",
        "warnings",
        "n_positive_MI",
        "auprc_MI",
        "n_positive_STTC",
        "auprc_STTC",
        "n_positive_CD",
        "auprc_CD",
        "n_positive_HYP",
        "auprc_HYP",
    ]
    return {
        "external_frozen_results_by_source.csv": result_cols,
        "external_frozen_results_pooled.csv": result_cols,
        "external_frozen_delta_by_source.csv": [
            "comparison",
            "method_a",
            "method_b",
            "seed",
            "pattern",
            "source",
            "n_aligned_records",
            "metric",
            "delta",
        ],
        "external_frozen_delta_pooled.csv": [
            "comparison",
            "method_a",
            "method_b",
            "seed",
            "pattern",
            "source",
            "n_aligned_records",
            "metric",
            "delta",
        ],
        "external_frozen_bootstrap_ci_by_source.csv": [
            "comparison",
            "method_a",
            "method_b",
            "seed",
            "pattern",
            "source",
            "metric",
            "observed_delta",
            "ci_low",
            "ci_high",
            "bootstrap_mean",
            "n_bootstrap",
            "n_bootstrap_valid",
            "bootstrap_level",
            "n_aligned_records",
        ],
        "external_frozen_bootstrap_ci_pooled.csv": [
            "comparison",
            "method_a",
            "method_b",
            "seed",
            "pattern",
            "source",
            "metric",
            "observed_delta",
            "ci_low",
            "ci_high",
            "bootstrap_mean",
            "n_bootstrap",
            "n_bootstrap_valid",
            "bootstrap_level",
            "n_aligned_records",
        ],
        "external_frozen_perclass_delta.csv": [
            "comparison",
            "method_a",
            "method_b",
            "seed",
            "pattern",
            "source",
            "class",
            "metric",
            "delta",
        ],
        "external_frozen_seed_summary.csv": [
            "scope",
            "source",
            "method_id",
            "pattern",
            "n_seeds",
            "seeds",
            "macro_auprc_available_mean",
            "macro_auprc_available_sd",
            "status",
        ],
    }


def write_placeholder_outputs(out_dir: Path, manifest_rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> None:
    write_csv(out_dir / "external_frozen_manifest_used.csv", manifest_rows)
    for filename, columns in output_headers().items():
        write_csv(out_dir / filename, [], fieldnames=columns)
    write_audit_reports(out_dir, report, [], [], [])


def write_audit_reports(
    out_dir: Path,
    report: Mapping[str, Any],
    by_source_rows: Sequence[Mapping[str, Any]],
    pooled_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_json(out_dir / "external_frozen_audit_report.json", dict(report))
    status = report.get("status")
    lines = [
        "# External Frozen Audit Runner Report",
        "",
        f"- Status: `{status}`",
        f"- Created at: `{report.get('created_at')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Sources: `{report.get('sources')}`",
        f"- Classes used: `{report.get('classes')}`",
        "- NORM external metric: `disallowed_and_not_computed`",
        "- Five-label external Macro AUPRC: `disallowed_and_not_computed`",
        "- Bootstrap level: `record`",
        f"- Manifest rows: `{report.get('n_manifest_records')}`",
        f"- Methods/seeds: `{report.get('method_seed_pairs')}`",
        f"- Patterns requested: `{report.get('patterns_requested')}`",
        f"- Patterns evaluated: `{report.get('patterns_evaluated')}`",
        "",
        "## Semantics",
        "",
        "- Records are eligible only from the requested approved sources, with paired WFDB header/waveform files and all 12 standard leads.",
        "- Reviewed SNOMED mapping is used only to construct MI/STTC/CD/HYP labels.",
        "- Records without a positive reviewed available-class label are retained as negative-only records and counted in the manifest.",
        "- Model output columns are selected from internal order `NORM, MI, STTC, CD, HYP`; only MI/STTC/CD/HYP are scored.",
        "- Missing leads are simulated by mask-aware mean-fill semantics after train-fold normalization.",
        "",
        "## Result Status",
        "",
    ]
    if report.get("dry_run"):
        lines.append("Dry-run completed metadata, mapping, checkpoint, class-order, pattern, and output-path checks only. No inference ran.")
    elif report.get("smoke_test"):
        lines.append("Smoke-test outputs are implementation checks only and must not be interpreted as evidence.")
    else:
        lines.append("Full frozen audit mode completed. Interpret per-source rows before pooled rows.")
    lines.extend(
        [
            "",
            f"- By-source result rows: `{len(by_source_rows)}`",
            f"- Pooled result rows: `{len(pooled_rows)}`",
            f"- Bootstrap CI rows: `{len(bootstrap_rows)}`",
            "",
            "## Limitations",
            "",
            "- External labels are reviewed mappings from SNOMED-style codes to PTB-XL superclasses, not native PTB-XL labels.",
            "- Missing-lead conditions are simulated masking on complete 12-lead records.",
            "- Current bootstrap is record-level because patient IDs are unavailable.",
            "- This report does not make a deployment or generalization claim.",
            "",
            "## Next Command",
            "",
            "Run the full frozen audit only after explicit approval, using the command recorded in the preflight report.",
        ]
    )
    (out_dir / "external_frozen_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    interpretation = [
        "# External Frozen Audit Interpretation",
        "",
        f"- Status: `{status}`",
        "- This file is generated by the audit runner.",
    ]
    if report.get("dry_run"):
        interpretation.append("- Dry-run only: no evidence interpretation is available.")
    elif report.get("smoke_test"):
        interpretation.append("- Smoke-test only: metrics, if present, are not evidence and should not be cited.")
    else:
        interpretation.append("- Interpret per-source results first and pooled results second.")
        interpretation.append("- Compare A4a against A1 and A2 without selecting checkpoints from these data.")
    (out_dir / "external_frozen_audit_interpretation.md").write_text("\n".join(interpretation) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen, evaluation-only external Challenge 2021 audit with "
            "available PTB-XL superclasses MI/STTC/CD/HYP. Dry-run and smoke-test "
            "modes are supported; NORM external metrics are refused."
        )
    )
    parser.add_argument("--root", required=True, type=Path, help="External Challenge 2021 root, usually data/external/challenge2021.")
    parser.add_argument("--mapping", required=True, type=Path, help="Reviewed SNOMED-to-PTBXL mapping CSV.")
    parser.add_argument("--sources", nargs="+", required=True, help="Approved sources to include, e.g. georgia cpsc_2018 cpsc_2018_extra.")
    parser.add_argument("--exclude-sources", nargs="+", required=True, help="Sources explicitly excluded from evaluation.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for manifest, metrics, deltas, bootstrap, and reports.")
    parser.add_argument("--classes", nargs="+", required=True, help="External available classes. Must be MI STTC CD HYP or an ordered subset.")
    parser.add_argument("--bootstrap-level", required=True, choices=["record"], help="Bootstrap unit. Current external data support record only.")
    parser.add_argument("--no-train", action="store_true", help="Required safety flag: do not train or fine-tune.")
    parser.add_argument("--no-tune", action="store_true", help="Required safety flag: do not tune thresholds or hyperparameters.")
    parser.add_argument("--no-calibration", action="store_true", help="Required safety flag: do not calibrate or temperature-scale.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="Methods to evaluate from existing checkpoints.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Requested seeds.")
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS), help="Lead patterns or aggregate pattern names.")
    parser.add_argument("--save-predictions", action="store_true", help="Save per-record logits/probabilities for selected available classes.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of record-level bootstrap replicates for paired deltas.")
    parser.add_argument("--bootstrap-seed", type=int, default=20260607, help="Random seed for bootstrap resampling.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--max-records-per-source", type=int, default=None, help="Limit records per source for dry-run/smoke-test.")
    parser.add_argument("--dry-run", action="store_true", help="Run preflight checks and write placeholder outputs without inference.")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny inference test; outputs are marked smoke_test=true.")
    parser.add_argument("--skip-a5-lite", action="store_true", help="Remove A5-lite from requested methods if present.")
    parser.add_argument("--include-a5-lite", action="store_true", help="Include A5-lite trade-off ablation when checkpoint loading is supported.")
    parser.add_argument("--include-kvisible", action="store_true", help="Enable optional k-visible random stress patterns.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress.")
    return parser.parse_args(argv)


def prepare_runs(methods: Sequence[str], seeds: Sequence[int], include_a5: bool, skip_a5: bool) -> list[MethodRun]:
    requested = list(dict.fromkeys(methods))
    if include_a5 and "A5_lite_confidence_consistency_0p05" not in requested:
        requested.append("A5_lite_confidence_consistency_0p05")
    if skip_a5:
        requested = [method for method in requested if method != "A5_lite_confidence_consistency_0p05"]
    if "A5_lite_confidence_consistency_0p05" in requested and not include_a5:
        raise RuntimeError("A5-lite is a trade-off ablation; pass --include-a5-lite to evaluate it")
    runs = discover_method_runs(requested)
    assert_no_forbidden_path_tokens_in_runs(runs)
    selected = [run for run in runs if run.method_id in requested and run.seed in set(seeds)]
    if not selected:
        raise RuntimeError("No requested method/seed checkpoints found")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    classes = validate_classes(args.classes)
    sources = validate_sources(args.sources)
    validate_required_flags(args)

    exclude_sources = tuple(str(item).strip() for item in args.exclude_sources if str(item).strip())
    forbidden_excluded_missing = sorted(FORBIDDEN_SOURCES.difference(exclude_sources))
    if forbidden_excluded_missing:
        raise RuntimeError(f"These forbidden sources must be listed in --exclude-sources: {forbidden_excluded_missing}")
    if set(sources).intersection(exclude_sources):
        raise RuntimeError("A source cannot be both included and excluded")

    root = args.root
    out_dir = args.out_dir
    mapping_path = args.mapping
    ensure_not_locked_output_dir(out_dir)
    ensure_no_forbidden_path_tokens([root, out_dir, mapping_path])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_bootstrap < 1:
        raise RuntimeError("--n-bootstrap must be positive")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be positive")
    if args.num_workers < 0:
        raise RuntimeError("--num-workers must be non-negative")
    if args.max_records_per_source is not None and args.max_records_per_source < 1:
        raise RuntimeError("--max-records-per-source must be positive when provided")

    mapping = load_reviewed_mapping(mapping_path)
    runs = prepare_runs(args.methods, args.seeds, args.include_a5_lite, args.skip_a5_lite)
    ensure_no_forbidden_path_tokens([run.output_dir for run in runs])

    registry = build_pattern_registry(args.include_kvisible)
    eval_pattern_names, aggregate_map = expand_patterns(args.patterns, registry)
    manifest_rows = build_external_manifest(
        root=root,
        sources=sources,
        mapping=mapping,
        classes=classes,
        max_records_per_source=args.max_records_per_source,
        dry_run=bool(args.dry_run),
        smoke_test=bool(args.smoke_test),
    )
    write_csv(out_dir / "external_frozen_manifest_used.csv", manifest_rows)

    method_seed_pairs = [f"{run.method_id}:seed{run.seed}" for run in runs]
    report_base: dict[str, Any] = {
        "created_at": utc_now(),
        "status": "dry_run" if args.dry_run else ("smoke_test" if args.smoke_test else "full_audit"),
        "mode": "dry_run" if args.dry_run else ("smoke_test" if args.smoke_test else "full_audit"),
        "dry_run": bool(args.dry_run),
        "smoke_test": bool(args.smoke_test),
        "root": str(root),
        "mapping": str(mapping_path),
        "sources": "|".join(sources),
        "exclude_sources": "|".join(exclude_sources),
        "classes": "|".join(classes),
        "internal_label_order": "|".join(INTERNAL_LABEL_ORDER),
        "external_class_indices": {label: INTERNAL_CLASS_INDEX[label] for label in classes},
        "n_manifest_records": len(manifest_rows),
        "method_seed_pairs": "|".join(method_seed_pairs),
        "patterns_requested": "|".join(args.patterns),
        "patterns_evaluated": "|".join(eval_pattern_names),
        "aggregate_patterns": {key: list(value) for key, value in aggregate_map.items()},
        "bootstrap_level": "record",
        "n_bootstrap": int(args.n_bootstrap),
        "safety_flags": {"no_train": True, "no_tune": True, "no_calibration": True},
        "norm_external_metric_computed": False,
        "five_label_external_macro_auprc_computed": False,
        "checkpoint_selection_on_external_data": False,
    }

    if args.dry_run:
        write_placeholder_outputs(out_dir, manifest_rows, report_base)
        if args.verbose:
            print(json.dumps(report_base, indent=2, ensure_ascii=False))
        return 0

    device = resolve_device_external(str(args.device))
    norm_stats_path = REPO_ROOT / "outputs/day1_audit/train_norm_stats.npz"
    if not norm_stats_path.exists():
        raise RuntimeError(f"Missing normalization stats: {norm_stats_path}")
    ensure_no_forbidden_path_tokens([norm_stats_path])

    by_source_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    predictions_by_source: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    predictions_dir = out_dir / "predictions"

    records_by_source = {
        source: [row for row in manifest_rows if row["source"] == source]
        for source in sources
    }
    for run in runs:
        config: dict[str, Any] = {}
        for pattern_name in eval_pattern_names:
            pattern = registry[pattern_name]
            records = [row for source in sources for row in records_by_source[source]]
            if args.verbose:
                print(f"Evaluating {run.method_run_id} {pattern_name} n={len(records)} on {device}", flush=True)
            pred = predict_external(
                run=run,
                config=config,
                records=records,
                pattern_name=pattern_name,
                pattern=pattern,
                classes=classes,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                norm_stats_path=norm_stats_path,
            )
            rows, pred_map = rows_for_scope(
                pred,
                run=run,
                pattern_name=pattern_name,
                classes=classes,
                dry_run=False,
                smoke_test=bool(args.smoke_test),
            )
            for row in rows:
                if row["scope"] == "pooled":
                    pooled_rows.append(row)
                else:
                    by_source_rows.append(row)
            predictions_by_source.update(pred_map)
            if args.save_predictions:
                save_prediction_csv(
                    predictions_dir / f"{run.method_id}_seed{run.seed}_{pattern_name}.csv",
                    pred,
                    classes,
                    run,
                )

    by_source_rows.extend(
        build_aggregate_metric_rows(
            by_source_rows,
            aggregates=aggregate_map,
            classes=classes,
        )
    )
    pooled_rows.extend(
        build_aggregate_metric_rows(
            pooled_rows,
            aggregates=aggregate_map,
            classes=classes,
        )
    )

    delta_by_source: list[dict[str, Any]] = []
    perclass_delta: list[dict[str, Any]] = []
    for source in sources:
        macro_rows, per_rows = build_delta_rows(predictions_by_source, classes=classes, source=source)
        delta_by_source.extend(macro_rows)
        perclass_delta.extend(per_rows)
    delta_pooled, per_rows_pooled = build_delta_rows(predictions_by_source, classes=classes, source="pooled")
    perclass_delta.extend(per_rows_pooled)
    seed_summary = build_seed_summary([*by_source_rows, *pooled_rows])

    headers = output_headers()
    write_csv(out_dir / "external_frozen_results_by_source.csv", by_source_rows, fieldnames=headers["external_frozen_results_by_source.csv"])
    write_csv(out_dir / "external_frozen_results_pooled.csv", pooled_rows, fieldnames=headers["external_frozen_results_pooled.csv"])
    write_csv(out_dir / "external_frozen_delta_by_source.csv", delta_by_source, fieldnames=headers["external_frozen_delta_by_source.csv"])
    write_csv(out_dir / "external_frozen_delta_pooled.csv", delta_pooled, fieldnames=headers["external_frozen_delta_pooled.csv"])
    write_csv(out_dir / "external_frozen_perclass_delta.csv", perclass_delta, fieldnames=headers["external_frozen_perclass_delta.csv"])
    write_csv(out_dir / "external_frozen_seed_summary.csv", seed_summary, fieldnames=headers["external_frozen_seed_summary.csv"])

    bootstrap_by_source: list[dict[str, Any]] = []
    for source in sources:
        bootstrap_by_source.extend(
            build_bootstrap_rows(
                predictions_by_source,
                classes=classes,
                source=source,
                n_bootstrap=int(args.n_bootstrap),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        )
    bootstrap_pooled = build_bootstrap_rows(
        predictions_by_source,
        classes=classes,
        source="pooled",
        n_bootstrap=int(args.n_bootstrap),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    write_csv(
        out_dir / "external_frozen_bootstrap_ci_by_source.csv",
        bootstrap_by_source,
        fieldnames=headers["external_frozen_bootstrap_ci_by_source.csv"],
    )
    write_csv(
        out_dir / "external_frozen_bootstrap_ci_pooled.csv",
        bootstrap_pooled,
        fieldnames=headers["external_frozen_bootstrap_ci_pooled.csv"],
    )

    report = dict(report_base)
    report.update(
        {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "n_by_source_result_rows": len(by_source_rows),
            "n_pooled_result_rows": len(pooled_rows),
            "n_delta_by_source_rows": len(delta_by_source),
            "n_delta_pooled_rows": len(delta_pooled),
            "n_bootstrap_by_source_rows": len(bootstrap_by_source),
            "n_bootstrap_pooled_rows": len(bootstrap_pooled),
            "outputs": list(EXPECTED_OUTPUT_FILES),
        }
    )
    write_audit_reports(out_dir, report, by_source_rows, pooled_rows, [*bootstrap_by_source, *bootstrap_pooled])
    if args.verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
