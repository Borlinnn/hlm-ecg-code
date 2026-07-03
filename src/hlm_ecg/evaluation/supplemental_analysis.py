"""Shared helpers for Week 5 BIBM stabilization analyses."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.evaluation.metrics import sigmoid
from hlm_ecg.evaluation.prediction_artifacts import safe_pattern_name
from hlm_ecg.statistics.bootstrap import (
    load_prediction_csv,
    paired_delta_summary,
    patient_groups,
    generate_patient_bootstrap_samples,
    sampled_indices_from_patients,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHODS = ("A1_random_dropout", "A2_structured_masking", "A4a_subclass_auxiliary")
OPTIONAL_METHODS = ("A0_full_no_masking", "A5_lite_confidence_consistency_0p05")
LABEL_METRICS = ("macro_auprc", "macro_auroc", "macro_f1", "bce_nll")


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" * len(columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(format_cell(row.get(col, "")) for col in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    text = str(value)
    if text.startswith("0.") or text.startswith("-0."):
        try:
            return f"{float(text):.4f}"
        except ValueError:
            return text
    return text


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}")
    return data


def _method_runs_from_manifest(method_ids: set[str]) -> list[MethodRun]:
    manifest_path = ROOT / "outputs/week3_multiseed_summary/multiseed_manifest.csv"
    runs: list[MethodRun] = []
    for row in read_csv(manifest_path):
        method_id = row.get("method_id", "")
        if method_id not in method_ids:
            continue
        output_dir = ROOT / row["output_dir"]
        run = make_method_run(method_id, int(row["seed"]), output_dir)
        if run is not None:
            runs.append(run)
    return runs


def _method_runs_from_results_lock(method_ids: set[str]) -> list[MethodRun]:
    summary_path = ROOT / "outputs/week3_results_lock/all_methods_summary.csv"
    runs: list[MethodRun] = []
    for row in read_csv(summary_path):
        method_id = row.get("method_id", "")
        if method_id not in method_ids:
            continue
        output_dir = ROOT / row["output_dir"]
        run = make_method_run(method_id, 42, output_dir)
        if run is not None:
            runs.append(run)
    return runs


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
    runs = _method_runs_from_manifest(requested)
    existing_keys = {(run.method_id, run.seed) for run in runs}
    for run in _method_runs_from_results_lock(requested):
        if (run.method_id, run.seed) not in existing_keys:
            runs.append(run)
            existing_keys.add((run.method_id, run.seed))
    runs.sort(key=lambda run: (run.method_id, run.seed))
    required = {"A1_random_dropout", "A2_structured_masking", "A4a_subclass_auxiliary"}.intersection(requested)
    present = {run.method_id for run in runs}
    missing = sorted(required.difference(present))
    if missing:
        raise FileNotFoundError(f"Missing required method checkpoints/configs/thresholds for: {missing}")
    return runs


def assert_no_records500_in_runs(runs: Sequence[MethodRun]) -> None:
    bad: list[str] = []
    for run in runs:
        for path in (run.output_dir, run.config_path, run.checkpoint_path, run.thresholds_path):
            if "records500" in str(path):
                bad.append(str(path))
        config_text = run.config_path.read_text(encoding="utf-8")
        if "records500" in config_text or "filename_hr" in config_text:
            bad.append(str(run.config_path))
    if bad:
        raise RuntimeError(f"records500/filename_hr detected in week5 method sources: {bad}")


def base_metadata(runs: Sequence[MethodRun], *, fill_mode: str, pattern_seed: int) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "fill_mode": fill_mode,
        "pattern_seed": int(pattern_seed),
        "label_order": list(LABEL_ORDER),
        "records500_used": False,
        "method_runs": [
            {
                "method_id": run.method_id,
                "seed": run.seed,
                "output_dir": str(run.output_dir.relative_to(ROOT)),
                "checkpoint_path": str(run.checkpoint_path.relative_to(ROOT)),
                "config_path": str(run.config_path.relative_to(ROOT)),
                "thresholds_path": str(run.thresholds_path.relative_to(ROOT)),
            }
            for run in runs
        ],
    }


def add_method_run_fields(rows: Sequence[Mapping[str, Any]], run: MethodRun) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "method_id": run.method_id,
                "seed": run.seed,
                "method_run_id": run.method_run_id,
                "output_dir": str(run.output_dir.relative_to(ROOT)),
                "checkpoint_path": str(run.checkpoint_path.relative_to(ROOT)),
                "thresholds_source_split": "val",
                "records500_used": False,
            }
        )
        out.append(item)
    return out


def summarize_multiseed(rows: Sequence[Mapping[str, Any]], *, group_cols: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        groups.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        out = {col: key[idx] for idx, col in enumerate(group_cols)}
        out["n_seeds"] = len({int(row["seed"]) for row in group})
        for metric in LABEL_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def delta_vs_baseline(rows: Sequence[Mapping[str, Any]], *, baseline_method: str = "A1_random_dropout") -> list[dict[str, Any]]:
    by_key = {
        (row["method_id"], int(row["seed"]), row["pattern"]): row
        for row in rows
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["method_id"] == baseline_method:
            continue
        base = by_key.get((baseline_method, int(row["seed"]), row["pattern"]))
        if base is None:
            continue
        out.append(
            {
                "method_id": row["method_id"],
                "baseline_method": baseline_method,
                "seed": int(row["seed"]),
                "pattern": row["pattern"],
                "delta_macro_auprc": float(row["macro_auprc"]) - float(base["macro_auprc"]),
                "delta_macro_auroc": float(row["macro_auroc"]) - float(base["macro_auroc"]),
                "delta_macro_f1": float(row["macro_f1"]) - float(base["macro_f1"]),
                "delta_bce_nll": float(row["bce_nll"]) - float(base["bce_nll"]),
            }
        )
    return out


def figure_macro_auprc(
    rows: Sequence[Mapping[str, Any]],
    *,
    path_prefix: Path,
    pattern_order: Sequence[str],
    method_order: Sequence[str],
    x_labels: Sequence[str],
    xlabel: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    summaries = summarize_multiseed(rows, group_cols=["method_id", "pattern"])
    by_key = {(row["method_id"], row["pattern"]): row for row in summaries}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(pattern_order))
    for method in method_order:
        means = []
        stds = []
        for pattern in pattern_order:
            row = by_key.get((method, pattern))
            means.append(np.nan if row is None else float(row["macro_auprc_mean"]))
            stds.append(0.0 if row is None else float(row["macro_auprc_std"]))
        ax.errorbar(x, means, yerr=stds, marker="o", linewidth=1.8, capsize=3, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro AUPRC")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path_prefix.with_suffix(suffix), dpi=200)
    plt.close(fig)


def ensure_empty_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def backup_if_exists(path: Path) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)


def paired_bootstrap_prediction_delta(
    *,
    predictions_dir: Path,
    method_a_run_id: str,
    method_b_run_id: str,
    method_a: str,
    method_b: str,
    seed: int,
    patterns: Sequence[str],
    fill_mode: str,
    split: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        path_a = predictions_dir / method_a_run_id / fill_mode / split / f"{safe_pattern_name(pattern)}.csv"
        path_b = predictions_dir / method_b_run_id / fill_mode / split / f"{safe_pattern_name(pattern)}.csv"
        if not path_a.exists() or not path_b.exists():
            continue
        data_a = load_prediction_csv(path_a, method_id=method_a_run_id, pattern=pattern, split=split, fill_mode=fill_mode)
        data_b = load_prediction_csv(path_b, method_id=method_b_run_id, pattern=pattern, split=split, fill_mode=fill_mode)
        if not np.array_equal(data_a.ecg_ids, data_b.ecg_ids):
            raise RuntimeError(f"Prediction ECG IDs are not aligned for {pattern}: {method_a_run_id} vs {method_b_run_id}")
        if not np.array_equal(data_a.targets, data_b.targets):
            raise RuntimeError(f"Prediction targets are not aligned for {pattern}: {method_a_run_id} vs {method_b_run_id}")
        observed_delta = macro_auprc_from_logits(data_a.logits, data_a.targets) - macro_auprc_from_logits(
            data_b.logits, data_b.targets
        )
        groups = patient_groups(data_a.patient_ids)
        samples = generate_patient_bootstrap_samples(data_a.patient_ids, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        deltas = []
        for sampled_patients in samples:
            indices = sampled_indices_from_patients(groups, sampled_patients)
            deltas.append(
                macro_auprc_from_logits(data_a.logits[indices], data_a.targets[indices])
                - macro_auprc_from_logits(data_b.logits[indices], data_b.targets[indices])
            )
        summary = paired_delta_summary(np.asarray(deltas, dtype=np.float64), observed_delta)
        rows.append(
            {
                "comparison_id": f"{method_a}_vs_{method_b}",
                "method_a": method_a,
                "method_b": method_b,
                "method_a_run_id": method_a_run_id,
                "method_b_run_id": method_b_run_id,
                "seed": int(seed),
                "pattern": pattern,
                "metric": "macro_auprc",
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_seed": int(bootstrap_seed),
                "sampling_unit": "patient",
                **summary,
            }
        )
    return rows


def macro_auprc_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    probs = sigmoid(logits)
    values = []
    for idx in range(len(LABEL_ORDER)):
        y_true = targets[:, idx]
        if np.unique(y_true).size < 2:
            continue
        values.append(float(average_precision_score(y_true, probs[:, idx])))
    if not values:
        return float("nan")
    return float(np.mean(values))


def markdown_report(path: Path, title: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
