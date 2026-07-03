"""Week 6 reviewer-defense evaluation helpers.

These utilities are evaluation-only. They load existing checkpoints and
validation thresholds, evaluate additional simulated missing-lead views, and
write outputs under a new Week 6 directory without changing locked results.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.data.waveforms import CANONICAL_LEADS
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.evaluation.evaluate_patterns import load_model_from_checkpoint
from hlm_ecg.evaluation.metrics import compute_multilabel_metrics
from hlm_ecg.evaluation.missing_patterns import MissingPattern, required_patterns
from hlm_ecg.evaluation.prediction_artifacts import (
    build_prediction_output_path,
    load_thresholds_val,
    prediction_rows,
    safe_pattern_name,
    save_predictions_csv,
)
from hlm_ecg.evaluation.supplemental_analysis import (
    MethodRun,
    base_metadata,
    discover_method_runs,
    git_commit,
    markdown_report,
    paired_bootstrap_prediction_delta,
    read_csv,
    summarize_multiseed,
    utc_now,
    write_csv,
    write_json,
    write_markdown_table,
)
from hlm_ecg.evaluation.supplemental_patterns import (
    KVisibleRandomPattern,
    challenge_reduced_lead_patterns,
    k_visible_random_patterns,
    pattern_metadata,
)
from hlm_ecg.training.train_baseline import forward_model, resolve_device


ROOT = Path(__file__).resolve().parents[3]
WEEK6_DIR = ROOT / "outputs/week6_reviewer_defense"
WEEK6_PATTERN_SEED = 20240606
NONINFERIORITY_MARGIN_AUPRC = -0.0100
IMPUTATION_STRATEGIES = (
    "mean_fill",
    "zero_fill",
    "physiology_limb_reconstruction_fill",
)
I_II_DERIVED_LIMB_LEADS = ("III", "aVR", "aVL", "aVF")
PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")
AVAILABILITY_VARIANTS = ("correct", "all_ones", "shuffled")
DEFAULT_NO_TRAIN_METHODS = (
    "A1_random_dropout",
    "A2_structured_masking",
    "A4a_subclass_auxiliary",
)
HARD_STRUCTURED_PATTERNS = (
    "limb-only / precordial-missing",
    "precordial-only / limb-missing",
    "V1-V3 missing",
    "V4-V6 missing",
)
CHALLENGE_RECON_PATTERNS = (
    "challenge_6_limb",
    "challenge_4_I_II_III_V2",
    "challenge_3_I_II_V2",
    "challenge_2_I_II",
)
K_BOUNDARY_PATTERNS = ("k3_visible_random", "k2_visible_random", "k1_visible_random")
BOUNDARY_PATTERNS = (*HARD_STRUCTURED_PATTERNS, *CHALLENGE_RECON_PATTERNS, *K_BOUNDARY_PATTERNS)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}")
    return data


def week6_pattern_registry(seed: int = WEEK6_PATTERN_SEED) -> dict[str, Any]:
    """Return all Week 6 fixed and deterministic random pattern definitions."""

    patterns: dict[str, Any] = {}
    patterns.update(required_patterns(seed))
    patterns.update(challenge_reduced_lead_patterns(seed))
    patterns.update(k_visible_random_patterns(seed))
    return patterns


def selected_patterns(names: Sequence[str] | None, *, seed: int = WEEK6_PATTERN_SEED) -> dict[str, Any]:
    registry = week6_pattern_registry(seed)
    if names is None:
        return registry
    unknown = set(names).difference(registry)
    if unknown:
        raise ValueError(f"Unknown Week 6 patterns: {sorted(unknown)}")
    return {name: registry[name] for name in names}


def _as_mask(pattern: Any, idx: int) -> np.ndarray:
    mask = np.asarray(pattern.mask_for_index(idx), dtype=np.float32)
    if mask.shape != (12,) or not np.all(np.isin(mask, [0.0, 1.0])) or float(mask.sum()) < 1.0:
        raise RuntimeError(f"Invalid Week 6 mask for {getattr(pattern, 'name', pattern)}: {mask}")
    return mask


def reconstruct_limb_leads_from_i_ii(raw_i: np.ndarray, raw_ii: np.ndarray) -> dict[str, np.ndarray]:
    """Return standard limb-lead reconstructions from raw I and II signals."""

    raw_i = np.asarray(raw_i, dtype=np.float32)
    raw_ii = np.asarray(raw_ii, dtype=np.float32)
    return {
        "III": raw_ii - raw_i,
        "aVR": -0.5 * (raw_i + raw_ii),
        "aVL": raw_i - 0.5 * raw_ii,
        "aVF": raw_ii - 0.5 * raw_i,
    }


def limb_reconstruction_applicability(pattern_name: str, pattern: Any, *, idx: int = 0) -> dict[str, Any]:
    """Describe whether I/II-derived limb reconstruction changes a pattern.

    This audit intentionally covers only analytic limb-lead relationships from
    measured I and II. It does not synthesize precordial leads.
    """

    mask = _as_mask(pattern, idx)
    lead_mask = {lead: int(mask[pos]) for pos, lead in enumerate(CANONICAL_LEADS)}
    available = [lead for lead in CANONICAL_LEADS if lead_mask[lead] == 1]
    missing = [lead for lead in CANONICAL_LEADS if lead_mask[lead] == 0]
    reconstructable = (
        [lead for lead in I_II_DERIVED_LIMB_LEADS if lead in missing]
        if lead_mask["I"] == 1 and lead_mask["II"] == 1
        else []
    )

    if not missing:
        no_op_reason = "no_missing_leads"
    elif reconstructable:
        no_op_reason = ""
    elif lead_mask["I"] != 1 or lead_mask["II"] != 1:
        no_op_reason = "i_or_ii_unavailable"
    elif all(lead in PRECORDIAL_LEADS for lead in missing):
        no_op_reason = "missing_chest_precordial_leads_not_synthesized"
    else:
        no_op_reason = "no_i_ii_derived_limb_leads_missing"

    return {
        "pattern": pattern_name,
        "available_leads": ",".join(available),
        "missing_leads": ",".join(missing),
        "reconstructable_missing_limb_leads": ",".join(reconstructable),
        "n_reconstructed_leads": int(len(reconstructable)),
        "no_op_reason": no_op_reason,
        "reconstruction_scope": "I/II-derived limb leads only; no precordial synthesis",
    }


class Week6ImputationDataset(PTBXLDataset):
    """PTB-XL test/validation dataset with explicit imputation strategies.

    The model availability mask always reflects measured leads, not imputed or
    reconstructed leads. This is important for the reviewer-defense question:
    "should we reconstruct first?" rather than silently telling the model that
    reconstructed leads were truly measured.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        split: str,
        pattern: Any,
        imputation_strategy: str,
        smoke_test: bool = False,
    ) -> None:
        if imputation_strategy not in IMPUTATION_STRATEGIES:
            raise ValueError(f"Unknown imputation strategy: {imputation_strategy}")
        paths = dict(config.get("paths", {}))
        smoke = dict(config.get("smoke", {}))
        limit = int(smoke.get(f"{split}_limit", 64)) if smoke_test else None
        self.week6_pattern = pattern
        self.imputation_strategy = imputation_strategy
        super().__init__(
            root=Path(paths.get("data_root", "data/ptb-xl")),
            index_csv=Path(paths.get("day1_index", "outputs/day1_audit/ptbxl_day1_index.csv")),
            norm_stats_path=Path(paths.get("norm_stats", "outputs/day1_audit/train_norm_stats.npz")),
            split=split,
            fill_mode="full",
            limit=limit,
        )

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        row = self.df.iloc[int(idx)]
        filename_lr = str(row["filename_lr"])
        raw, fields = self._read_raw(filename_lr)
        mask = _as_mask(self.week6_pattern, int(idx))
        x = self._impute(raw, mask).T.astype(np.float32, copy=False)
        y = row[list(LABEL_ORDER)].to_numpy(dtype=np.float32)
        out = {
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(float(row["patient_id"])) if "patient_id" in row.index and not np.isnan(float(row["patient_id"])) else -1,
            "strat_fold": int(row["strat_fold"]) if "strat_fold" in row.index and not np.isnan(float(row["strat_fold"])) else -1,
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "lead_mask": torch.from_numpy(mask.astype(np.float32)),
            "availability_mask": torch.from_numpy(mask.astype(np.float32)),
            "split": str(row["split"]),
            "filename_lr": filename_lr,
            "fs": int(fields.get("fs", -1)),
            "lead_names": list(fields.get("sig_name", [])),
        }
        return out

    def _impute(self, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if self.imputation_strategy == "zero_fill":
            filled = raw.copy()
            filled[:, mask == 0] = 0.0
            return (filled - self.mean.reshape(1, 12)) / self.std.reshape(1, 12)

        x = (raw - self.mean.reshape(1, 12)) / self.std.reshape(1, 12)
        if np.all(mask == 1):
            return x
        x = x.copy()
        x[:, mask == 0] = 0.0
        if self.imputation_strategy == "physiology_limb_reconstruction_fill":
            self._apply_limb_reconstruction(raw, mask, x)
        return x

    def _apply_limb_reconstruction(self, raw: np.ndarray, mask: np.ndarray, x: np.ndarray) -> None:
        lead_to_idx = {lead: idx for idx, lead in enumerate(CANONICAL_LEADS)}
        i_idx = lead_to_idx["I"]
        ii_idx = lead_to_idx["II"]
        if mask[i_idx] != 1 or mask[ii_idx] != 1:
            return
        raw_i = raw[:, i_idx].astype(np.float32, copy=False)
        raw_ii = raw[:, ii_idx].astype(np.float32, copy=False)
        reconstructed = reconstruct_limb_leads_from_i_ii(raw_i, raw_ii)
        for lead, raw_values in reconstructed.items():
            lead_idx = lead_to_idx[lead]
            if mask[lead_idx] == 0:
                x[:, lead_idx] = (raw_values - self.mean[lead_idx]) / self.std[lead_idx]


def _tensor_to_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return arr.astype(dtype) if dtype is not None else arr


def transform_availability_mask(mask: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "correct":
        return mask
    if variant == "all_ones":
        return torch.ones_like(mask)
    if variant == "shuffled":
        if int(mask.shape[0]) <= 1:
            return mask
        return torch.roll(mask, shifts=1, dims=0)
    raise ValueError(f"Unknown availability variant: {variant}")


@torch.no_grad()
def collect_week6_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    availability_variant: str = "correct",
) -> dict[str, Any]:
    model.eval()
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    ecg_ids: list[np.ndarray] = []
    patient_ids: list[np.ndarray] = []
    strat_folds: list[np.ndarray] = []
    availability_masks: list[np.ndarray] = []
    splits: list[str] = []
    for batch in loader:
        batch_for_forward = dict(batch)
        original_mask = batch["availability_mask"].to(dtype=torch.float32)
        forward_mask = transform_availability_mask(original_mask, availability_variant)
        batch_for_forward["availability_mask"] = forward_mask
        logits = forward_model(model, batch_for_forward, device=device).detach().cpu().numpy()
        logits_all.append(logits)
        targets_all.append(batch["y"].detach().cpu().numpy())
        ecg_ids.append(_tensor_to_numpy(batch["ecg_id"], dtype=np.int64))
        patient_ids.append(_tensor_to_numpy(batch.get("patient_id", np.full(logits.shape[0], -1)), dtype=np.int64))
        strat_folds.append(_tensor_to_numpy(batch.get("strat_fold", np.full(logits.shape[0], -1)), dtype=np.int64))
        availability_masks.append(forward_mask.detach().cpu().numpy().astype(np.float32))
        split_value = batch.get("split", "")
        if isinstance(split_value, (list, tuple)):
            splits.extend(str(x) for x in split_value)
        else:
            splits.extend([str(split_value)] * int(logits.shape[0]))
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "targets": np.concatenate(targets_all, axis=0),
        "ecg_ids": np.concatenate(ecg_ids, axis=0),
        "patient_ids": np.concatenate(patient_ids, axis=0),
        "strat_folds": np.concatenate(strat_folds, axis=0),
        "availability_masks": np.concatenate(availability_masks, axis=0),
        "splits": splits,
    }


def metrics_row(
    *,
    run: MethodRun,
    pattern: str,
    fill_mode: str,
    metrics: Mapping[str, Any],
    n: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method_id": run.method_id,
        "seed": int(run.seed),
        "method_run_id": run.method_run_id,
        "pattern": pattern,
        "fill_mode": fill_mode,
        "n": int(n),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "macro_f1": metrics["macro_f1"],
        "bce_nll": metrics["bce_nll"],
        "thresholds_source_split": "val",
        "records500_used": False,
        "output_dir": str(run.output_dir.relative_to(ROOT)),
    }
    per_class = metrics.get("per_class_auprc", {})
    if isinstance(per_class, Mapping):
        for label in LABEL_ORDER:
            row[f"auprc_{label}"] = per_class.get(label)
    if extra:
        row.update(dict(extra))
    return row


def evaluate_week6_pattern(
    *,
    run: MethodRun,
    pattern_name: str,
    pattern: Any,
    imputation_strategy: str,
    split: str = "test",
    smoke_test: bool = False,
    save_predictions: bool = False,
    predictions_dir: Path | None = None,
    availability_variant: str = "correct",
    prediction_method_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    config = load_yaml(run.config_path)
    device = resolve_device(str(config.get("device", "auto")))
    model = load_model_from_checkpoint(run.checkpoint_path, config, device)
    thresholds, _threshold_map, threshold_source_split = load_thresholds_val(run.output_dir)
    eval_cfg = dict(config.get("evaluation", {}))
    train_cfg = dict(config.get("training", {}))
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 64)))
    num_workers = int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 0)))
    dataset = Week6ImputationDataset(
        config=config,
        split=split,
        pattern=pattern,
        imputation_strategy=imputation_strategy,
        smoke_test=smoke_test,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    collected = collect_week6_predictions(model, loader, device=device, availability_variant=availability_variant)
    metrics = compute_multilabel_metrics(collected["logits"], collected["targets"], thresholds=thresholds)
    row = metrics_row(
        run=run,
        pattern=pattern_name,
        fill_mode=imputation_strategy,
        metrics=metrics,
        n=int(collected["targets"].shape[0]),
        extra={"availability_variant": availability_variant},
    )
    prediction_info = None
    if save_predictions:
        if predictions_dir is None:
            raise ValueError("predictions_dir is required when save_predictions=True")
        method_for_predictions = prediction_method_id or run.method_run_id
        pred_rows = prediction_rows(
            method_id=method_for_predictions,
            pattern=pattern_name,
            fill_mode=imputation_strategy,
            split=split,
            random_seed=WEEK6_PATTERN_SEED,
            threshold_source_split=threshold_source_split,
            thresholds=thresholds,
            collected=collected,
        )
        pred_path = build_prediction_output_path(
            predictions_dir,
            method_id=method_for_predictions,
            fill_mode=imputation_strategy,
            split=split,
            pattern=pattern_name,
        )
        prediction_info = {
            "method_id": method_for_predictions,
            "split": split,
            "fill_mode": imputation_strategy,
            "pattern": pattern_name,
            "availability_variant": availability_variant,
            **save_predictions_csv(pred_path, pred_rows),
        }
    return row, prediction_info


def delta_vs_baseline_with_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str = "A1_random_dropout",
    group_fields: Sequence[str] = ("seed", "pattern", "fill_mode"),
) -> list[dict[str, Any]]:
    indexed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (row["method_id"], *[row[field] for field in group_fields])
        indexed[key] = row
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["method_id"] == baseline_method:
            continue
        base_key = (baseline_method, *[row[field] for field in group_fields])
        base = indexed.get(base_key)
        if base is None:
            continue
        item = {
            "method_id": row["method_id"],
            "baseline_method": baseline_method,
            **{field: row[field] for field in group_fields},
            "delta_macro_auprc": float(row["macro_auprc"]) - float(base["macro_auprc"]),
            "delta_macro_auroc": float(row["macro_auroc"]) - float(base["macro_auroc"]),
            "delta_macro_f1": float(row["macro_f1"]) - float(base["macro_f1"]),
            "delta_bce_nll": float(row["bce_nll"]) - float(base["bce_nll"]),
        }
        out.append(item)
    return out


def aggregate_pattern_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    aggregate_name: str,
    patterns: Sequence[str],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    pattern_set = set(patterns)
    for row in rows:
        if row["pattern"] not in pattern_set:
            continue
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        if len({row["pattern"] for row in group}) != len(pattern_set):
            continue
        item = {field: key[idx] for idx, field in enumerate(group_fields)}
        item["pattern"] = aggregate_name
        item["n_patterns"] = len(pattern_set)
        for metric in ("macro_auroc", "macro_auprc", "macro_f1", "bce_nll"):
            item[metric] = float(np.mean([float(row[metric]) for row in group]))
        for label in LABEL_ORDER:
            col = f"auprc_{label}"
            values = [float(row[col]) for row in group if row.get(col) not in (None, "")]
            if values:
                item[col] = float(np.mean(values))
        out.append(item)
    return out


def write_manifest(path: Path, *, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    write_json(path, {"metadata": dict(metadata), "rows": list(rows)})


def method_runs_for_week6(methods: Sequence[str] | None = None) -> list[MethodRun]:
    requested = list(methods or DEFAULT_NO_TRAIN_METHODS)
    runs = discover_method_runs(requested)
    return [run for run in runs if "smoke" not in str(run.output_dir)]


def specialist_training_allowed() -> bool:
    return os.environ.get("WEEK6_ALLOW_SPECIALIST_TRAINING", "false").lower() == "true"


def noninferiority_decision(ci_low: float, margin: float = NONINFERIORITY_MARGIN_AUPRC) -> str:
    if float(ci_low) > float(margin):
        return "noninferior_with_margin"
    return "not_established"


def write_week6_summary_markdown(path: Path, title: str, rows: Sequence[str]) -> None:
    markdown_report(path, title, rows)
