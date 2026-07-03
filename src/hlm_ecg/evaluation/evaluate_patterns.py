"""Evaluate a full-lead baseline under missing-lead test patterns."""

import csv
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.evaluation.metrics import compute_multilabel_metrics
from hlm_ecg.evaluation.missing_patterns import MissingPattern, required_patterns
from hlm_ecg.evaluation.prediction_artifacts import (
    build_prediction_output_path,
    collect_predictions,
    load_thresholds_val,
    prediction_rows,
    save_predictions_csv,
)
from hlm_ecg.training.train_baseline import build_model, forward_model, json_default, resolve_device, write_json


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    targets_all = []
    for batch in loader:
        logits_all.append(forward_model(model, batch, device=device).detach().cpu().numpy())
        targets_all.append(batch["y"].cpu().numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(targets_all, axis=0)


def build_pattern_dataset(
    config: Mapping[str, object],
    pattern: MissingPattern,
    *,
    fill_mode: str,
    split: str = "test",
    smoke_test: bool,
) -> PTBXLDataset:
    paths = dict(config.get("paths", {}))
    smoke = dict(config.get("smoke", {}))
    limit = int(smoke.get("test_limit", 64)) if smoke_test else None
    if pattern.name == "full":
        return PTBXLDataset(
            root=Path(paths.get("data_root", "data/ptb-xl")),
            index_csv=Path(paths.get("day1_index", "outputs/day1_audit/ptbxl_day1_index.csv")),
            norm_stats_path=Path(paths.get("norm_stats", "outputs/day1_audit/train_norm_stats.npz")),
            split=split,
            fill_mode="full",
            limit=limit,
        )
    return PTBXLDataset(
        root=Path(paths.get("data_root", "data/ptb-xl")),
        index_csv=Path(paths.get("day1_index", "outputs/day1_audit/ptbxl_day1_index.csv")),
        norm_stats_path=Path(paths.get("norm_stats", "outputs/day1_audit/train_norm_stats.npz")),
        split=split,
        fill_mode=fill_mode,
        mask_provider=pattern.mask_for_index,
        limit=limit,
    )


def load_thresholds(path: Path) -> Sequence[float]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError(f"Missing thresholds in {path}")
    return [float(thresholds[label]) for label in LABEL_ORDER]


def load_model_from_checkpoint(checkpoint_path: Path, config: Mapping[str, object], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", config)
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_missing_patterns(
    *,
    checkpoint_path: Path,
    config: Mapping[str, object],
    fill_mode: str,
    split: str = "test",
    patterns: Sequence[str] | None = None,
    method_id: str | None = None,
    save_predictions: bool = False,
    predictions_dir: Path | None = None,
    write_metrics: bool = True,
    smoke_test: bool = False,
    pattern_registry: Mapping[str, MissingPattern] | None = None,
) -> Dict[str, object]:
    if fill_mode not in {"zero_fill", "mean_fill"}:
        raise ValueError("fill_mode must be zero_fill or mean_fill")
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    paths = dict(config.get("paths", {}))
    output_dir = Path(paths.get("output_dir", "outputs/week1_full_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.get("device", "auto")))
    thresholds, _threshold_map, threshold_source_split = load_thresholds_val(output_dir)
    model = load_model_from_checkpoint(checkpoint_path, config, device)

    eval_cfg = dict(config.get("evaluation", {}))
    batch_size = int(eval_cfg.get("batch_size", dict(config.get("training", {})).get("batch_size", 64)))
    num_workers = int(eval_cfg.get("num_workers", dict(config.get("training", {})).get("num_workers", 0)))
    seed = int(eval_cfg.get("pattern_seed", 20240604))

    rows = []
    details: Dict[str, object] = {"fill_mode": fill_mode, "patterns": {}}
    all_patterns = dict(required_patterns(seed) if pattern_registry is None else pattern_registry)
    selected_names = list(all_patterns) if patterns is None else list(patterns)
    unknown = set(selected_names).difference(all_patterns)
    if unknown:
        raise ValueError(f"Unknown patterns requested: {sorted(unknown)}")
    prediction_files = []
    for name in selected_names:
        pattern = all_patterns[name]
        dataset = build_pattern_dataset(config, pattern, fill_mode=fill_mode, split=split, smoke_test=smoke_test)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if save_predictions:
            collected = collect_predictions(
                model,
                loader,
                device=device,
                forward_fn=lambda model_, batch_: forward_model(model_, batch_, device=device),
            )
            logits = collected["logits"]
            targets = collected["targets"]
        else:
            logits, targets = predict(model, loader, device=device)
        metrics = compute_multilabel_metrics(logits, targets, thresholds=thresholds)
        row = {
            "pattern": name,
            "fill_mode": fill_mode,
            "n": int(targets.shape[0]),
            "macro_auroc": metrics["macro_auroc"],
            "macro_auprc": metrics["macro_auprc"],
            "macro_f1": metrics["macro_f1"],
            "bce_nll": metrics["bce_nll"],
        }
        rows.append(row)
        details["patterns"][name] = {
            "pattern": pattern.to_dict(),
            "metrics": metrics,
        }
        if save_predictions:
            if predictions_dir is None:
                raise ValueError("predictions_dir is required when save_predictions=True")
            if method_id is None:
                raise ValueError("method_id is required when save_predictions=True")
            pred_rows = prediction_rows(
                method_id=method_id,
                pattern=name,
                fill_mode=fill_mode,
                split=split,
                random_seed=seed,
                threshold_source_split=threshold_source_split,
                thresholds=thresholds,
                collected=collected,
            )
            pred_path = build_prediction_output_path(
                predictions_dir,
                method_id=method_id,
                fill_mode=fill_mode,
                split=split,
                pattern=name,
            )
            file_info = save_predictions_csv(pred_path, pred_rows)
            prediction_files.append(
                {
                    "method_id": method_id,
                    "split": split,
                    "fill_mode": fill_mode,
                    "pattern": name,
                    "threshold_source_split": threshold_source_split,
                    "n_labels": len(LABEL_ORDER),
                    "has_logits": True,
                    "has_probabilities": True,
                    "has_thresholds": True,
                    **file_info,
                }
            )

    csv_path = output_dir / f"test_missing_patterns_{fill_mode}.csv"
    json_path = output_dir / f"test_missing_patterns_{fill_mode}.json"
    if write_metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["pattern", "fill_mode", "n", "macro_auroc", "macro_auprc", "macro_f1", "bce_nll"],
            )
            writer.writeheader()
            writer.writerows(rows)
        write_json(json_path, details)
    return {
        "fill_mode": fill_mode,
        "split": split,
        "csv": str(csv_path) if write_metrics else None,
        "json": str(json_path) if write_metrics else None,
        "rows": rows,
        "prediction_files": prediction_files,
    }
