"""Training loop for A4a subclass auxiliary ablation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.data.subclass_labels import load_subclass_vocab, write_kept_subclass_artifacts
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.evaluation.metrics import compute_multilabel_metrics, tune_thresholds_on_validation
from hlm_ecg.losses.hierarchy import load_parent_indices
from hlm_ecg.losses.subclass_auxiliary import SubclassAuxiliaryLoss, SubclassAuxiliaryLossConfig
from hlm_ecg.training.train_baseline import (
    build_model,
    build_train_lead_mask_sampler,
    forward_model,
    json_default,
    make_loader,
    model_requires_availability_mask,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def prepare_subclass_artifacts(config: Mapping[str, object]) -> dict[str, Path]:
    paths = dict(config.get("paths", {}))
    sub_cfg = dict(config.get("subclass_auxiliary", {}))
    seed = int(config.get("seed", 42))
    output_dir = Path(paths.get("output_dir", "outputs/week2_subclass_auxiliary/subclass_aux_seed42"))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = write_kept_subclass_artifacts(
        root=Path(paths.get("data_root", "data/ptb-xl")),
        day1_index=Path(paths.get("day1_index", "outputs/day1_audit/ptbxl_day1_index.csv")),
        output_dir=output_dir,
        min_train_pos=int(sub_cfg.get("min_train_pos", 50)),
        seed=seed,
    )
    return artifact_paths


def build_subclass_dataset(config: Mapping[str, object], split: str, *, smoke_test: bool = False) -> PTBXLDataset:
    paths = dict(config.get("paths", {}))
    smoke = dict(config.get("smoke", {}))
    seed = int(config.get("seed", 42))
    limit = int(smoke.get(f"{split}_limit", 64)) if smoke_test else None
    train_mask_sampler = None
    fill_mode = "full"
    if split == "train":
        train_mask_sampler, fill_mode = build_train_lead_mask_sampler(config, seed=seed)
    return PTBXLDataset(
        root=Path(paths.get("data_root", "data/ptb-xl")),
        index_csv=Path(paths.get("day1_index", "outputs/day1_audit/ptbxl_day1_index.csv")),
        norm_stats_path=Path(paths.get("norm_stats", "outputs/day1_audit/train_norm_stats.npz")),
        split=split,
        fill_mode=fill_mode,
        lead_mask_sampler=train_mask_sampler,
        subclass_index_csv=Path(paths["subclass_index"]),
        subclass_vocab_path=Path(paths["subclass_vocab"]),
        limit=limit,
    )


def forward_multitask(model: nn.Module, batch: Mapping[str, object], *, device: torch.device) -> Mapping[str, torch.Tensor]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    if model_requires_availability_mask(model):
        mask = batch.get("availability_mask", batch.get("lead_mask"))
        if mask is None:
            raise RuntimeError("Multitask model requires availability_mask")
        outputs = model(x, availability_mask=mask.to(device=device, dtype=torch.float32))
    else:
        outputs = model(x)
    if not isinstance(outputs, Mapping):
        raise RuntimeError("Multitask model must return logits_super and logits_sub")
    return outputs


def build_subclass_criterion(config: Mapping[str, object]) -> SubclassAuxiliaryLoss:
    sub_cfg = dict(config.get("subclass_auxiliary", {}))
    model_cfg = dict(config.get("model", {}))
    hier_cfg = dict(config.get("hierarchy_loss", {}))
    use_hierarchy = bool(model_cfg.get("use_hierarchy_loss", False)) or bool(hier_cfg.get("enabled", False))
    parent_indices = None
    if use_hierarchy:
        if not bool(model_cfg.get("enable_subclass_auxiliary", False)):
            raise RuntimeError("use_hierarchy_loss=true requires model.enable_subclass_auxiliary=true")
        paths = dict(config.get("paths", {}))
        vocab_path = paths.get("subclass_vocab")
        mapping_path = paths.get("subclass_parent_mapping")
        if not vocab_path or not mapping_path:
            raise RuntimeError("use_hierarchy_loss=true requires subclass_vocab and subclass_parent_mapping paths")
        parent_indices = load_parent_indices(vocab_path=vocab_path, mapping_path=mapping_path)

    return SubclassAuxiliaryLoss(
        SubclassAuxiliaryLossConfig(
            lambda_sub=float(sub_cfg.get("lambda_sub", 0.2)),
            ignore_only_dropped=bool(sub_cfg.get("subclass_loss_ignore_only_dropped", True)),
            use_hierarchy_loss=use_hierarchy,
            lambda_hier=float(hier_cfg.get("lambda_hier", model_cfg.get("lambda_hier", 0.0))),
            hierarchy_parent_indices=parent_indices,
            hierarchy_violation_eps=float(hier_cfg.get("hierarchy_violation_eps", 0.0)),
        )
    )


def run_subclass_epoch(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: SubclassAuxiliaryLoss,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    maxima: dict[str, float] = {}
    total = 0
    for batch in loader:
        batch_size = int(batch["x"].shape[0])
        batch_device = {
            "y": batch["y"].to(device=device, dtype=torch.float32),
            "y_sub": batch["y_sub"].to(device=device, dtype=torch.float32),
            "has_only_dropped_subclass": batch["has_only_dropped_subclass"].to(device=device, dtype=torch.float32),
        }
        if train:
            optimizer.zero_grad(set_to_none=True)
        outputs = forward_multitask(model, batch, device=device)
        loss_info = criterion(outputs, batch_device)
        if train:
            loss_info["loss"].backward()
            optimizer.step()
        for key, value in loss_info.items():
            scalar = float(value.detach().item())
            if key == "max_violation_margin":
                maxima[key] = max(maxima.get(key, -float("inf")), scalar)
            else:
                totals[key] = totals.get(key, 0.0) + scalar * batch_size
        total += batch_size
    out = {key: value / max(total, 1) for key, value in totals.items()}
    out.update(maxima)
    return out


@torch.no_grad()
def predict_super_logits(model: nn.Module, loader, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    targets_all = []
    for batch in loader:
        logits = forward_model(model, batch, device=device).detach().cpu().numpy()
        logits_all.append(logits)
        targets_all.append(batch["y"].cpu().numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(targets_all, axis=0)


def write_subclass_train_log(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "epoch",
        "train_loss",
        "train_loss_super",
        "train_loss_sub",
        "val_loss",
        "val_loss_super",
        "val_loss_sub",
        "val_macro_auprc",
        "val_macro_auroc",
        "val_macro_f1",
    ]
    optional_fieldnames = [
        "train_loss_hier",
        "val_loss_hier",
        "train_violation_rate",
        "val_violation_rate",
        "train_mean_violation_margin",
        "val_mean_violation_margin",
        "train_max_violation_margin",
        "val_max_violation_margin",
    ]
    fieldnames = base_fieldnames + [key for key in optional_fieldnames if any(key in row for row in rows)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def train_subclass_auxiliary(
    config: Mapping[str, object],
    *,
    max_epochs: int | None = None,
    smoke_test: bool = False,
) -> Dict[str, object]:
    output_dir = Path(dict(config.get("paths", {})).get("output_dir", "outputs/week2_subclass_auxiliary/subclass_aux_seed42"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))

    artifact_paths = prepare_subclass_artifacts(config)
    cfg_paths = dict(config.get("paths", {}))
    cfg_paths.update({key: str(path) for key, path in artifact_paths.items()})
    config = dict(config)
    config["paths"] = cfg_paths
    vocab = load_subclass_vocab(cfg_paths["subclass_vocab"])
    model_cfg = dict(config.get("model", {}))
    model_cfg["num_subclasses"] = int(vocab["num_subclasses"])
    config["model"] = model_cfg

    train_ds = build_subclass_dataset(config, "train", smoke_test=smoke_test)
    val_ds = build_subclass_dataset(config, "val", smoke_test=smoke_test)
    test_ds = build_subclass_dataset(config, "test", smoke_test=smoke_test)
    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)
    test_loader = make_loader(test_ds, config, train=False)

    model = build_model(config).to(device)
    criterion = build_subclass_criterion(config)
    training = dict(config.get("training", {}))
    epochs = int(max_epochs if max_epochs is not None else training.get("max_epochs", 30))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler_name = str(training.get("scheduler", "reduce_on_plateau"))
    scheduler = None
    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_score = -float("inf")
    best_epoch = -1
    patience = int(training.get("early_stopping_patience", 8))
    epochs_without_improvement = 0
    log_rows = []
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_losses = run_subclass_epoch(model, train_loader, device=device, criterion=criterion, optimizer=optimizer)
        val_losses = run_subclass_epoch(model, val_loader, device=device, criterion=criterion, optimizer=None)
        val_logits, val_targets = predict_super_logits(model, val_loader, device=device)
        val_metrics = compute_multilabel_metrics(val_logits, val_targets)
        val_score = val_metrics["macro_auprc"] if val_metrics["macro_auprc"] is not None else -float("inf")
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(float(val_score))
        elif scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "val_macro_auprc": val_metrics["macro_auprc"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        row.update({f"train_{key}": value for key, value in train_losses.items()})
        row.update({f"val_{key}": value for key, value in val_losses.items()})
        log_rows.append(row)
        if best_epoch < 0 or float(val_score) > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model=model, config=config, epoch=epoch, val_macro_auprc=val_metrics["macro_auprc"])
        else:
            epochs_without_improvement += 1
        if not smoke_test and epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_logits, val_targets = predict_super_logits(model, val_loader, device=device)
    threshold_info = tune_thresholds_on_validation(val_logits, val_targets)
    thresholds = threshold_info["threshold_array"]
    val_metrics = compute_multilabel_metrics(val_logits, val_targets, thresholds=thresholds)
    test_logits, test_targets = predict_super_logits(model, test_loader, device=device)
    test_metrics = compute_multilabel_metrics(test_logits, test_targets, thresholds=thresholds)

    write_subclass_train_log(output_dir / "train_log.csv", log_rows)
    write_json(output_dir / "thresholds_val.json", threshold_info)
    write_json(output_dir / "val_metrics.json", val_metrics)
    write_json(output_dir / "test_full_metrics.json", test_metrics)
    return {
        "output_dir": str(output_dir),
        "best_model": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_macro_auprc": best_score,
        "subclass_vocab": str(cfg_paths["subclass_vocab"]),
        "subclass_parent_mapping": str(cfg_paths["subclass_parent_mapping"]),
        "subclass_index": str(cfg_paths["subclass_index"]),
        "num_subclasses": int(vocab["num_subclasses"]),
        "train_records": len(train_ds),
        "val_records": len(val_ds),
        "test_records": len(test_ds),
        "device": str(device),
        "smoke_test": bool(smoke_test),
    }
