"""Training loop for A5 confidence-weighted consistency ablation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hlm_ecg.data.subclass_labels import load_subclass_vocab
from hlm_ecg.datasets.paired_views import PairedFullMaskedPTBXLDataset
from hlm_ecg.evaluation.metrics import compute_multilabel_metrics, tune_thresholds_on_validation
from hlm_ecg.losses.confidence_consistency import (
    ConfidenceConsistencyLossConfig,
    ConfidenceWeightedConsistencyLoss,
)
from hlm_ecg.losses.hierarchy import ParentChildHierarchyLoss, load_parent_indices
from hlm_ecg.losses.subclass_auxiliary import SubclassAuxiliaryLoss, SubclassAuxiliaryLossConfig
from hlm_ecg.training.train_baseline import (
    build_model,
    make_loader,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_json,
)
from hlm_ecg.training.train_subclass_auxiliary import (
    build_subclass_dataset,
    forward_multitask,
    predict_super_logits,
    prepare_subclass_artifacts,
    run_subclass_epoch,
)


def _subclass_loss(
    logits_sub: torch.Tensor,
    y_sub: torch.Tensor,
    has_only_dropped_subclass: torch.Tensor,
    *,
    ignore_only_dropped: bool,
) -> torch.Tensor:
    per_entry = F.binary_cross_entropy_with_logits(logits_sub, y_sub, reduction="none")
    per_sample = per_entry.mean(dim=1)
    if not ignore_only_dropped:
        return per_sample.mean()
    weights = (1.0 - has_only_dropped_subclass.to(device=per_sample.device, dtype=torch.float32)).clamp(0.0, 1.0)
    denom = weights.sum().clamp(min=1.0)
    return (per_sample * weights).sum() / denom


def forward_paired(model: nn.Module, batch: Mapping[str, object], *, device: torch.device) -> dict[str, Mapping[str, torch.Tensor]]:
    x_full = batch["x_full"].to(device=device, dtype=torch.float32)
    x_mask = batch["x_mask"].to(device=device, dtype=torch.float32)
    mask_full = batch["availability_mask_full"].to(device=device, dtype=torch.float32)
    mask_mask = batch["availability_mask_mask"].to(device=device, dtype=torch.float32)
    outputs_full = model(x_full, availability_mask=mask_full)
    outputs_mask = model(x_mask, availability_mask=mask_mask)
    if not isinstance(outputs_full, Mapping) or not isinstance(outputs_mask, Mapping):
        raise RuntimeError("A5 requires model outputs with logits_super and logits_sub")
    for name, outputs in (("full", outputs_full), ("mask", outputs_mask)):
        if "logits_super" not in outputs or "logits_sub" not in outputs:
            raise RuntimeError(f"A5 {name} view requires logits_super and logits_sub")
    return {"full": outputs_full, "mask": outputs_mask}


def run_consistency_epoch(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    consistency_loss: ConfidenceWeightedConsistencyLoss,
    lambda_sub: float,
    lambda_cons: float,
    hierarchy_loss: ParentChildHierarchyLoss | None = None,
    lambda_hier: float = 0.0,
    ignore_only_dropped: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    bce = nn.BCEWithLogitsLoss()
    totals: dict[str, float] = {}
    minima: dict[str, float] = {}
    maxima: dict[str, float] = {}
    total = 0
    for batch in loader:
        batch_size = int(batch["x_full"].shape[0])
        y = batch["y"].to(device=device, dtype=torch.float32)
        y_sub = batch["y_sub"].to(device=device, dtype=torch.float32)
        dropped = batch["has_only_dropped_subclass"].to(device=device, dtype=torch.float32)
        if train:
            optimizer.zero_grad(set_to_none=True)
        outputs = forward_paired(model, batch, device=device)
        full = outputs["full"]
        masked = outputs["mask"]
        loss_super_full = bce(full["logits_super"], y)
        loss_super_mask = bce(masked["logits_super"], y)
        loss_sub_full = _subclass_loss(
            full["logits_sub"],
            y_sub,
            dropped,
            ignore_only_dropped=ignore_only_dropped,
        )
        loss_sub_mask = _subclass_loss(
            masked["logits_sub"],
            y_sub,
            dropped,
            ignore_only_dropped=ignore_only_dropped,
        )
        cons_info = consistency_loss(full["logits_super"], masked["logits_super"])
        hier_values: dict[str, torch.Tensor] = {}
        hier_loss_total = torch.zeros((), device=y.device, dtype=loss_super_full.dtype)
        if hierarchy_loss is not None and float(lambda_hier) > 0.0:
            hier_full = hierarchy_loss(full["logits_super"], full["logits_sub"])
            hier_mask = hierarchy_loss(masked["logits_super"], masked["logits_sub"])
            hier_loss_total = hier_full["loss_hier"] + hier_mask["loss_hier"]
            hier_values = {
                "loss_hier_full": hier_full["loss_hier"],
                "loss_hier_mask": hier_mask["loss_hier"],
                "hier_violation_rate_full": hier_full["violation_rate"],
                "hier_violation_rate_mask": hier_mask["violation_rate"],
                "hier_mean_violation_margin_full": hier_full["mean_violation_margin"],
                "hier_mean_violation_margin_mask": hier_mask["mean_violation_margin"],
                "hier_max_violation_margin_full": hier_full["max_violation_margin"],
                "hier_max_violation_margin_mask": hier_mask["max_violation_margin"],
            }
        total_loss = (
            loss_super_full
            + loss_super_mask
            + float(lambda_sub) * (loss_sub_full + loss_sub_mask)
            + float(lambda_hier) * hier_loss_total
            + float(lambda_cons) * cons_info["cw_consistency_loss"]
        )
        if train:
            total_loss.backward()
            optimizer.step()
        values = {
            "loss": total_loss,
            "loss_super_full": loss_super_full,
            "loss_super_mask": loss_super_mask,
            "loss_sub_full": loss_sub_full,
            "loss_sub_mask": loss_sub_mask,
            "cw_consistency_loss": cons_info["cw_consistency_loss"],
            "mean_consistency_weight": cons_info["mean_consistency_weight"],
            "full_mean_confidence": cons_info["full_mean_confidence"],
            "masked_mean_confidence": cons_info["masked_mean_confidence"],
        }
        values.update(hier_values)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_size
        for key in ("min_consistency_weight",):
            scalar = float(cons_info[key].detach().item())
            minima[key] = min(minima.get(key, float("inf")), scalar)
        for key in ("max_consistency_weight",):
            scalar = float(cons_info[key].detach().item())
            maxima[key] = max(maxima.get(key, -float("inf")), scalar)
        total += batch_size
    out = {key: value / max(total, 1) for key, value in totals.items()}
    out.update(minima)
    out.update(maxima)
    return out


def write_consistency_train_log(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_loss_super_full",
        "train_loss_super_mask",
        "train_loss_sub_full",
        "train_loss_sub_mask",
        "train_cw_consistency_loss",
        "train_mean_consistency_weight",
        "train_min_consistency_weight",
        "train_max_consistency_weight",
        "train_full_mean_confidence",
        "train_masked_mean_confidence",
        "val_loss",
        "val_loss_super",
        "val_loss_sub",
        "val_macro_auprc",
        "val_macro_auroc",
        "val_macro_f1",
    ]
    optional = [
        "train_loss_hier_full",
        "train_loss_hier_mask",
        "train_hier_violation_rate_full",
        "train_hier_violation_rate_mask",
        "train_hier_mean_violation_margin_full",
        "train_hier_mean_violation_margin_mask",
        "train_hier_max_violation_margin_full",
        "train_hier_max_violation_margin_mask",
        "val_loss_hier",
        "val_violation_rate",
        "val_mean_violation_margin",
        "val_max_violation_margin",
    ]
    fieldnames = fieldnames + [key for key in optional if any(key in row for row in rows)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def train_confidence_consistency(
    config: Mapping[str, object],
    *,
    max_epochs: int | None = None,
    smoke_test: bool = False,
) -> Dict[str, object]:
    output_dir = Path(dict(config.get("paths", {})).get("output_dir", "outputs/week2_confidence_consistency/consistency_seed42"))
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

    train_base = build_subclass_dataset(config, "train", smoke_test=smoke_test)
    train_ds = PairedFullMaskedPTBXLDataset(train_base)
    val_ds = build_subclass_dataset(config, "val", smoke_test=smoke_test)
    test_ds = build_subclass_dataset(config, "test", smoke_test=smoke_test)
    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)
    test_loader = make_loader(test_ds, config, train=False)

    model = build_model(config).to(device)
    sub_cfg = dict(config.get("subclass_auxiliary", {}))
    cons_cfg = dict(config.get("confidence_consistency", {}))
    hier_cfg = dict(config.get("hierarchy_loss", {}))
    lambda_sub = float(sub_cfg.get("lambda_sub", 0.2))
    lambda_cons = float(cons_cfg.get("lambda_cons", model_cfg.get("lambda_cons", 0.1)))
    use_hierarchy = bool(model_cfg.get("use_hierarchy_loss", False)) or bool(hier_cfg.get("enabled", False))
    lambda_hier = float(hier_cfg.get("lambda_hier", model_cfg.get("lambda_hier", 0.0)))
    parent_indices = None
    hierarchy_loss = None
    if use_hierarchy:
        parent_indices = load_parent_indices(
            vocab_path=cfg_paths["subclass_vocab"],
            mapping_path=cfg_paths["subclass_parent_mapping"],
        )
        hierarchy_loss = ParentChildHierarchyLoss(
            parent_indices,
            violation_eps=float(hier_cfg.get("hierarchy_violation_eps", 0.0)),
        ).to(device)
    consistency_loss = ConfidenceWeightedConsistencyLoss(
        ConfidenceConsistencyLossConfig(
            lambda_cons=lambda_cons,
            gamma=float(cons_cfg.get("consistency_gamma", 1.0)),
            enabled=bool(cons_cfg.get("enabled", model_cfg.get("use_confidence_weighted_consistency", True))),
        )
    )
    val_criterion = SubclassAuxiliaryLoss(
        SubclassAuxiliaryLossConfig(
            lambda_sub=lambda_sub,
            ignore_only_dropped=bool(sub_cfg.get("subclass_loss_ignore_only_dropped", True)),
            use_hierarchy_loss=use_hierarchy,
            lambda_hier=lambda_hier,
            hierarchy_parent_indices=parent_indices,
            hierarchy_violation_eps=float(hier_cfg.get("hierarchy_violation_eps", 0.0)),
        )
    )

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
        train_losses = run_consistency_epoch(
            model,
            train_loader,
            device=device,
            consistency_loss=consistency_loss,
            lambda_sub=lambda_sub,
            lambda_cons=lambda_cons,
            hierarchy_loss=hierarchy_loss,
            lambda_hier=lambda_hier,
            ignore_only_dropped=bool(sub_cfg.get("subclass_loss_ignore_only_dropped", True)),
            optimizer=optimizer,
        )
        val_losses = run_subclass_epoch(model, val_loader, device=device, criterion=val_criterion, optimizer=None)
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

    write_consistency_train_log(output_dir / "train_log.csv", log_rows)
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
