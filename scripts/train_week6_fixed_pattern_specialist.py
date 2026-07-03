#!/usr/bin/env python3
"""Train a gated Week 6 fixed-pattern specialist baseline.

This script is intentionally blocked unless `WEEK6_ALLOW_SPECIALIST_TRAINING=true`.
It exists so the reviewer-defense package has an implemented specialist baseline
path without accidentally launching new training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hlm_ecg.evaluation.metrics import compute_multilabel_metrics, tune_thresholds_on_validation
from hlm_ecg.evaluation.week6_defense import (
    ROOT,
    Week6ImputationDataset,
    selected_patterns,
    specialist_training_allowed,
)
from hlm_ecg.training.train_baseline import (
    build_model,
    forward_model,
    json_default,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gated Week 6 fixed-pattern specialist.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write gate/status metadata without training.")
    return parser.parse_args()


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_train_log(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "val_loss", "val_macro_auprc", "val_macro_auroc", "val_macro_f1"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _make_loader(dataset, config: Mapping[str, Any], *, train: bool) -> DataLoader:
    training = dict(config.get("training", {}))
    batch_size = int(training.get("batch_size", 64))
    num_workers = int(training.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)) + (1 if train else 1000))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, generator=generator)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> tuple[Any, Any]:
    model.eval()
    logits_all = []
    targets_all = []
    for batch in loader:
        logits_all.append(forward_model(model, batch, device=device).detach().cpu())
        targets_all.append(batch["y"].detach().cpu())
    return torch.cat(logits_all, dim=0).numpy(), torch.cat(targets_all, dim=0).numpy()


def run_epoch(model, loader, *, device: torch.device, criterion, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    for batch in loader:
        y = batch["y"].to(device=device, dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch, device=device)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * int(y.shape[0])
        total += int(y.shape[0])
    return total_loss / max(total, 1)


def build_specialist_dataset(config: Mapping[str, Any], split: str, *, smoke_test: bool):
    spec = dict(config.get("week6_specialist", {}))
    pattern_name = str(spec.get("pattern", ""))
    if not pattern_name:
        raise RuntimeError("week6_specialist.pattern is required")
    patterns = selected_patterns([pattern_name])
    return Week6ImputationDataset(
        config=config,
        split=split,
        pattern=patterns[pattern_name],
        imputation_strategy=str(spec.get("imputation_strategy", "mean_fill")),
        smoke_test=smoke_test,
    )


def train_specialist(config: Mapping[str, Any], *, max_epochs: int | None, smoke_test: bool) -> dict[str, Any]:
    output_dir = Path(dict(config.get("paths", {})).get("output_dir", "outputs/week6_reviewer_defense/fixed_pattern_specialists"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    train_ds = build_specialist_dataset(config, "train", smoke_test=smoke_test)
    val_ds = build_specialist_dataset(config, "val", smoke_test=smoke_test)
    test_ds = build_specialist_dataset(config, "test", smoke_test=smoke_test)
    train_loader = _make_loader(train_ds, config, train=True)
    val_loader = _make_loader(val_ds, config, train=False)
    test_loader = _make_loader(test_ds, config, train=False)

    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    training_cfg = dict(config.get("training", {}))
    epochs = int(max_epochs if max_epochs is not None else training_cfg.get("max_epochs", 30))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    patience = int(training_cfg.get("early_stopping_patience", 8))
    best_score = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    log_rows = []
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device=device, criterion=criterion, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device=device, criterion=criterion, optimizer=None)
        val_logits, val_targets = predict_logits(model, val_loader, device=device)
        val_metrics = compute_multilabel_metrics(val_logits, val_targets)
        val_score = val_metrics["macro_auprc"] or -float("inf")
        scheduler.step(float(val_score))
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auprc": val_metrics["macro_auprc"],
                "val_macro_auroc": val_metrics["macro_auroc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
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
    val_logits, val_targets = predict_logits(model, val_loader, device=device)
    threshold_info = tune_thresholds_on_validation(val_logits, val_targets)
    thresholds = threshold_info["threshold_array"]
    val_metrics = compute_multilabel_metrics(val_logits, val_targets, thresholds=thresholds)
    test_logits, test_targets = predict_logits(model, test_loader, device=device)
    test_metrics = compute_multilabel_metrics(test_logits, test_targets, thresholds=thresholds)
    write_train_log(output_dir / "train_log.csv", log_rows)
    write_json(output_dir / "thresholds_val.json", threshold_info)
    write_json(output_dir / "val_metrics.json", val_metrics)
    write_json(output_dir / "test_full_metrics.json", test_metrics)
    return {
        "output_dir": str(output_dir),
        "best_model": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_macro_auprc": best_score,
        "device": str(device),
        "smoke_test": bool(smoke_test),
        "records500_used": False,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = str(args.output_dir)
    output_dir = Path(dict(config.get("paths", {})).get("output_dir", "outputs/week6_reviewer_defense/fixed_pattern_specialists"))
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_status = {
        "specialist_training_allowed": specialist_training_allowed(),
        "dry_run": bool(args.dry_run),
        "gate": "WEEK6_ALLOW_SPECIALIST_TRAINING=true",
        "config": str(args.config),
        "output_dir": str(output_dir),
        "records500_used": False,
    }
    write_json(output_dir / "specialist_training_gate_status.json", gate_status)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if args.dry_run or not specialist_training_allowed():
        print(json.dumps({**gate_status, "status": "training_not_started"}, indent=2))
        return
    result = train_specialist(config, max_epochs=args.max_epochs, smoke_test=args.smoke_test)
    print(json.dumps({**gate_status, "status": "trained", "result": result}, indent=2))


if __name__ == "__main__":
    main()
