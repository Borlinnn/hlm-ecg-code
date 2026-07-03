"""Full-lead supervised baseline training."""

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hlm_ecg.data.lead_dropout import build_random_lead_dropout
from hlm_ecg.data.lead_masking import LeadMaskSampler, build_structured_lead_masking
from hlm_ecg.data.ptbxl_labels import LABEL_ORDER
from hlm_ecg.datasets.ptbxl_dataset import PTBXLDataset
from hlm_ecg.evaluation.metrics import compute_multilabel_metrics, tune_thresholds_on_validation
from hlm_ecg.models.backbones import build_feature_backbone
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability
from hlm_ecg.models.task_wrappers import BackboneSubclassClassifier, MaskTokenBackboneClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_model(config: Mapping[str, object]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    if bool(model_cfg.get("use_availability_embedding", False)):
        return ResNet1DAvailability(
            architecture=str(model_cfg.get("architecture", "resnet1d_tiny")),
            in_channels=int(model_cfg.get("in_channels", 12)),
            num_classes=int(model_cfg.get("num_classes", 5)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            layers=tuple(model_cfg.get("layers", [1, 1, 1, 1])),
            kernel_size=int(model_cfg.get("kernel_size", 7)),
            availability_embedding_dim=int(model_cfg.get("availability_embedding_dim", 32)),
            mask_mlp_hidden_dim=int(model_cfg.get("mask_mlp_hidden_dim", 32)),
            use_subclass_auxiliary=bool(model_cfg.get("enable_subclass_auxiliary", False)),
            num_subclasses=(
                None
                if model_cfg.get("num_subclasses") is None
                else int(model_cfg.get("num_subclasses"))
            ),
            inception_depth=int(model_cfg.get("inception_depth", 6)),
            inception_bottleneck_channels=(
                None
                if model_cfg.get("inception_bottleneck_channels") is None
                else int(model_cfg.get("inception_bottleneck_channels"))
            ),
            use_learnable_mask_token=bool(model_cfg.get("use_learnable_mask_token", False)),
            signal_length=int(model_cfg.get("signal_length", 1000)),
        )
    if bool(model_cfg.get("enable_subclass_auxiliary", False)):
        return BackboneSubclassClassifier(model_cfg=model_cfg)
    if bool(model_cfg.get("use_learnable_mask_token", False)):
        return MaskTokenBackboneClassifier(model_cfg=model_cfg)
    return build_feature_backbone(model_cfg)


def model_requires_availability_mask(model: nn.Module) -> bool:
    return bool(getattr(model, "requires_availability_mask", False))


def forward_model(model: nn.Module, batch: Mapping[str, object], *, device: torch.device) -> torch.Tensor:
    x = batch["x"].to(device=device, dtype=torch.float32)
    if model_requires_availability_mask(model):
        mask = batch.get("availability_mask", batch.get("lead_mask"))
        if mask is None:
            raise RuntimeError("Availability model requires batch['availability_mask'] or batch['lead_mask']")
        availability_mask = mask.to(device=device, dtype=torch.float32)
        output = model(x, availability_mask=availability_mask)
    else:
        output = model(x)
    if isinstance(output, Mapping):
        return output["logits_super"]
    return output


def build_train_lead_mask_sampler(config: Mapping[str, object], *, seed: int) -> tuple[LeadMaskSampler | None, str]:
    """Build the optional train-time mask sampler without making the model mask-aware."""
    random_enabled = bool(dict(config.get("train_augmentation", {})).get("enabled", False))
    structured_enabled = bool(dict(config.get("structured_masking", {})).get("enabled", False))
    if random_enabled and structured_enabled:
        raise ValueError("Enable only one of train_augmentation or structured_masking")

    if structured_enabled:
        structured_cfg = dict(config.get("structured_masking", {}))
        sampler = build_structured_lead_masking(config, seed=seed)
        return sampler, str(structured_cfg.get("fill_mode", "mean_fill"))

    if random_enabled:
        aug = dict(config.get("train_augmentation", {}))
        sampler = build_random_lead_dropout(config, seed=seed)
        return sampler, str(aug.get("fill_mode", "mean_fill"))

    return None, "full"


def build_dataset(config: Mapping[str, object], split: str, *, smoke_test: bool = False) -> PTBXLDataset:
    paths = dict(config.get("paths", {}))
    smoke = dict(config.get("smoke", {}))
    seed = int(config.get("seed", 20240604))
    limit = None
    if smoke_test:
        limit = int(smoke.get(f"{split}_limit", 64))
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
        limit=limit,
    )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "set_random_seed"):
        info.dataset.set_random_seed(worker_seed + worker_id)


def make_loader(dataset: PTBXLDataset, config: Mapping[str, object], *, train: bool) -> DataLoader:
    training = dict(config.get("training", {}))
    batch_size = int(training.get("batch_size", 64))
    num_workers = int(training.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 20240604)) + (1 if train else 1000))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        if train:
            optimizer.zero_grad(set_to_none=True)
        if model_requires_availability_mask(model):
            mask = batch.get("availability_mask", batch.get("lead_mask"))
            if mask is None:
                raise RuntimeError("Availability model requires batch['availability_mask'] or batch['lead_mask']")
            logits = model(x, availability_mask=mask.to(device=device, dtype=torch.float32))
        else:
            logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * int(x.shape[0])
        total += int(x.shape[0])
    return total_loss / max(total, 1)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    targets_all = []
    for batch in loader:
        y = batch["y"].cpu().numpy()
        logits = forward_model(model, batch, device=device).detach().cpu().numpy()
        logits_all.append(logits)
        targets_all.append(y)
    return np.concatenate(logits_all, axis=0), np.concatenate(targets_all, axis=0)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    config: Mapping[str, object],
    epoch: int,
    val_macro_auprc: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "epoch": int(epoch),
            "val_macro_auprc": val_macro_auprc,
            "label_order": list(LABEL_ORDER),
        },
        path,
    )


def write_train_log(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss", "val_macro_auprc", "val_macro_auroc", "val_macro_f1"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def train_full_baseline(
    config: Mapping[str, object],
    *,
    max_epochs: int | None = None,
    smoke_test: bool = False,
) -> Dict[str, object]:
    output_dir = Path(dict(config.get("paths", {})).get("output_dir", "outputs/week1_full_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 20240604)))
    device = resolve_device(str(config.get("device", "auto")))

    train_ds = build_dataset(config, "train", smoke_test=smoke_test)
    val_ds = build_dataset(config, "val", smoke_test=smoke_test)
    test_ds = build_dataset(config, "test", smoke_test=smoke_test)
    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)
    test_loader = make_loader(test_ds, config, train=False)

    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
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
        train_loss = run_epoch(model, train_loader, device=device, criterion=criterion, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device=device, criterion=criterion, optimizer=None)
        val_logits, val_targets = predict_logits(model, val_loader, device=device)
        val_metrics = compute_multilabel_metrics(val_logits, val_targets)
        val_score = val_metrics["macro_auprc"]
        if val_score is None:
            val_score = -float("inf")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(float(val_score))
        elif scheduler is not None:
            scheduler.step()

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
        "train_records": len(train_ds),
        "val_records": len(val_ds),
        "test_records": len(test_ds),
        "device": str(device),
        "smoke_test": bool(smoke_test),
    }
