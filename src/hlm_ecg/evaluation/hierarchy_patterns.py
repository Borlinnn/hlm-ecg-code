"""Missing-pattern evaluation with hierarchy diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from hlm_ecg.evaluation.evaluate_patterns import build_pattern_dataset, load_model_from_checkpoint, load_thresholds
from hlm_ecg.evaluation.metrics import compute_multilabel_metrics
from hlm_ecg.evaluation.missing_patterns import required_patterns
from hlm_ecg.losses.hierarchy import ParentChildHierarchyLoss, load_parent_indices
from hlm_ecg.training.train_baseline import resolve_device, write_json


def hierarchy_loss_from_config(config: Mapping[str, object]) -> ParentChildHierarchyLoss:
    paths = dict(config.get("paths", {}))
    vocab_path = paths.get("subclass_vocab")
    mapping_path = paths.get("subclass_parent_mapping")
    if not vocab_path or not mapping_path:
        raise RuntimeError("Hierarchy evaluation requires subclass_vocab and subclass_parent_mapping paths")
    hier_cfg = dict(config.get("hierarchy_loss", {}))
    parent_indices = load_parent_indices(vocab_path=vocab_path, mapping_path=mapping_path)
    return ParentChildHierarchyLoss(
        parent_indices,
        violation_eps=float(hier_cfg.get("hierarchy_violation_eps", 0.0)),
    )


@torch.no_grad()
def predict_with_hierarchy_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    hierarchy_loss: ParentChildHierarchyLoss,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    logits_all = []
    targets_all = []
    totals = {"hierarchy_loss": 0.0, "violation_rate": 0.0, "mean_violation_margin": 0.0}
    max_violation_margin = 0.0
    total = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        mask = batch.get("availability_mask", batch.get("lead_mask"))
        if mask is None:
            raise RuntimeError("Hierarchy evaluation requires availability_mask")
        outputs = model(x, availability_mask=mask.to(device=device, dtype=torch.float32))
        if not isinstance(outputs, Mapping) or "logits_super" not in outputs or "logits_sub" not in outputs:
            raise RuntimeError("Hierarchy evaluation requires model outputs with logits_super and logits_sub")
        diag = hierarchy_loss(outputs["logits_super"], outputs["logits_sub"])
        batch_size = int(x.shape[0])
        totals["hierarchy_loss"] += float(diag["loss_hier"].detach().item()) * batch_size
        totals["violation_rate"] += float(diag["violation_rate"].detach().item()) * batch_size
        totals["mean_violation_margin"] += float(diag["mean_violation_margin"].detach().item()) * batch_size
        max_violation_margin = max(max_violation_margin, float(diag["max_violation_margin"].detach().item()))
        total += batch_size
        logits_all.append(outputs["logits_super"].detach().cpu().numpy())
        targets_all.append(batch["y"].cpu().numpy())

    diagnostics = {key: value / max(total, 1) for key, value in totals.items()}
    diagnostics["max_violation_margin"] = max_violation_margin
    return np.concatenate(logits_all, axis=0), np.concatenate(targets_all, axis=0), diagnostics


def evaluate_hierarchy_patterns_in_memory(
    *,
    checkpoint_path: Path,
    config: Mapping[str, object],
    fill_mode: str,
    smoke_test: bool = False,
) -> dict[str, object]:
    if fill_mode not in {"zero_fill", "mean_fill"}:
        raise ValueError("fill_mode must be zero_fill or mean_fill")
    paths = dict(config.get("paths", {}))
    output_dir = Path(paths.get("output_dir", "outputs/week2_hierarchy_ablation"))
    device = resolve_device(str(config.get("device", "auto")))
    thresholds = load_thresholds(output_dir / "thresholds_val.json")
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    hierarchy_loss = hierarchy_loss_from_config(config).to(device)

    eval_cfg = dict(config.get("evaluation", {}))
    batch_size = int(eval_cfg.get("batch_size", dict(config.get("training", {})).get("batch_size", 64)))
    num_workers = int(eval_cfg.get("num_workers", dict(config.get("training", {})).get("num_workers", 0)))
    seed = int(eval_cfg.get("pattern_seed", 20240604))

    rows = []
    details: Dict[str, object] = {"fill_mode": fill_mode, "patterns": {}}
    for name, pattern in required_patterns(seed).items():
        dataset = build_pattern_dataset(config, pattern, fill_mode=fill_mode, smoke_test=smoke_test)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logits, targets, diagnostics = predict_with_hierarchy_diagnostics(
            model,
            loader,
            device=device,
            hierarchy_loss=hierarchy_loss,
        )
        metrics = compute_multilabel_metrics(logits, targets, thresholds=thresholds)
        row = {
            "pattern": name,
            "fill_mode": fill_mode,
            "n": int(targets.shape[0]),
            "macro_auroc": metrics["macro_auroc"],
            "macro_auprc": metrics["macro_auprc"],
            "macro_f1": metrics["macro_f1"],
            "bce_nll": metrics["bce_nll"],
            **diagnostics,
        }
        rows.append(row)
        details["patterns"][name] = {
            "pattern": pattern.to_dict(),
            "metrics": metrics,
            "hierarchy_diagnostics": diagnostics,
        }
    return {"fill_mode": fill_mode, "rows": rows, "details": details}


def evaluate_hierarchy_missing_patterns(
    *,
    checkpoint_path: Path,
    config: Mapping[str, object],
    fill_mode: str,
    smoke_test: bool = False,
) -> dict[str, object]:
    paths = dict(config.get("paths", {}))
    output_dir = Path(paths.get("output_dir", "outputs/week2_hierarchy_ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_hierarchy_patterns_in_memory(
        checkpoint_path=checkpoint_path,
        config=config,
        fill_mode=fill_mode,
        smoke_test=smoke_test,
    )
    rows = result["rows"]
    csv_path = output_dir / f"test_missing_patterns_{fill_mode}.csv"
    json_path = output_dir / f"test_missing_patterns_{fill_mode}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pattern",
                "fill_mode",
                "n",
                "macro_auroc",
                "macro_auprc",
                "macro_f1",
                "bce_nll",
                "hierarchy_loss",
                "violation_rate",
                "mean_violation_margin",
                "max_violation_margin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, result["details"])
    return {
        "fill_mode": fill_mode,
        "csv": str(csv_path),
        "json": str(json_path),
        "rows": rows,
    }
