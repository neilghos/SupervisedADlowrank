"""Jointly train and evaluate SpatialAD with patch-density ablations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryROC
from tqdm.auto import tqdm

from src.data import make_supervised_vad_dataset
from src.model import SpatialAD
from src.torchdata import VADTimeSeriesDataset, fit_sensor_reference_stats, resolve_dataset_root, select_training_samples


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def stratified_split(samples, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    labels = samples["label_index"].to_numpy()
    train_indices, val_indices = [], []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_val = min(max(1, int(round(len(indices) * val_fraction))), len(indices) - 1)
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return samples.iloc[train_indices].reset_index(drop=True), samples.iloc[val_indices].reset_index(drop=True)


def make_loader(samples, *, timestamps, batch_size, num_workers, shuffle):
    usable = (len(samples) // timestamps) * timestamps
    if usable < timestamps:
        raise ValueError(f"Split has {len(samples)} images; timestamps={timestamps}")
    dataset = VADTimeSeriesDataset(samples.iloc[:usable].reset_index(drop=True), image_size=255, timestamps=timestamps, drop_last=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    return dataset, loader


def bpr_loss(logits: Tensor, labels: Tensor) -> Tensor:
    logits = logits.flatten()
    binary = labels.flatten().bool()
    positive, negative = logits[binary], logits[~binary]
    if positive.numel() == 0 or negative.numel() == 0:
        return F.binary_cross_entropy_with_logits(logits, binary.float())
    return -F.logsigmoid(positive[:, None] - negative[None, :]).mean()


def classification_loss(logits: Tensor, labels: Tensor, loss_type: str, pos_weight: Tensor) -> Tensor:
    if loss_type == "bpr":
        return bpr_loss(logits, labels)
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)


@torch.inference_mode()
def collect_predictions(model, loader, device, show_progress=False):
    model.eval()
    scores, labels = [], []
    iterator = tqdm(loader, desc="validation", unit="batch", leave=False) if show_progress else loader
    for features, target in iterator:
        output = model(features.to(device, non_blocking=True))
        scores.append(output.logits.detach().flatten().cpu())
        labels.append(target.flatten().cpu())
    return torch.cat(scores), torch.cat(labels)


def paper_metrics(scores: Tensor, labels: Tensor) -> dict[str, float]:
    scores, labels = scores.float().cpu(), labels.long().cpu()
    if labels.unique().numel() < 2:
        return {"auroc": float("nan"), "fpr_at_95_tpr": float("nan")}
    auroc = float(BinaryAUROC()(scores, labels).item())
    fpr, tpr, _ = BinaryROC()(scores, labels)
    valid = torch.where(tpr >= 0.95)[0]
    index = valid[0] if len(valid) else torch.argmin(torch.abs(tpr - 0.95))
    return {"auroc": auroc, "fpr_at_95_tpr": float(fpr[index].item())}


def group_masks(groups: list[str]) -> dict[str, Tensor]:
    values = np.asarray(groups, dtype=object)
    return {"all": torch.ones(len(groups), dtype=torch.bool), "seen_defects": torch.from_numpy(np.isin(values, ["good", "bad"])), "unseen_defects": torch.from_numpy(np.isin(values, ["good", "bad_unseen_defects"]))}


def groups_for_dataset(dataset: VADTimeSeriesDataset) -> list[str]:
    values = dataset.samples["label"].astype(str).to_numpy()
    groups = []
    for start in dataset.starts:
        groups.extend(values[start : start + dataset.timestamps].tolist())
    return groups


def evaluate_test(model, dataset, loader, device):
    scores, labels = collect_predictions(model, loader, device)
    groups = groups_for_dataset(dataset)
    print("test metrics:")
    for name, mask in group_masks(groups).items():
        metrics = paper_metrics(scores[mask], labels[mask])
        print(f"  {name}: AUROC={metrics['auroc'] * 100:.2f}, FPR@95TPR={metrics['fpr_at_95_tpr'] * 100:.2f}")


def train_one_epoch(model, loader, optimizer, device, *, loss_type, pos_weight, epoch, epochs):
    model.train()
    total = 0.0
    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", unit="batch")
    for features, labels in progress:
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = classification_loss(output.logits, labels, loss_type, pos_weight)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().item())
        progress.set_postfix(loss=f"{total / (progress.n + 1):.4f}")
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="D:/lowrank/VAD")
    parser.add_argument("--regime", choices=("low", "high"), default="high")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--timestamps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-size", type=int, choices=(224, 448), default=224)
    parser.add_argument("--backbone", choices=("resnet18", "resnet50", "dino_small"), default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial-mode", choices=("transformer", "pool"), default="transformer")
    parser.add_argument("--reference-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-pooling", choices=("mean", "attention"), default="mean")
    parser.add_argument("--loss", choices=("bce", "bpr"), default="bce")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--z-clip", type=float, default=8.0)
    parser.add_argument("--output-dir", default="results/spatial_ad_dino")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    dataset_root = resolve_dataset_root(args.root)
    selected = select_training_samples(make_supervised_vad_dataset(dataset_root, split="train"), args.regime, args.seed)
    reference_mean, reference_std = fit_sensor_reference_stats(selected, image_size=255)
    train_samples, val_samples = stratified_split(selected, args.val_fraction, args.seed)
    train_dataset, train_loader = make_loader(train_samples, timestamps=args.timestamps, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataset, val_loader = make_loader(val_samples, timestamps=args.timestamps, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_dataset, test_loader = make_loader(make_supervised_vad_dataset(dataset_root, split="test"), timestamps=args.timestamps, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = SpatialAD(image_size=255, input_size=args.input_size, backbone=args.backbone, pretrained=args.pretrained, freeze_backbone=not args.train_backbone, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout, spatial_mode=args.spatial_mode, use_reference_z=args.reference_z, patch_pooling=args.patch_pooling, reference_mean=reference_mean, reference_std=reference_std, z_clip=args.z_clip).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_labels = torch.from_numpy(train_dataset.labels.astype(np.float32))
    n_positive = train_labels.sum().item()
    n_negative = train_labels.numel() - n_positive
    pos_weight = torch.tensor([n_negative / max(n_positive, 1.0)], dtype=torch.float32, device=device)

    output_dir = Path(args.output_dir) / args.regime / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"device: {device}")
    print(f"model: {args.backbone} input_size={args.input_size} mode={args.spatial_mode} reference_z={args.reference_z} patch_pooling={args.patch_pooling} loss={args.loss}")
    print(f"backbone pretrained: {args.pretrained}, frozen: {not args.train_backbone}")
    feature_h, feature_w = model.feature_grid
    print(f"spatial feature tokens: {feature_h}x{feature_w} ({feature_h * feature_w})")
    print(f"train/val/test images: {len(train_dataset.samples)}/{len(val_dataset.samples)}/{len(test_dataset.samples)}")
    print(f"training labels: good={int(n_negative)}, bad={int(n_positive)}")

    best_auc, best_epoch, stale = -float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_type=args.loss, pos_weight=pos_weight, epoch=epoch, epochs=args.epochs)
        val_scores, val_labels = collect_predictions(model, val_loader, device, show_progress=True)
        val_auc = paper_metrics(val_scores, val_labels)["auroc"]
        history.append({"epoch": epoch, "loss": train_loss, "val_auroc": val_auc})
        print(f"epoch {epoch}: loss={train_loss:.4f}, val AUROC={val_auc * 100:.2f}")
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "val_auroc": val_auc, "args": vars(args)}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best validation AUROC: {best_auc * 100:.2f} at epoch {best_epoch}")
    evaluate_test(model, test_dataset, test_loader, device)


if __name__ == "__main__":
    main()
