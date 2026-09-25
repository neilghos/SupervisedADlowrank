"""Jointly train/evaluate SpatialAD with ResNet or DINO spatial features."""

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


def reverse_infonce_loss(embeddings: Tensor, labels: Tensor, temperature: float) -> Tensor:
    z = F.normalize(embeddings.reshape(-1, embeddings.shape[-1]), dim=-1)
    y = labels.reshape(-1).bool()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if z.shape[0] < 2 or y.all() or (~y).all():
        return z.sum() * 0.0
    similarity = (z @ z.transpose(0, 1)) / temperature
    off_diagonal = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    losses = []
    for anchor_mask, positive_mask, negative_mask in ((~y, ~y, y), (y, y, ~y)):
        for index in torch.where(anchor_mask)[0]:
            positive = off_diagonal[index] & positive_mask
            negative = negative_mask
            if positive.any() and negative.any():
                losses.append(torch.logsumexp(similarity[index][negative], 0) - torch.logsumexp(similarity[index][positive], 0))
    return torch.stack(losses).mean() if losses else z.sum() * 0.0


def classification_loss(logits: Tensor, labels: Tensor, loss_type: str, pos_weight: Tensor) -> Tensor:
    return bpr_loss(logits, labels) if loss_type == "bpr" else F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)


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


def train_one_epoch(model, loader, optimizer, device, *, loss_type, contrastive_weight, temperature, pos_weight, epoch, epochs):
    model.train()
    total = total_cls = total_contrastive = 0.0
    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", unit="batch")
    for features, labels in progress:
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        cls = classification_loss(output.logits, labels, loss_type, pos_weight)
        contrastive = reverse_infonce_loss(output.contrastive_embedding, labels, temperature)
        loss = cls + contrastive_weight * contrastive
        loss.backward()
        optimizer.step()
        total += float(loss.detach().item())
        total_cls += float(cls.detach().item())
        total_contrastive += float(contrastive.detach().item())
        progress.set_postfix(loss=f"{total / (progress.n + 1):.4f}", reverse=f"{total_contrastive / (progress.n + 1):.4f}")
    count = max(1, len(loader))
    return {"loss": total / count, "classification_loss": total_cls / count, "reverse_infonce_loss": total_contrastive / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="D:/lowrank/VAD")
    parser.add_argument("--regime", choices=("low", "high"), default="high")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--timestamps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", choices=("resnet18", "resnet50", "dino_small"), default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=("bce", "bpr"), default="bce")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
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

    model = SpatialAD(image_size=255, backbone=args.backbone, pretrained=args.pretrained, freeze_backbone=not args.train_backbone, d_model=args.d_model, projection_dim=args.projection_dim, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout, reference_mean=reference_mean, reference_std=reference_std, z_clip=args.z_clip).to(device)
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
    print(f"model: {args.backbone} SpatialAD joint training")
    print(f"backbone pretrained: {args.pretrained}, frozen: {not args.train_backbone}")
    print(f"spatial feature tokens: {model.feature_grid}x{model.feature_grid} ({model.feature_grid ** 2})")
    print(f"contrastive weight: {args.contrastive_weight}")
    print(f"train/val/test images: {len(train_dataset.samples)}/{len(val_dataset.samples)}/{len(test_dataset.samples)}")
    print(f"training labels: good={int(n_negative)}, bad={int(n_positive)}")

    best_auc, best_epoch, stale = -float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_type=args.loss, contrastive_weight=args.contrastive_weight, temperature=args.temperature, pos_weight=pos_weight, epoch=epoch, epochs=args.epochs)
        val_scores, val_labels = collect_predictions(model, val_loader, device, show_progress=True)
        val_auc = paper_metrics(val_scores, val_labels)["auroc"]
        history.append({"epoch": epoch, **train_metrics, "val_auroc": val_auc})
        print(f"epoch {epoch}: loss={train_metrics['loss']:.4f}, cls={train_metrics['classification_loss']:.4f}, reverse={train_metrics['reverse_infonce_loss']:.4f}, val AUROC={val_auc * 100:.2f}")
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
