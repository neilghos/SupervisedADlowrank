"""Train and evaluate the patch-based SpatialAD model on VAD.

The runner deliberately avoids the old Anomalib/Lightning wrapper.  VAD is
loaded through ``src.torchdata`` as image-as-time-series windows:

    features: [B, 65025, T]
    labels:   [B, T]

The timestamp axis is only a batching axis here; SpatialAD processes every
image independently and performs attention over spatial patch tokens.
"""

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
from src.torchdata import (
    VADTimeSeriesDataset,
    resolve_dataset_root,
    select_training_samples,
)


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
    """Split selected training images while preserving good/bad proportions."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    labels = samples["label_index"].to_numpy()
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"Need at least two samples for label {label}")
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_val = min(n_val, len(indices) - 1)
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train = samples.iloc[train_indices].reset_index(drop=True)
    val = samples.iloc[val_indices].reset_index(drop=True)
    return train, val


def trim_to_windows(samples, timestamps: int):
    """Keep only complete windows, matching VADTimeSeriesDataset's contract."""

    usable = (len(samples) // timestamps) * timestamps
    if usable < timestamps:
        raise ValueError(
            f"Split has {len(samples)} images, which is insufficient for "
            f"timestamps={timestamps}"
        )
    return samples.iloc[:usable].reset_index(drop=True)


def make_loader(
    samples,
    *,
    timestamps: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[VADTimeSeriesDataset, DataLoader]:
    dataset = VADTimeSeriesDataset(
        trim_to_windows(samples, timestamps),
        image_size=255,
        timestamps=timestamps,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def bpr_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Pairwise ranking loss, with BCE fallback for single-class windows."""

    logits = logits.flatten()
    labels = labels.flatten().bool()
    positive = logits[labels]
    negative = logits[~labels]
    if positive.numel() == 0 or negative.numel() == 0:
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    differences = positive[:, None] - negative[None, :]
    return -F.logsigmoid(differences).mean()


def compute_loss(
    model: SpatialAD,
    output,
    features: Tensor,
    labels: Tensor,
    *,
    loss_type: str,
    reconstruction_weight: float,
    classification_weight: float,
    pos_weight: Tensor | None,
) -> dict[str, Tensor]:
    target_images = features.permute(0, 2, 1).reshape_as(output.reconstruction)
    reconstruction_loss = F.mse_loss(output.reconstruction, target_images)
    if loss_type == "bpr":
        classification_loss = bpr_loss(output.logits, labels)
    else:
        classification_loss = F.binary_cross_entropy_with_logits(
            output.logits,
            labels.float(),
            pos_weight=pos_weight,
        )
    total = (
        reconstruction_weight * reconstruction_loss
        + classification_weight * classification_loss
    )
    return {
        "loss": total,
        "reconstruction_loss": reconstruction_loss,
        "classification_loss": classification_loss,
    }


@torch.inference_mode()
def collect_predictions(
    model: SpatialAD,
    loader: DataLoader,
    device: torch.device,
    *,
    show_progress: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    scores: list[Tensor] = []
    labels: list[Tensor] = []
    reconstruction_scores: list[Tensor] = []
    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="validation", unit="batch", leave=False)
    for features, target in iterator:
        features = features.to(device, non_blocking=True)
        output = model(features)
        scores.append(output.logits.detach().flatten().cpu())
        labels.append(target.detach().flatten().cpu())
        reconstruction_scores.append(output.reconstruction_score.detach().flatten().cpu())
    return torch.cat(scores), torch.cat(labels), torch.cat(reconstruction_scores)


def paper_metrics(scores: Tensor, labels: Tensor) -> dict[str, float]:
    """Compute classification AUROC and FPR at 95% TPR."""

    scores = scores.float().cpu()
    labels = labels.long().cpu()
    if labels.unique().numel() < 2:
        return {"auroc": float("nan"), "fpr_at_95_tpr": float("nan")}

    auroc = float(BinaryAUROC()(scores, labels).item())
    fpr, tpr, _ = BinaryROC()(scores, labels)
    valid = torch.where(tpr >= 0.95)[0]
    index = valid[0] if len(valid) else torch.argmin(torch.abs(tpr - 0.95))
    return {
        "auroc": auroc,
        "fpr_at_95_tpr": float(fpr[index].item()),
    }


def test_group_masks(groups: list[str]) -> dict[str, Tensor]:
    group_tensor = np.asarray(groups, dtype=object)
    return {
        "all": torch.ones(len(groups), dtype=torch.bool),
        "seen_defects": torch.from_numpy(
            np.isin(group_tensor, ["good", "bad"])
        ),
        "unseen_defects": torch.from_numpy(
            np.isin(group_tensor, ["good", "bad_unseen_defects"])
        ),
    }


def groups_for_dataset(dataset: VADTimeSeriesDataset) -> list[str]:
    labels = dataset.samples["label"].astype(str).to_numpy()
    groups: list[str] = []
    for start in dataset.starts:
        groups.extend(labels[start : start + dataset.timestamps].tolist())
    return groups


def evaluate_test(
    model: SpatialAD,
    dataset: VADTimeSeriesDataset,
    loader: DataLoader,
    device: torch.device,
) -> None:
    scores, labels, reconstruction_scores = collect_predictions(model, loader, device)
    groups = groups_for_dataset(dataset)
    if len(groups) != len(labels):
        raise RuntimeError(f"Group count {len(groups)} does not match predictions {len(labels)}")

    print("test metrics:")
    for name, mask in test_group_masks(groups).items():
        metrics = paper_metrics(scores[mask], labels[mask])
        print(
            f"  {name}: AUROC={metrics['auroc'] * 100:.2f}, "
            f"FPR@95TPR={metrics['fpr_at_95_tpr'] * 100:.2f}"
        )

    reconstruction_metrics = paper_metrics(reconstruction_scores, labels)
    print(
        "  reconstruction-only: "
        f"AUROC={reconstruction_metrics['auroc'] * 100:.2f}, "
        f"FPR@95TPR={reconstruction_metrics['fpr_at_95_tpr'] * 100:.2f}"
    )


def train_one_epoch(
    model: SpatialAD,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_type: str,
    reconstruction_weight: float,
    classification_weight: float,
    pos_weight: Tensor | None,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "reconstruction_loss": 0.0, "classification_loss": 0.0}
    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", unit="batch")
    for features, labels in progress:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        losses = compute_loss(
            model,
            output,
            features,
            labels,
            loss_type=loss_type,
            reconstruction_weight=reconstruction_weight,
            classification_weight=classification_weight,
            pos_weight=pos_weight,
        )
        losses["loss"].backward()
        optimizer.step()
        for name in totals:
            totals[name] += float(losses[name].detach().item())
        mean_loss = totals["loss"] / (progress.n + 1)
        progress.set_postfix(loss=f"{mean_loss:.4f}")
    count = max(1, len(loader))
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="D:/lowrank/VAD")
    parser.add_argument("--regime", choices=("low", "high"), default="low")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--timestamps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=("bce", "bpr"), default="bce")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/spatial_ad")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )

    dataset_root = resolve_dataset_root(args.root)
    selected = select_training_samples(
        make_supervised_vad_dataset(dataset_root, split="train"),
        regime=args.regime,
        seed=args.seed,
    )
    train_samples, val_samples = stratified_split(selected, args.val_fraction, args.seed)
    train_dataset, train_loader = make_loader(
        train_samples,
        timestamps=args.timestamps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataset, val_loader = make_loader(
        val_samples,
        timestamps=args.timestamps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_samples = make_supervised_vad_dataset(dataset_root, split="test")
    test_dataset, test_loader = make_loader(
        test_samples,
        timestamps=args.timestamps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = SpatialAD(
        image_size=255,
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_labels = torch.from_numpy(train_dataset.labels.astype(np.float32))
    n_positive = train_labels.sum().item()
    n_negative = train_labels.numel() - n_positive
    pos_weight = torch.tensor(
        [n_negative / max(n_positive, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    checkpoint_dir = Path(args.output_dir) / args.regime / f"seed{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"device: {device}")
    print("model: SpatialAD")
    print(f"patches: {model.grid_size}x{model.grid_size} ({model.n_patches} tokens)")
    print(f"train images/windows: {len(train_dataset.samples)}/{len(train_loader)}")
    print(f"validation images/windows: {len(val_dataset.samples)}/{len(val_loader)}")
    print(f"test images/windows: {len(test_dataset.samples)}/{len(test_loader)}")
    print(f"training labels: good={int(n_negative)}, bad={int(n_positive)}")
    print(f"best checkpoint: {checkpoint_path}")

    best_auc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_type=args.loss,
            reconstruction_weight=args.reconstruction_weight,
            classification_weight=args.classification_weight,
            pos_weight=pos_weight,
            epoch=epoch,
            epochs=args.epochs,
        )
        val_scores, val_labels, _ = collect_predictions(
            model,
            val_loader,
            device,
            show_progress=True,
        )
        val_metrics = paper_metrics(val_scores, val_labels)
        val_auc = val_metrics["auroc"]
        row = {"epoch": epoch, **train_metrics, "val_auroc": val_auc}
        history.append(row)
        print(
            f"epoch {epoch}: loss={train_metrics['loss']:.4f}, "
            f"cls={train_metrics['classification_loss']:.4f}, "
            f"val AUROC={val_auc * 100:.2f}"
        )

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_auroc": val_auc,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("No valid checkpoint was produced; validation AUROC was undefined")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    Path(checkpoint_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    print(f"best validation AUROC: {best_auc * 100:.2f} at epoch {best_epoch}")
    evaluate_test(model, test_dataset, test_loader, device)


if __name__ == "__main__":
    main()
