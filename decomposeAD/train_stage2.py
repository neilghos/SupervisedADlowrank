from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from dataloader.datamodule import DATASETS_PATH, DECOMPOSITION_CACHE_ROOT
from determinism import make_deterministic
from perturbation.precompute_stage2_cache import STAGE2_CACHE_ROOT
from stage2_module import Stage2DataModule, Stage2LightningModule


def resolve_stage1_checkpoint(mvtec_class: str, checkpoint_arg: str | None, stage1_root: str) -> str:
    if checkpoint_arg:
        return checkpoint_arg

    class_root = Path(stage1_root) / mvtec_class
    if not class_root.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint directory does not exist: {class_root}")

    run_dirs = sorted([path for path in class_root.iterdir() if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No stage-1 runs found under: {class_root}")

    latest_run = run_dirs[-1]
    ckpts = sorted(latest_run.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found under: {latest_run}")
    return str(ckpts[-1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-class", type=str, default="bottle")
    parser.add_argument("--stage1-checkpoint", type=str, default=None)
    parser.add_argument("--stage1-checkpoint-dir", type=str, default="/data/stepdown vision/checkpoints/stage1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--decomposition-weight", type=float, default=1.0)
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=DATASETS_PATH)
    parser.add_argument("--clean-cache-root", type=str, default=DECOMPOSITION_CACHE_ROOT)
    parser.add_argument("--stage2-cache-root", type=str, default=str(STAGE2_CACHE_ROOT))
    parser.add_argument("--checkpoint-dir", type=str, default="/data/stepdown vision/checkpoints/stage2")
    parser.add_argument("--plot-dir", type=str, default="/data/stepdown vision/plots/stage2")
    return parser.parse_args()


def _collect_epoch_series(metrics_path: Path):
    series = {
        "train/decomposition_loss_epoch": {},
        "val/decomposition_loss": {},
        "train/fusion_loss_epoch": {},
        "val/fusion_loss": {},
    }

    with metrics_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch_value = row.get("epoch")
            if epoch_value in (None, ""):
                continue
            epoch = int(float(epoch_value))
            for metric_name in series:
                value = row.get(metric_name)
                if value not in (None, ""):
                    series[metric_name][epoch] = float(value)

    return {name: [(idx, values[idx]) for idx in sorted(values)] for name, values in series.items()}


def plot_recon_losses(metrics_path: Path, output_path: Path, mvtec_class: str):
    curves = _collect_epoch_series(metrics_path)

    plt.figure(figsize=(9, 6))
    for metric_name, label in [
        ("train/decomposition_loss_epoch", "train decomposition"),
        ("val/decomposition_loss", "val decomposition"),
        ("train/fusion_loss_epoch", "train fusion"),
        ("val/fusion_loss", "val fusion"),
    ]:
        points = curves[metric_name]
        if not points:
            continue
        epochs, values = zip(*points)
        plt.plot(epochs, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("MSE reconstruction loss")
    plt.title(f"Stage 2 reconstruction losses: {mvtec_class}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    make_deterministic(args.seed)
    stage1_checkpoint = resolve_stage1_checkpoint(
        mvtec_class=args.mvtec_class,
        checkpoint_arg=args.stage1_checkpoint,
        stage1_root=args.stage1_checkpoint_dir,
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    datamodule = Stage2DataModule(
        mvtec_class=args.mvtec_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        data_root=args.data_root,
        clean_cache_root=args.clean_cache_root,
        stage2_cache_root=args.stage2_cache_root,
        seed=args.seed,
    )
    datamodule.setup("fit")
    split = datamodule.split_summary()
    print(
        f"[stage2 split] class={split['mvtec_class']} "
        f"total_normal_train={split['full_train_size']} "
        f"inner_train={split['inner_train_size']} "
        f"inner_val={split['inner_val_size']} "
        f"perturbation_combinations={split['perturbation_combinations']} "
        f"augmented_train={split['augmented_train_size']} "
        f"augmented_val={split['augmented_val_size']}"
    )
    artifact_dir = Path(args.plot_dir) / args.mvtec_class / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    module = Stage2LightningModule(
        learning_rate=args.learning_rate,
        decomposition_weight=args.decomposition_weight,
        fusion_weight=args.fusion_weight,
        stage1_checkpoint=stage1_checkpoint,
        artifact_dir=str(artifact_dir),
    )
    logger = CSVLogger(
        save_dir="/data/stepdown vision/logs",
        name=f"stage2/{args.mvtec_class}",
        version=run_id,
    )
    checkpoint = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/{args.mvtec_class}/{run_id}",
        filename="{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint],
        log_every_n_steps=1,
        logger=logger,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    trainer.fit(module, datamodule=datamodule)

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    plot_output = artifact_dir / "recon_losses.png"
    plot_recon_losses(metrics_path, plot_output, args.mvtec_class)


if __name__ == "__main__":
    main()
