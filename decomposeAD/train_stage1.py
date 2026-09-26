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

from dataloader.datamodule import DATASETS_PATH, DECOMPOSITION_CACHE_ROOT, _CLASSNAMES
from determinism import make_deterministic
from stage1_module import Stage1DataModule, Stage1LightningModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-class", type=str, default="bottle")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--decomposition-weight", type=float, default=1.0)
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=DATASETS_PATH)
    parser.add_argument("--cache-root", type=str, default=DECOMPOSITION_CACHE_ROOT)
    parser.add_argument("--checkpoint-dir", type=str, default="/data/stepdown vision/checkpoints/stage1")
    parser.add_argument("--plot-dir", type=str, default="/data/stepdown vision/plots/stage1")
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
    plt.ylabel("L1 reconstruction loss")
    plt.title(f"Stage 1 reconstruction losses: {mvtec_class}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_single_class(args, mvtec_class: str):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    datamodule = Stage1DataModule(
        mvtec_class=mvtec_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        data_root=args.data_root,
        cache_root=args.cache_root,
        seed=args.seed,
    )
    datamodule.setup("fit")
    split = datamodule.split_summary()
    print(
        f"[stage1 split] class={split['mvtec_class']} "
        f"total_normal_train={split['full_train_size']} "
        f"inner_train={split['inner_train_size']} "
        f"inner_val={split['inner_val_size']}"
    )
    module = Stage1LightningModule(
        learning_rate=args.learning_rate,
        decomposition_weight=args.decomposition_weight,
        fusion_weight=args.fusion_weight,
    )
    logger = CSVLogger(
        save_dir="/data/stepdown vision/logs",
        name=f"stage1/{mvtec_class}",
        version=run_id,
    )

    checkpoint = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/{mvtec_class}/{run_id}",
        filename="{epoch:02d}-{val_loss:.4f}",
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
    plot_output = Path(args.plot_dir) / mvtec_class / run_id / "recon_losses.png"
    plot_recon_losses(metrics_path, plot_output, mvtec_class)



def main():
    args = parse_args()
    make_deterministic(args.seed)

    classes = _CLASSNAMES if args.mvtec_class == "all" else [args.mvtec_class]
    for index, mvtec_class in enumerate(classes, start=1):
        print(f"[stage1] starting class {index}/{len(classes)}: {mvtec_class}")
        run_single_class(args, mvtec_class)


if __name__ == "__main__":
    main()
