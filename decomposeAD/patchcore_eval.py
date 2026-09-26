from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader.datamodule import (
    DATASETS_PATH,
    DECOMPOSITION_CACHE_ROOT,
    MVTecTestDataset,
)
from determinism import make_deterministic
from models.restorationnet import AnalyticalRestorationNet
from mvtec_metric import (
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)


COMPONENT_NAMES = [
    "cb_detail",
    "cb_fill",
    "cr_detail",
    "cr_fill",
    "y_coarse_fill",
    "y_edge_detail",
    "y_mid_fill",
    "y_texture_detail",
]


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore-style evaluator for the restoration model.")
    parser.add_argument("--mvtec-class", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=DATASETS_PATH)
    parser.add_argument("--clean-cache-root", type=str, default=DECOMPOSITION_CACHE_ROOT)
    parser.add_argument("--plot-dir", type=str, default="/data/stepdown vision/plots/eval")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> AnalyticalRestorationNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model = AnalyticalRestorationNet(in_channels=3, num_decomp_maps=len(COMPONENT_NAMES))
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[patchcore_eval] missing keys: {missing}")
    if unexpected:
        print(f"[patchcore_eval] unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    model: AnalyticalRestorationNet,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    rows: list[dict[str, float | int | str]] = []
    image_scores: list[float] = []
    labels: list[int] = []
    segmentations: list[np.ndarray] = []
    masks_gt: list[np.ndarray] = []

    for batch in dataloader:
        images = batch["image_raw"].to(device)
        masks = batch["mask"]
        batch_labels = [int(x) for x in batch["label"]]
        paths = batch["path"]

        _, restored = model(images)
        residual = (restored - images).pow(2)
        segmentation = residual.mean(dim=1)
        scores = segmentation.flatten(1).mean(dim=1)

        for index in range(images.size(0)):
            image_score = float(scores[index].detach().cpu())
            image_scores.append(image_score)
            labels.append(batch_labels[index])
            segmentations.append(segmentation[index].detach().cpu().numpy())
            masks_gt.append(masks[index, 0].detach().cpu().numpy())
            rows.append(
                {
                    "path": paths[index],
                    "label": batch_labels[index],
                    "input_restore_mse": image_score,
                }
            )

    image_metrics = compute_imagewise_retrieval_metrics(image_scores, labels)
    full_pixel_metrics = compute_pixelwise_retrieval_metrics(segmentations, masks_gt)

    anomaly_segmentations = []
    anomaly_masks = []
    for segmentation, mask in zip(segmentations, masks_gt):
        if np.sum(mask) > 0:
            anomaly_segmentations.append(segmentation)
            anomaly_masks.append(mask)

    anomaly_pixel_auroc = float("nan")
    if anomaly_segmentations:
        anomaly_pixel_auroc = float(
            compute_pixelwise_retrieval_metrics(anomaly_segmentations, anomaly_masks)["auroc"]
        )

    normal_scores = [score for score, label in zip(image_scores, labels) if label == 0]
    anomaly_scores = [score for score, label in zip(image_scores, labels) if label == 1]

    metrics = {
        "instance_auroc": float(image_metrics["auroc"]),
        "full_pixel_auroc": float(full_pixel_metrics["auroc"]),
        "anomaly_pixel_auroc": anomaly_pixel_auroc,
        "normal_mean_mse": float(np.mean(normal_scores)) if normal_scores else float("nan"),
        "normal_std_mse": float(np.std(normal_scores)) if normal_scores else float("nan"),
        "anomaly_mean_mse": float(np.mean(anomaly_scores)) if anomaly_scores else float("nan"),
        "anomaly_std_mse": float(np.std(anomaly_scores)) if anomaly_scores else float("nan"),
        "total_samples": float(len(rows)),
        "normal_count": float(len(normal_scores)),
        "anomaly_count": float(len(anomaly_scores)),
    }
    return rows, metrics


def save_outputs(
    artifact_dir: Path,
    mvtec_class: str,
    rows: list[dict[str, float | int | str]],
    metrics: dict[str, float],
):
    artifact_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifact_dir / "test_input_restore_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "input_restore_mse"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = artifact_dir / "test_input_restore_summary.txt"
    with summary_path.open("w") as handle:
        handle.write(f"dataset_name: {mvtec_class}\n")
        handle.write(f"total_samples: {int(metrics['total_samples'])}\n")
        handle.write(f"normal_count: {int(metrics['normal_count'])}\n")
        handle.write(f"anomaly_count: {int(metrics['anomaly_count'])}\n")
        handle.write(f"normal_mean_mse: {metrics['normal_mean_mse']:.8f}\n")
        handle.write(f"normal_std_mse: {metrics['normal_std_mse']:.8f}\n")
        handle.write(f"anomaly_mean_mse: {metrics['anomaly_mean_mse']:.8f}\n")
        handle.write(f"anomaly_std_mse: {metrics['anomaly_std_mse']:.8f}\n")
        handle.write(f"instance_auroc: {metrics['instance_auroc']:.8f}\n")
        handle.write(f"full_pixel_auroc: {metrics['full_pixel_auroc']:.8f}\n")
        handle.write(f"anomaly_pixel_auroc: {metrics['anomaly_pixel_auroc']:.8f}\n")
        handle.write(f"image_auroc: {metrics['instance_auroc']:.8f}\n")

    final_results_path = artifact_dir / "final_results.csv"
    with final_results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_name", "instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset_name": mvtec_class,
                "instance_auroc": metrics["instance_auroc"],
                "full_pixel_auroc": metrics["full_pixel_auroc"],
                "anomaly_pixel_auroc": metrics["anomaly_pixel_auroc"],
            }
        )

    normal_scores = [float(row["input_restore_mse"]) for row in rows if int(row["label"]) == 0]
    anomaly_scores = [float(row["input_restore_mse"]) for row in rows if int(row["label"]) == 1]
    plt.figure(figsize=(8, 5))
    if normal_scores:
        plt.hist(normal_scores, bins=40, alpha=0.6, label="normal", color="tab:blue")
    if anomaly_scores:
        plt.hist(anomaly_scores, bins=40, alpha=0.6, label="anomaly", color="tab:red")
    plt.xlabel("MSE(input, corrected)")
    plt.ylabel("Count")
    plt.title("Stage 2 test-score separation")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(artifact_dir / "test_input_restore_hist.png", dpi=200)
    plt.close()



def main():
    args = parse_args()
    make_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir = Path(args.plot_dir) / args.mvtec_class / run_id

    dataset = MVTecTestDataset(
        cls=args.mvtec_class,
        source=args.data_root,
        cache_root=args.clean_cache_root,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_model(args.stage2_checkpoint, device)
    rows, metrics = evaluate_model(model, dataloader, device)
    save_outputs(artifact_dir, args.mvtec_class, rows, metrics)
    print(f"[patchcore_eval] artifact_dir={artifact_dir}")
    print(
        f"[patchcore_eval] {args.mvtec_class}: "
        f"instance_auroc={metrics['instance_auroc']:.4f}, "
        f"full_pixel_auroc={metrics['full_pixel_auroc']:.4f}, "
        f"anomaly_pixel_auroc={metrics['anomaly_pixel_auroc']:.4f}"
    )


if __name__ == "__main__":
    main()
