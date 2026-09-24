"""Train the spatial-attention MIL anomaly detector on VAD."""

import argparse
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=r"Importing from timm\.models\.layers is deprecated.*")
warnings.filterwarnings("ignore", message=r".*dinov2 package is deprecated.*")
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=r".*persistent_workers=True.*")
torch.set_float32_matmul_precision("high")

from anomalib.engine import Engine

from lowrank_vad.metrics import evaluate_attention_maps, evaluate_model, evaluate_model_by_test_group
from lowrank_vad.model import SpatialMIL
from lowrank_vad.validated import ValidatedSupervisedVAD


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("low", "high"), default="low")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="D:/lowrank/VAD")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--loss", choices=("bce", "bpr"), default="bce")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--topk-fraction", type=float, default=0.05)
    parser.add_argument("--token-variance-threshold", type=float, default=0.0)
    args = parser.parse_args()
    seed_everything(args.seed)

    datamodule = ValidatedSupervisedVAD(
        root=args.root,
        category="vad",
        regime=args.regime,
        seed=args.seed,
        val_fraction=0.20,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=args.num_workers,
    )
    model = SpatialMIL(
        backbone=args.backbone,
        loss_type=args.loss,
        epochs=args.epochs,
        topk_fraction=args.topk_fraction,
        token_variance_threshold=args.token_variance_threshold,
    )
    engine = Engine(
        default_root_dir=f"results/vad_spatial_mil/{args.backbone}/{args.regime}/seed{args.seed}",
        max_epochs=1,
        num_sanity_val_steps=0,
    )

    datamodule.prepare_data()
    datamodule.setup()
    print("model: spatial attention MIL")
    print("backbone:", args.backbone)
    print("loss:", args.loss)
    print("train split:", len(datamodule.train_data))
    print("validation split:", len(datamodule.val_data))
    print("test split remains untouched until final scoring")

    engine.fit(model=model, datamodule=datamodule)
    metrics = evaluate_model(model, datamodule)
    print(f"Classification AUROC: {metrics['classification_auroc'] * 100:.2f}")
    print(f"FPR@95TPR: {metrics['fpr_at_95_tpr']:.2f}")

    print("test-group diagnostics:")
    for name, group_metrics in evaluate_model_by_test_group(model, datamodule).items():
        print(
            f"  {name}: AUROC={group_metrics['classification_auroc'] * 100:.2f}, "
            f"FPR@95TPR={group_metrics['fpr_at_95_tpr']:.2f}"
        )

    print("attention-map diagnostics:")
    for aggregation, groups in evaluate_attention_maps(model, datamodule).items():
        summary = ", ".join(
            f"{group}={values['classification_auroc'] * 100:.2f}"
            for group, values in groups.items()
        )
        print(f"  {aggregation}: {summary}")


if __name__ == "__main__":
    main()
