"""Train supervised low-rank anomaly detection and print paper metrics."""

import argparse
import warnings

import torch

# Keep the experiment output focused on progress, metrics, and real failures.
warnings.filterwarnings("ignore", message=r"Importing from timm\.models\.layers is deprecated.*")
warnings.filterwarnings("ignore", message=r".*dinov2 package is deprecated.*")
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=r".*persistent_workers=True.*")
torch.set_float32_matmul_precision("high")

from anomalib.engine import Engine

from lowrank_vad.data import SupervisedVAD, describe_vad_test_split
from lowrank_vad.metrics import evaluate_model
from lowrank_vad.model import SupervisedLowRank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("low", "high"), default="low")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="D:/lowrank/VAD")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    datamodule = SupervisedVAD(
        root=args.root,
        category="vad",
        regime=args.regime,
        seed=args.seed,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=args.num_workers,
    )
    model = SupervisedLowRank(rank=50)
    engine = Engine(
        default_root_dir=f"results/vad_supervised_lowrank/{args.regime}/seed{args.seed}",
        max_epochs=1,
        num_sanity_val_steps=0,
    )

    datamodule.prepare_data()
    datamodule.setup()
    print("regime:", args.regime)
    print("normal training images: 2000")
    print("bad training images:", datamodule.train_bad_count)
    print("total training images:", len(datamodule.train_data))
    print("test split:", describe_vad_test_split(datamodule))

    engine.fit(model=model, datamodule=datamodule)
    metrics = evaluate_model(model, datamodule)
    print(f"Classification AUROC: {metrics['classification_auroc'] * 100:.2f}")
    print(f"FPR@95TPR: {metrics['fpr_at_95_tpr']:.2f}")


if __name__ == "__main__":
    main()
