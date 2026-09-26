from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dataloader.datamodule import DECOMPOSITION_CACHE_ROOT, DATASETS_PATH, _CLASSNAMES
from perturbation.precompute_stage2_cache import STAGE2_CACHE_ROOT


ROOT = Path("/data/stepdown vision")
PREP_SCRIPT = ROOT / "prepare_mvtec_package.py"
STAGE1_SCRIPT = ROOT / "train_stage1.py"
STAGE2_SCRIPT = ROOT / "train_stage2.py"
EVAL_SCRIPT = ROOT / "eval.py"
RUNS_ROOT = ROOT / "runs" / "mvtec_pipeline"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-class", type=str, default="all")
    parser.add_argument("--data-root", type=str, default=DATASETS_PATH)
    parser.add_argument("--clean-cache-root", type=str, default=DECOMPOSITION_CACHE_ROOT)
    parser.add_argument("--stage2-cache-root", type=str, default=str(STAGE2_CACHE_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-package", action="store_true")
    return parser.parse_args()


def resolve_classes(arg: str) -> list[str]:
    if arg == "all":
        return list(_CLASSNAMES)
    return [name.strip() for name in arg.split(",") if name.strip()]


def latest_checkpoint(class_dir: Path) -> Path:
    if not class_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {class_dir}")
    run_dirs = sorted([path for path in class_dir.iterdir() if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {class_dir}")
    latest_run = run_dirs[-1]
    ckpts = sorted(latest_run.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {latest_run}")
    return ckpts[-1]


def latest_artifact_dir(class_dir: Path) -> Path:
    if not class_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {class_dir}")
    run_dirs = sorted([path for path in class_dir.iterdir() if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No artifact run directories found under: {class_dir}")
    return run_dirs[-1]


def parse_summary(summary_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with summary_path.open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            result[key.strip()] = value.strip()
    return result


def write_results_csv(csv_path: Path, rows: list[dict[str, str | float]]):
    fields = [
        "class_name",
        "total_samples",
        "normal_count",
        "anomaly_count",
        "normal_mean_mse",
        "normal_std_mse",
        "anomaly_mean_mse",
        "anomaly_std_mse",
        "image_auroc",
        "stage1_ckpt",
        "stage2_ckpt",
        "eval_dir",
    ]
    metric_fields = [
        "normal_mean_mse",
        "normal_std_mse",
        "anomaly_mean_mse",
        "anomaly_std_mse",
        "image_auroc",
    ]

    mean_row: dict[str, str | float] = {field: "" for field in fields}
    mean_row["class_name"] = "Mean"
    if rows:
        for field in metric_fields:
            values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
            if values:
                mean_row[field] = sum(values) / len(values)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)


def class_clean_cache_dir(clean_cache_root: str, mvtec_class: str) -> Path:
    root = Path(clean_cache_root)
    return root.parent / mvtec_class / root.name


def class_stage2_cache_dir(stage2_cache_root: str, mvtec_class: str) -> Path:
    return Path(stage2_cache_root) / mvtec_class


def run_command(command: list[str]):
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(command, check=True, cwd=str(ROOT), env=env)


def main():
    args = parse_args()
    classes = resolve_classes(args.mvtec_class)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_root = RUNS_ROOT / run_id
    stage1_ckpt_root = session_root / "checkpoints" / "stage1"
    stage2_ckpt_root = session_root / "checkpoints" / "stage2"
    stage1_plot_root = session_root / "plots" / "stage1"
    stage2_plot_root = session_root / "plots" / "stage2"
    eval_plot_root = session_root / "plots" / "eval"
    results_csv = session_root / "results.csv"

    rows: list[dict[str, str | float]] = []

    for index, mvtec_class in enumerate(classes, start=1):
        print(f"[pipeline] class {index}/{len(classes)}: {mvtec_class}")

        run_command(
            [
                sys.executable,
                str(PREP_SCRIPT),
                "--mvtec-class",
                mvtec_class,
                "--data-root",
                args.data_root,
                "--clean-cache-root",
                args.clean_cache_root,
                "--stage2-cache-root",
                args.stage2_cache_root,
            ]
        )

        run_command(
            [
                sys.executable,
                str(STAGE1_SCRIPT),
                "--mvtec-class",
                mvtec_class,
                "--epochs",
                str(args.stage1_epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--val-split",
                str(args.val_split),
                "--seed",
                str(args.seed),
                "--data-root",
                args.data_root,
                "--cache-root",
                args.clean_cache_root,
                "--checkpoint-dir",
                str(stage1_ckpt_root),
                "--plot-dir",
                str(stage1_plot_root),
            ]
        )

        stage1_ckpt = latest_checkpoint(stage1_ckpt_root / mvtec_class)

        run_command(
            [
                sys.executable,
                str(STAGE2_SCRIPT),
                "--mvtec-class",
                mvtec_class,
                "--stage1-checkpoint",
                str(stage1_ckpt),
                "--epochs",
                str(args.stage2_epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--val-split",
                str(args.val_split),
                "--seed",
                str(args.seed),
                "--data-root",
                args.data_root,
                "--clean-cache-root",
                args.clean_cache_root,
                "--stage2-cache-root",
                args.stage2_cache_root,
                "--checkpoint-dir",
                str(stage2_ckpt_root),
                "--plot-dir",
                str(stage2_plot_root),
            ]
        )

        stage2_ckpt = latest_checkpoint(stage2_ckpt_root / mvtec_class)

        run_command(
            [
                sys.executable,
                str(EVAL_SCRIPT),
                "--mvtec-class",
                mvtec_class,
                "--stage2-checkpoint",
                str(stage2_ckpt),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--val-split",
                str(args.val_split),
                "--seed",
                str(args.seed),
                "--data-root",
                args.data_root,
                "--clean-cache-root",
                args.clean_cache_root,
                "--stage2-cache-root",
                args.stage2_cache_root,
                "--plot-dir",
                str(eval_plot_root),
            ]
        )

        eval_dir = latest_artifact_dir(eval_plot_root / mvtec_class)
        summary = parse_summary(eval_dir / "test_input_restore_summary.txt")
        row: dict[str, str | float] = {
            "class_name": mvtec_class,
            "total_samples": summary.get("total_samples", ""),
            "normal_count": summary.get("normal_count", ""),
            "anomaly_count": summary.get("anomaly_count", ""),
            "normal_mean_mse": summary.get("normal_mean_mse", ""),
            "normal_std_mse": summary.get("normal_std_mse", ""),
            "anomaly_mean_mse": summary.get("anomaly_mean_mse", ""),
            "anomaly_std_mse": summary.get("anomaly_std_mse", ""),
            "image_auroc": summary.get("image_auroc", ""),
            "stage1_ckpt": str(stage1_ckpt),
            "stage2_ckpt": str(stage2_ckpt),
            "eval_dir": str(eval_dir),
        }
        rows.append(row)
        write_results_csv(results_csv, rows)
        print(f"[pipeline] appended results for {mvtec_class} -> {results_csv}")

        if not args.keep_package:
            clean_dir = class_clean_cache_dir(args.clean_cache_root, mvtec_class)
            stage2_dir = class_stage2_cache_dir(args.stage2_cache_root, mvtec_class)
            if clean_dir.exists():
                shutil.rmtree(clean_dir)
                print(f"[pipeline] deleted clean cache: {clean_dir}")
            if stage2_dir.exists():
                shutil.rmtree(stage2_dir)
                print(f"[pipeline] deleted stage2 cache: {stage2_dir}")

    print(f"[pipeline] finished -> {results_csv}")


if __name__ == "__main__":
    main()
