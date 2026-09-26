from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


RSNA_ROOT = "/data/imageaddatasets/rsna-pneumonia-detection-challenge"
RSNA_IMAGE_DIR = "stage_2_train_images"
RSNA_CLASS_CSV = "stage_2_detailed_class_info.csv"
RSNA_BOX_CSV = "stage_2_train_labels.csv"


def _require_pydicom():
    try:
        import pydicom  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "RSNA loader requires `pydicom`. Install it in the active env before using "
            "`dataloader/imagedatamodule.py`."
        ) from exc
    return pydicom


def load_rsna_image(path: str | Path, image_size: int = 224) -> torch.Tensor:
    pydicom = _require_pydicom()
    dcm = pydicom.dcmread(str(path))
    image = dcm.pixel_array.astype("float32")
    image -= image.min()
    image /= max(image.max(), 1e-6)
    image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
    image = transforms.Resize((image_size, image_size), antialias=True)(image)
    return image


def _read_class_info(root: Path) -> dict[str, str]:
    mapping = {}
    with (root / RSNA_CLASS_CSV).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["patientId"]] = row["class"]
    return mapping


def _read_box_info(root: Path) -> dict[str, list[dict[str, float]]]:
    boxes: dict[str, list[dict[str, float]]] = defaultdict(list)
    with (root / RSNA_BOX_CSV).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["patientId"]
            if row["Target"] == "1":
                boxes[patient_id].append(
                    {
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "width": float(row["width"]),
                        "height": float(row["height"]),
                    }
                )
    return boxes


def build_rsna_records(root: str = RSNA_ROOT) -> list[dict[str, Any]]:
    root_path = Path(root)
    class_map = _read_class_info(root_path)
    box_map = _read_box_info(root_path)
    image_dir = root_path / RSNA_IMAGE_DIR

    records = []
    for image_path in sorted(image_dir.glob("*.dcm")):
        patient_id = image_path.stem
        class_name = class_map[patient_id]
        records.append(
            {
                "patient_id": patient_id,
                "path": str(image_path),
                "label": 0 if class_name == "Normal" else 1,
                "class_name": class_name,
                "boxes": box_map.get(patient_id, []),
            }
        )
    return records


class RSNADataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_size: int = 224):
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = load_rsna_image(record["path"], image_size=self.image_size)
        return {
            "image_raw": image,
            "label": record["label"],
            "class_name": record["class_name"],
            "patient_id": record["patient_id"],
            "boxes": record["boxes"],
            "path": record["path"],
        }


class RSNAImageDataset:
    def __init__(
        self,
        root: str = RSNA_ROOT,
        image_size: int = 224,
        train_split: float = 0.7,
        val_split: float = 0.15,
        seed: int = 42,
    ):
        self.root = root
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed

        records = build_rsna_records(root)
        dataset = RSNADataset(records, image_size=image_size)

        train_size = int(len(dataset) * train_split)
        val_size = int(len(dataset) * val_split)
        test_size = len(dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )

    def get_datasets(self):
        return self.train_ds, self.val_ds, self.test_ds

    def get_dataloaders(self, batch_size: int = 8, num_workers: int = 0):
        return (
            DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        )


if __name__ == "__main__":
    dataset = RSNAImageDataset()
    train_ds, val_ds, test_ds = dataset.get_datasets()
    print(f"rsna train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    sample = train_ds[0]
    print(f"sample image: {sample['image_raw'].shape}")
    print(f"sample label: {sample['label']}")
    print(f"sample class: {sample['class_name']}")
    print(f"sample patient: {sample['patient_id']}")
