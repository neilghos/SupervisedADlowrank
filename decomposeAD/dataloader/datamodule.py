from pathlib import Path
import subprocess
import sys
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


_CLASSNAMES = [
    "bottle",
    "carpet",
    "transistor",
    "leather",
    "capsule",
    "grid",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "cable",
    "wood",
    "zipper",
]


DATASETS_PATH = "/data/imageaddatasets/mvtec_anomaly_detection"
DECOMPOSITION_CACHE_ROOT = "/data/cache/decompositions/mvtec/ycbcr"
PRECOMPUTE_SCRIPT = "/data/stepdown vision/decomposition/precompute_decomposition.py"
DEFAULT_RESIZE_SIZE = 256
DEFAULT_IMAGE_SIZE = 224


def _class_cache_root(cache_root: str, cls: str) -> Path:
    cache_root_path = Path(cache_root)
    return cache_root_path.parent / cls / cache_root_path.name


def _decomposition_path(class_root: str, image_path: str, cache_root: str) -> Path:
    relative = Path(image_path).relative_to(Path(class_root)).with_suffix(".pt")
    return Path(cache_root) / relative


def _load_components(class_root: str, image_path: str, cache_root: str) -> dict:
    cache_path = _decomposition_path(class_root, image_path, cache_root)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing decomposition cache file: {cache_path}. "
            f"Precompute the decomposition cache before training."
        )
    sample = torch.load(cache_path, map_location="cpu", weights_only=False)
    return {name: value.float() for name, value in sample["components"].items()}


def _cache_matches_size(cache_root: str, expected_resize: int, expected_size: int) -> bool:
    cache_root_path = Path(cache_root)
    sample_path = next(cache_root_path.rglob("*.pt"), None)
    if sample_path is None:
        return False
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    if sample.get("resize_size") != expected_resize or sample.get("image_size") != expected_size:
        return False
    image = sample["image"]
    if image.shape[-2:] != (expected_size, expected_size):
        return False
    first_component = next(iter(sample["components"].values()))
    return first_component.shape[-2:] == (expected_size, expected_size)


def _ensure_decomposition_cache(
    cls: str,
    source: str,
    cache_root: str,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    class_root = Path(source) / cls
    cache_root_path = _class_cache_root(cache_root, cls)
    expected_dirs = [cache_root_path / "train", cache_root_path / "test"]
    if all(path.exists() for path in expected_dirs) and _cache_matches_size(str(cache_root_path), resize_size, image_size):
        return

    command = [
        sys.executable,
        PRECOMPUTE_SCRIPT,
        "--yrcrb",
        "--data-root",
        str(class_root),
        "--cache-root",
        str(cache_root_path.parent),
        "--split",
        "all",
    ]
    if any(path.exists() for path in expected_dirs):
        command.append("--overwrite")

    subprocess.run(
        command,
        check=True,
    )


class MVTecDataset:
    def __init__(
        self,
        cls: str,
        source: str = DATASETS_PATH,
        cache_root: str = DECOMPOSITION_CACHE_ROOT,
        resize: int = DEFAULT_RESIZE_SIZE,
        size: int = DEFAULT_IMAGE_SIZE,
    ):
        """
            This constructor is used to initialized an instance of MVTecDataset. It is identified by the following parameters : 

            - cls : This string value corresponds to the class of the dataset that is considered

            - source : This string represents the path to the dataset 

            - size : This integer represents the size of the dataset 

            This __init__ method assigns the values of the parameters to the class attributes and creates the
            train and test datasets from the local dataset content.
        """
        self.cls = cls
        self.source = source
        self.cache_root = cache_root
        self.resize = resize
        self.size = size
        _ensure_decomposition_cache(cls, source, cache_root, resize, size)
        self.train_ds = MVTecTrainDataset(cls, source, cache_root, resize, size)
        self.test_ds = MVTecTestDataset(cls, source, cache_root, resize, size)

    def get_datasets(self):
        """
            Return the train and test datasets
        """
        return self.train_ds, self.test_ds

    def get_dataloaders(self, num_workers=8, batch_size=1):
        """
            This method creating Dataloaders from the train_dataset and test_dataset
            In PyTorch, a Dataloader is a class allowing loading data from a dataset and creating an iterator over the data. 
            It allows to easily and efficiently load data 
            The Dataloader takes several parameters : batch_size, shuffle (= True if we want to have the data reshuffled at every epoch),
            num_workers that allow to parallelize calculation and make it faster, pin_memory that contributes in improving the calculation
        """
        # Creating a Dataloader from the training data
        train_dataloader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        # Creating a DataLoader from the test_dataset
        test_dataloader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_dataloader, test_dataloader


class MVTecTrainDataset(ImageFolder):
    """
        This class allows the creation of a instance of MVTecTrainDataset. It takes four parameters : 

        - cls : Representing the name of the dataset class considered

        - source : String representing the path of the dataset

         - resize : Integer representing the size to resize the images (the default value is 256)

         - imagesize : Integer representing the size of the images (the default value is 224)

        In this method, several transformations to the images are performed. Resizing the image to the specified resize size, cropping the images 
        to the specified imagesize size, converting the image to a Pytorch tensor and normalizing the image with a mean and std value. 
    """
    def __init__(
        self,
        cls: str,
        source: str = DATASETS_PATH,
        cache_root: str = DECOMPOSITION_CACHE_ROOT,
        resize: int = DEFAULT_RESIZE_SIZE,
        imagesize: int = DEFAULT_IMAGE_SIZE,
    ):
        super().__init__(
            root=source + "/" + cls + "/" + "train",
            transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ])
        )
        self.cls = cls
        self.source = source
        self.class_root = str(Path(source) / cls)
        self.cache_root = str(_class_cache_root(cache_root, cls))
        self.resize = resize
        self.size = imagesize

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        components = _load_components(self.class_root, path, self.cache_root)
        return {
            "image_raw": sample,
            "components": components,
            "label": label,
            "path": path,
        }


class MVTecTestDataset(ImageFolder):
    """
        This class allows the creation of a instance of MVTecTestDataset. It takes four parameters : 

        - cls : Representing the name of the dataset class considered

        - source : String representing the path of the dataset

         - resize : Integer representing the size to resize the images (the default value is 256)

         - imagesize : Integer representing the size of the images (the default value is 224)

        In this method, several transformations to the images are performed. Resizing the image to the specified resize size, cropping the images 
        to the specified imagesize size, converting the image to a Pytorch tensor and normalizing the image with a mean and std value. 

        Then the __getitem__ method is used to get a sample from the dataset. It takes only one parameter, index, an integer that represents 
        the index of the sample to be retrieved.
    """
    def __init__(
        self,
        cls: str,
        source: str = DATASETS_PATH,
        cache_root: str = DECOMPOSITION_CACHE_ROOT,
        resize: int = DEFAULT_RESIZE_SIZE,
        imagesize: int = DEFAULT_IMAGE_SIZE,
    ):
        super().__init__(
            root=source + "/" + cls + "/" + "test",
            transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]),
            target_transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ])
        )
        self.cls = cls
        self.source = source
        self.class_root = str(Path(source) / cls)
        self.cache_root = str(_class_cache_root(cache_root, cls))
        self.resize = resize
        self.size = imagesize

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:
            target = Image.new('RGB', (self.size, self.size))
            sample_class = 0

        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            target = self.loader(target_path)
            sample_class = 1

        # Application of the transformations
        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        components = _load_components(self.class_root, path, self.cache_root)
        return {
            "image_raw": sample,
            "mask": target[:1],
            "components": components,
            "label": sample_class,
            "path": path,
        }


if __name__ == "__main__":
    dataset = MVTecDataset("cable", source=DATASETS_PATH, cache_root=DECOMPOSITION_CACHE_ROOT)
    train_ds, test_ds = dataset.get_datasets()

    train_sample = train_ds[0]
    train_cache_path = _decomposition_path(train_ds.class_root, train_sample["path"], train_ds.cache_root)
    train_cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)

    print("=== train sample diagnostic ===")
    print(f"path:              {train_sample['path']}")
    print(f"cache path:        {train_cache_path}")
    print(f"cache source_path: {train_cache['source_path']}")
    print(f"path match:        {train_sample['path'] == train_cache['source_path']}")
    print(f"image_raw shape:   {tuple(train_sample['image_raw'].shape)}")
    print(f"label:             {train_sample['label']}")
    print(f"component keys:    {sorted(train_sample['components'].keys())}")
    for key, value in train_sample["components"].items():
        print(f"{key:16s} shape={tuple(value.shape)}")
    print(
        "image diff vs cache:",
        float(torch.mean(torch.abs(train_sample["image_raw"] - train_cache["image"]))),
    )
    for key in train_sample["components"]:
        diff = torch.mean(torch.abs(train_sample["components"][key] - train_cache["components"][key]))
        print(f"{key:16s} diff={float(diff):.8f}")

    anomalous_test_sample = None
    for index in range(len(test_ds)):
        candidate = test_ds[index]
        if candidate["label"] == 1:
            anomalous_test_sample = candidate
            break

    if anomalous_test_sample is not None:
        test_cache_path = _decomposition_path(test_ds.class_root, anomalous_test_sample["path"], test_ds.cache_root)
        test_cache = torch.load(test_cache_path, map_location="cpu", weights_only=False)
        print("\n=== anomalous test sample diagnostic ===")
        print(f"path:              {anomalous_test_sample['path']}")
        print(f"cache path:        {test_cache_path}")
        print(f"cache source_path: {test_cache['source_path']}")
        print(f"path match:        {anomalous_test_sample['path'] == test_cache['source_path']}")
        print(f"image_raw shape:   {tuple(anomalous_test_sample['image_raw'].shape)}")
        print(f"mask shape:        {tuple(anomalous_test_sample['mask'].shape)}")
        print(f"label:             {anomalous_test_sample['label']}")
        print(f"component keys:    {sorted(anomalous_test_sample['components'].keys())}")
        for key, value in anomalous_test_sample["components"].items():
            print(f"{key:16s} shape={tuple(value.shape)}")
        print(
            "image diff vs cache:",
            float(torch.mean(torch.abs(anomalous_test_sample["image_raw"] - test_cache["image"]))),
        )
        for key in anomalous_test_sample["components"]:
            diff = torch.mean(torch.abs(anomalous_test_sample["components"][key] - test_cache["components"][key]))
            print(f"{key:16s} diff={float(diff):.8f}")
