"""Dataset adapters and the exact fixed-rotation view construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets as tvd
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import ExperimentConfig

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def _sample_image(sample: Any) -> Any:
    return sample[0] if isinstance(sample, (tuple, list)) else sample


def _to_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        value = image.detach().clone().float()
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.max() > 1:
            value = value / 255.0
        return value
    array = np.asarray(image)
    if not array.flags.writeable:
        array = array.copy()
    return TF.to_tensor(array)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[0]
    if channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5] * channels
        std = [0.5] * channels
    return TF.normalize(image, mean, std)


def prepare_view(image: Any, image_size: int, angle: float = 0.0) -> torch.Tensor:
    """Resize, apply the only allowed augmentation, and normalize an image."""
    value = _to_tensor(image)
    value = TF.resize(value, [image_size, image_size], antialias=True)
    if angle % 360:
        value = TF.rotate(
            value,
            float(angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0,
        )
    return normalize_image(value)


class ImageOnlyDataset(Dataset):
    def __init__(self, wrapped: Dataset):
        self.wrapped = wrapped

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, index: int) -> Any:
        return _sample_image(self.wrapped[index])


class RecursiveImageDataset(Dataset):
    def __init__(self, root: Path, grayscale: bool = False):
        self.paths = sorted(path for pattern in IMAGE_EXTENSIONS for path in root.rglob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"no supported image files found below {root}")
        self.mode = "L" if grayscale else "RGB"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        with Image.open(self.paths[index]) as image:
            return image.convert(self.mode).copy()


class SyntheticRotationDataset(Dataset):
    """Small deterministic fixture with visible orientation structure."""

    def __init__(self, size: int = 32, length: int = 32, variant: int = 0):
        self.size = size
        self.length = length
        self.variant = variant

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(10_000 * self.variant + index)
        image = torch.zeros(3, self.size, self.size)
        width = 2 + (index % 3)
        position = 3 + (index * 5) % max(self.size - 7, 1)
        if self.variant % 2 == 0:
            image[0, :, position : position + width] = 0.9
            image[1, position : position + width, :] = 0.55
        else:
            diagonal = torch.arange(self.size)
            image[1, diagonal, torch.roll(diagonal, shifts=index % self.size)] = 0.9
            image[2, :, position : position + width] = 0.55
        image += 0.03 * torch.rand(image.shape, generator=generator)
        return image.clamp(0, 1)


class FixedRotationPairDataset(Dataset):
    def __init__(self, base: Dataset, image_size: int, angle: float):
        self.base = base
        self.image_size = image_size
        self.angle = angle

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = _sample_image(self.base[index])
        return prepare_view(image, self.image_size), prepare_view(image, self.image_size, self.angle)


def _torchvision_dataset(name: str, root: Path) -> Dataset | None:
    root.mkdir(parents=True, exist_ok=True)
    if name == "cifar10":
        return ConcatDataset(
            [tvd.CIFAR10(root, train=True, download=True), tvd.CIFAR10(root, train=False, download=True)]
        )
    if name == "mnist":
        return ConcatDataset(
            [tvd.MNIST(root, train=True, download=True), tvd.MNIST(root, train=False, download=True)]
        )
    if name == "fashion_mnist":
        return ConcatDataset(
            [
                tvd.FashionMNIST(root, train=True, download=True),
                tvd.FashionMNIST(root, train=False, download=True),
            ]
        )
    if name == "svhn":
        return ConcatDataset(
            [tvd.SVHN(root, split="train", download=True), tvd.SVHN(root, split="test", download=True)]
        )
    if name == "eurosat":
        return tvd.EuroSAT(root, download=True)
    if name == "caltech101":
        return tvd.Caltech101(root, download=True)
    if name == "flowers102":
        return ConcatDataset(
            [tvd.Flowers102(root, split=split, download=True) for split in ("train", "val", "test")]
        )
    if name == "food101":
        return ConcatDataset(
            [tvd.Food101(root, split="train", download=True), tvd.Food101(root, split="test", download=True)]
        )
    return None


def _limit_dataset(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
    if not maximum or len(dataset) <= maximum:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def build_image_dataset(
    config: ExperimentConfig, name: str, section_name: str = "classification"
) -> Dataset:
    section = config.section(section_name)
    maximum = section.get("max_samples")
    if name.startswith("synthetic_"):
        variant = 0 if name.endswith("a") else 1
        dataset: Dataset = SyntheticRotationDataset(
            size=int(section["image_size"]), length=int(maximum or 32), variant=variant
        )
        return dataset

    dataset_root = config.data_root / name
    built_in = _torchvision_dataset(name, dataset_root)
    if built_in is not None:
        return _limit_dataset(ImageOnlyDataset(built_in), maximum, config.seed)
    if name == "brats":
        dataset = BraTSSliceDataset(dataset_root, include_mask=False)
    else:
        grayscale = name in {"lung_mask", "chest_xray14"}
        image_root = dataset_root / "images" if (dataset_root / "images").is_dir() else dataset_root
        dataset = RecursiveImageDataset(image_root, grayscale=grayscale)
    return _limit_dataset(dataset, maximum, config.seed)


def fixed_split_indices(
    length: int, validation_fraction: float, test_fraction: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be non-negative and sum to less than one")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    test_size = int(round(length * test_fraction))
    validation_size = int(round(length * validation_fraction))
    test = indices[:test_size]
    validation = indices[test_size : test_size + validation_size]
    train = indices[test_size + validation_size :]
    return train, validation, test


class PairedMaskDataset(Dataset):
    def __init__(self, root: Path, image_size: int, grayscale: bool = False):
        self.image_size = image_size
        image_root = root / "images"
        mask_root = root / "masks"
        if not image_root.is_dir() or not mask_root.is_dir():
            raise FileNotFoundError(f"expected images/ and masks/ under {root}")
        image_paths = sorted(path for pattern in IMAGE_EXTENSIONS for path in image_root.rglob(pattern))
        mask_paths = sorted(path for pattern in IMAGE_EXTENSIONS for path in mask_root.rglob(pattern))
        masks: dict[str, Path] = {}
        for path in mask_paths:
            stem = path.stem.removesuffix("_mask").removesuffix("-mask")
            masks[stem] = path
        self.pairs = [(path, masks[path.stem]) for path in image_paths if path.stem in masks]
        if not self.pairs:
            raise FileNotFoundError(f"no matching image/mask basenames under {root}")
        self.image_mode = "L" if grayscale else "RGB"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as handle:
            image = _to_tensor(handle.convert(self.image_mode))
        with Image.open(mask_path) as handle:
            mask = _to_tensor(handle.convert("L"))
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = TF.resize(
            mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST
        )
        return normalize_image(image), (mask > 0.5).float()


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.images = SyntheticRotationDataset(size=size, length=length, variant=0)
        self.size = size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        mask = (image[0:1] > 0.4).float()
        return normalize_image(image), mask


class BraTSSliceDataset(Dataset):
    """Four-modality axial BraTS slices with case-level enumeration."""

    modalities = ("flair", "t1", "t1ce", "t2")

    def __init__(
        self,
        root: Path,
        image_size: int | None = None,
        include_mask: bool = False,
        slice_index: int | None = None,
    ):
        self.root = root
        self.image_size = image_size
        self.include_mask = include_mask
        self.slice_index = slice_index
        cases = []
        for flair in sorted(root.rglob("*_flair.nii*")):
            prefix = str(flair).replace("_flair.nii.gz", "").replace("_flair.nii", "")
            modality_paths = [Path(f"{prefix}_{modality}.nii.gz") for modality in self.modalities]
            modality_paths = [
                path if path.exists() else Path(str(path).removesuffix(".gz")) for path in modality_paths
            ]
            seg = Path(f"{prefix}_seg.nii.gz")
            if not seg.exists():
                seg = Path(str(seg).removesuffix(".gz"))
            if all(path.exists() for path in modality_paths) and (seg.exists() or not include_mask):
                cases.append((modality_paths, seg))
        if not cases:
            raise FileNotFoundError(f"no complete BraTS cases found below {root}")
        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    @staticmethod
    def _scale(array: np.ndarray) -> np.ndarray:
        foreground = array[array != 0]
        if foreground.size == 0:
            return np.zeros_like(array, dtype=np.float32)
        low, high = np.percentile(foreground, [1, 99])
        return np.clip((array - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)

    def __getitem__(self, index: int) -> Any:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("BraTS support requires `pip install -e '.[medical]'`") from exc
        modality_paths, seg_path = self.cases[index]
        volumes = [np.asarray(nib.load(path).dataobj) for path in modality_paths]
        z = self.slice_index if self.slice_index is not None else volumes[0].shape[2] // 2
        image = torch.from_numpy(np.stack([self._scale(volume[:, :, z]) for volume in volumes]))
        if not self.include_mask:
            return image
        mask_array = np.asarray(nib.load(seg_path).dataobj)[:, :, z]
        mask_array = np.where(mask_array == 4, 3, mask_array).astype(np.int64)
        mask = torch.from_numpy(mask_array)
        if self.image_size:
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            mask = TF.resize(
                mask.unsqueeze(0),
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            ).squeeze(0)
        return normalize_image(image), mask


def build_segmentation_dataset(config: ExperimentConfig, name: str) -> Dataset:
    section = config.section("segmentation")
    image_size = int(section["image_size"])
    maximum = section.get("max_samples")
    if name.startswith("synthetic_"):
        dataset: Dataset = SyntheticSegmentationDataset(image_size, int(maximum or 16))
    elif name == "brats":
        dataset = BraTSSliceDataset(config.data_root / name, image_size=image_size, include_mask=True)
    elif name == "lung_mask":
        dataset = PairedMaskDataset(config.data_root / name, image_size=image_size, grayscale=True)
    elif name == "kvasir_seg":
        dataset = PairedMaskDataset(config.data_root / name, image_size=image_size, grayscale=False)
    else:
        raise ValueError(f"unsupported segmentation dataset: {name}")
    return _limit_dataset(dataset, maximum, config.seed)


class SegmentationImageView(Dataset):
    def __init__(self, base: Dataset, image_size: int, angle: float):
        self.base = base
        self.image_size = image_size
        self.angle = angle

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.base[index]
        # Segmentation datasets are already normalized, so map back to [0, 1]
        # before the fixed transform and normalize exactly once.
        channels = image.shape[0]
        if channels == 3:
            mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
            std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        else:
            mean = image.new_full((channels, 1, 1), 0.5)
            std = image.new_full((channels, 1, 1), 0.5)
        raw = (image * std + mean).clamp(0, 1)
        return prepare_view(raw, self.image_size), prepare_view(raw, self.image_size, self.angle)
