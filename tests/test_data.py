import torch
from PIL import Image

from rotation_patterns.config import ExperimentConfig, load_config
from rotation_patterns.data import (
    FixedRotationPairDataset,
    SyntheticRotationDataset,
    _torchvision_dataset,
    build_image_dataset,
    build_segmentation_dataset,
    fixed_split_indices,
    prepare_view,
)


def test_fixed_rotation_views_are_shaped_and_distinct() -> None:
    base = SyntheticRotationDataset(size=32, length=4)
    original, rotated = FixedRotationPairDataset(base, image_size=32, angle=90)[0]
    assert original.shape == rotated.shape == (3, 32, 32)
    assert not torch.allclose(original, rotated)


def test_zero_and_full_rotation_are_identity_under_convention() -> None:
    image = SyntheticRotationDataset(size=32, length=1)[0]
    zero = prepare_view(image, 32, 0)
    full = prepare_view(image, 32, 360)
    assert torch.equal(zero, full)


def test_split_is_disjoint_and_deterministic() -> None:
    first = fixed_split_indices(101, validation_fraction=0.1, test_fraction=0.2, seed=7)
    second = fixed_split_indices(101, validation_fraction=0.1, test_fraction=0.2, seed=7)
    assert first == second
    train, validation, test = first
    assert len(set(train) & set(validation)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(train) + len(validation) + len(test) == 101


def test_synthetic_segmentation_fixture() -> None:
    dataset = build_segmentation_dataset(load_config("configs/smoke.yaml"), "synthetic_a")
    image, mask = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_tiny_imagenet_uses_only_official_training_directory(tmp_path) -> None:
    root = tmp_path / "tiny_imagenet"
    for split in ("train", "val"):
        directory = root / split / "class_a"
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8), color="white").save(directory / f"{split}.png")
    base = load_config("configs/smoke.yaml")
    raw = {**base.raw, "data_root": str(tmp_path)}
    config = ExperimentConfig(base.path, raw)
    dataset = build_image_dataset(config, "tiny_imagenet")
    assert len(dataset) == 1


def test_caltech101_uses_full_manual_image_folder(tmp_path) -> None:
    assert _torchvision_dataset("caltech101", tmp_path) is None
