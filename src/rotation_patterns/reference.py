"""Load and stage the Figure 1 measurements bundled with this project."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np

from .config import ExperimentConfig


EXPECTED_SHAPE = (16, 16, 3600)
BUNDLED_ARTIFACT = "figure1_curves.npz"
BUNDLED_METADATA = "metadata.json"

DATASET_NAMES = (
    "CIFAR-10",
    "MNIST",
    "Fashion-MNIST",
    "Tiny ImageNet",
    "BraTS",
    "Lung Mask Image Dataset",
    "Kvasir-SEG",
    "Stanford Dogs",
    "iNaturalist",
    "PlantVillage",
    "ChestX-ray14",
    "Street View House Numbers",
    "EuroSAT",
    "Caltech-101",
    "Flowers-102",
    "Food-101",
)

ENCODER_METHOD_NAMES = (
    "MoCo ResNet-18",
    "MoCo ConvNeXt-Tiny",
    "MoCo ViT-B/16",
    "MoCo EfficientNet-B0",
    "MoCo RegNetY-400MF",
    "MoCo WideResNet-50-2",
    "MoCo MobileNetV2",
    "MoCo Swin-Tiny",
    "SimCLR ResNet-18",
    "SimCLR ConvNeXt-Tiny",
    "SimCLR ViT-B/16",
    "SimCLR EfficientNet-B0",
    "SimCLR RegNetY-400MF",
    "SimCLR WideResNet-50-2",
    "SimCLR MobileNetV2",
    "SimCLR Swin-Tiny",
)

_FILE_PATTERN = re.compile(r"^plot_([1-9]|1[0-6])_([1-9]|1[0-6])\.csv$")


@dataclass(frozen=True)
class ReferenceCurves:
    """Validated angle grid, curve tensor, and its two categorical axes."""

    angles: np.ndarray
    tensor: np.ndarray
    datasets: tuple[str, ...] = DATASET_NAMES
    encoder_methods: tuple[str, ...] = ENCODER_METHOD_NAMES

    @property
    def mapping(self) -> dict[str, Any]:
        return reference_mapping()


def reference_mapping() -> dict[str, Any]:
    """Return the published one-based file-index to label mapping."""

    return {
        "datasets": {index + 1: name for index, name in enumerate(DATASET_NAMES)},
        "encoder_methods": {
            index + 1: name for index, name in enumerate(ENCODER_METHOD_NAMES)
        },
        "filename_pattern": "plot_{dataset_index}_{encoder_method_index}.csv",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_angles() -> np.ndarray:
    return np.arange(EXPECTED_SHAPE[2], dtype=np.float64) / 10.0


def _validate_reference(
    angles: np.ndarray,
    tensor: np.ndarray,
    datasets: tuple[str, ...] = DATASET_NAMES,
    encoder_methods: tuple[str, ...] = ENCODER_METHOD_NAMES,
) -> ReferenceCurves:
    angles = np.asarray(angles, dtype=np.float64)
    tensor = np.asarray(tensor, dtype=np.float64)
    if angles.shape != (EXPECTED_SHAPE[2],):
        raise ValueError(f"angle grid has shape {angles.shape}; expected (3600,)")
    if not np.allclose(angles, _expected_angles(), rtol=0.0, atol=1e-10):
        raise ValueError("angle grid is not the published 0.0..359.9 grid")
    if tensor.shape != EXPECTED_SHAPE:
        raise ValueError(f"curve tensor has shape {tensor.shape}; expected {EXPECTED_SHAPE}")
    if not np.isfinite(tensor).all() or np.any((tensor < 0.0) | (tensor > 1.0)):
        raise ValueError("curve tensor contains non-finite accuracy or a value outside [0, 1]")
    if datasets != DATASET_NAMES or encoder_methods != ENCODER_METHOD_NAMES:
        raise ValueError("bundled curve-axis labels do not match the published experiment")
    return ReferenceCurves(angles, tensor, datasets, encoder_methods)


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one legacy two-column curve CSV for researcher-supplied imports."""

    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not parse reference curve {path}: {exc}") from exc
    if values.shape != (EXPECTED_SHAPE[2], 2):
        raise ValueError(f"reference curve {path} has shape {values.shape}; expected (3600, 2)")
    angles, accuracy = values[:, 0], values[:, 1]
    if not np.allclose(angles, _expected_angles(), rtol=0.0, atol=1e-10):
        raise ValueError(f"reference curve {path} does not use the 0.0..359.9 grid")
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError(f"reference curve {path} contains invalid accuracy values")
    return angles, accuracy


def _load_npz(path: Path) -> ReferenceCurves:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"angles", "tensor", "datasets", "encoder_methods"}
            if not required <= set(data.files):
                raise ValueError(f"bundled artifact is missing arrays: {sorted(required-set(data.files))}")
            angles = np.asarray(data["angles"], dtype=np.float64)
            tensor = np.asarray(data["tensor"], dtype=np.float64)
            datasets = tuple(str(value) for value in data["datasets"].tolist())
            encoder_methods = tuple(str(value) for value in data["encoder_methods"].tolist())
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load curve artifact {path}: {exc}") from exc
    return _validate_reference(angles, tensor, datasets, encoder_methods)


def _load_csv_directory(raw_data: Path) -> ReferenceCurves:
    expected_names = {
        f"plot_{row}_{column}.csv"
        for row in range(1, EXPECTED_SHAPE[0] + 1)
        for column in range(1, EXPECTED_SHAPE[1] + 1)
    }
    actual_names = {path.name for path in raw_data.glob("*.csv")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "curve directory must contain exactly 256 CSVs; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    tensor = np.empty(EXPECTED_SHAPE, dtype=np.float64)
    common_angles: np.ndarray | None = None
    for path in sorted(raw_data.glob("*.csv")):
        match = _FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected curve filename: {path.name}")
        row, column = (int(value) - 1 for value in match.groups())
        angles, accuracy = _read_curve(path)
        if common_angles is None:
            common_angles = angles
        elif not np.array_equal(angles, common_angles):
            raise ValueError(f"angle grid differs in {path}")
        tensor[row, column] = accuracy
    if common_angles is None:
        raise ValueError(f"no curve CSVs found under {raw_data}")
    return _validate_reference(common_angles, tensor)


def _reference_root(source: ExperimentConfig | str | Path) -> Path:
    if isinstance(source, ExperimentConfig):
        return source.output_root / "reference"
    return Path(source)


def load_reference(source: ExperimentConfig | str | Path) -> ReferenceCurves:
    """Load the staged NPZ artifact or an exact 256-CSV researcher export."""

    root = _reference_root(source)
    if root.is_file() and root.suffix == ".npz":
        metadata_path = root.parent / BUNDLED_METADATA
        if metadata_path.is_file():
            metadata = _metadata(metadata_path)
            if metadata.get("artifact") == root.name:
                return _load_checked_artifact(root, metadata)
        return _load_npz(root)
    artifact = root / BUNDLED_ARTIFACT
    if artifact.is_file():
        metadata_path = root / BUNDLED_METADATA
        if not metadata_path.is_file():
            raise FileNotFoundError(f"checksum metadata not found at {metadata_path}")
        return _load_checked_artifact(artifact, _metadata(metadata_path))
    raw_data = root if root.name == "raw_data" else root / "raw_data"
    if raw_data.is_dir():
        return _load_csv_directory(raw_data)
    raise FileNotFoundError(
        f"curve measurements not found at {root}; run "
        "`rotation-patterns fetch-reference --config ...` first"
    )


def load_reference_tensor(source: ExperimentConfig | str | Path) -> np.ndarray:
    """Return only the validated ``16 x 16 x 3600`` accuracy tensor."""

    return load_reference(source).tensor


def _bundled_file(name: str):
    return resources.files("rotation_patterns.reference_data").joinpath(name)


def _metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact metadata must be a JSON object: {path}")
    return value


def bundled_reference_metadata() -> dict[str, Any]:
    """Return a copy of the checksum metadata shipped with the measurements."""

    with resources.as_file(_bundled_file(BUNDLED_METADATA)) as metadata_path:
        return dict(_metadata(Path(metadata_path)))


def _load_checked_artifact(artifact: Path, metadata: dict[str, Any]) -> ReferenceCurves:
    if metadata.get("artifact") != artifact.name:
        raise ValueError("Figure 1 artifact name does not match metadata")
    if _sha256(artifact) != metadata.get("artifact_sha256"):
        raise ValueError("Figure 1 artifact checksum does not match metadata")
    curves = _load_npz(artifact)
    tensor_sha256 = hashlib.sha256(curves.tensor.tobytes(order="C")).hexdigest()
    if tensor_sha256 != metadata.get("tensor_sha256"):
        raise ValueError("Figure 1 tensor checksum does not match metadata")
    return curves


def load_bundled_reference() -> ReferenceCurves:
    """Load the lossless original measurements included in the Python package."""

    with resources.as_file(_bundled_file(BUNDLED_ARTIFACT)) as artifact_path:
        artifact = Path(artifact_path)
        metadata = bundled_reference_metadata()
        return _load_checked_artifact(artifact, metadata)


def fetch_reference(config: ExperimentConfig) -> Path:
    """Stage the bundled measurements under the configured output directory."""

    # Validate the package artifact before reading or copying any existing staged data.
    load_bundled_reference()
    destination = config.output_root / "reference"
    if (destination / BUNDLED_ARTIFACT).is_file() and (destination / BUNDLED_METADATA).is_file():
        try:
            staged_metadata = _metadata(destination / BUNDLED_METADATA)
            with resources.as_file(_bundled_file(BUNDLED_METADATA)) as metadata_path:
                bundled_metadata = _metadata(Path(metadata_path))
            if staged_metadata == bundled_metadata:
                _load_checked_artifact(destination / BUNDLED_ARTIFACT, staged_metadata)
                return destination
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reference-", dir=destination.parent) as temp_name:
        temporary = Path(temp_name)
        staged = temporary / "staged"
        staged.mkdir()
        for name in (BUNDLED_ARTIFACT, BUNDLED_METADATA):
            with resources.as_file(_bundled_file(name)) as source_path:
                shutil.copy2(source_path, staged / name)
        load_reference(staged)

        backup = temporary / "previous"
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staged, destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
    return destination
