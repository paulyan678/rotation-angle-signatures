"""Fetch and validate my published Figure 1 measurement archive.

I download the curve CSVs from an immutable commit of ``paulyan678/thesis`` and keep the
published measurements separate from new experiment aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from . import __version__
from .config import ExperimentConfig


REFERENCE_COMMIT = "bf9d88733752448c193b7b43356a7e083b021a7b"
REFERENCE_URL = (
    f"https://codeload.github.com/paulyan678/thesis/zip/{REFERENCE_COMMIT}"
)
EXPECTED_SHAPE = (16, 16, 3600)
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 1024 * 1024
MAX_REFERENCE_BYTES = 64 * 1024 * 1024

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

_MEMBER_PATTERN = re.compile(
    r"^[^/]+/raw_data/plot_([1-9]|1[0-6])_([1-9]|1[0-6])\.csv$"
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


def _raw_data_directory(source: ExperimentConfig | str | Path) -> Path:
    if isinstance(source, ExperimentConfig):
        root = source.output_root / "reference"
    else:
        root = Path(source)
    return root if root.name == "raw_data" else root / "raw_data"


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not parse reference curve {path}: {exc}") from exc
    if values.shape != (EXPECTED_SHAPE[2], 2):
        raise ValueError(
            f"reference curve {path} has shape {values.shape}; expected (3600, 2)"
        )
    angles, accuracy = values[:, 0], values[:, 1]
    expected_angles = np.arange(EXPECTED_SHAPE[2], dtype=np.float64) / 10.0
    if not np.allclose(angles, expected_angles, rtol=0.0, atol=1e-10):
        raise ValueError(f"reference curve {path} does not use the 0.0..359.9 grid")
    if not np.isfinite(accuracy).all():
        raise ValueError(f"reference curve {path} contains non-finite accuracies")
    if np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError(f"reference curve {path} contains accuracy outside [0, 1]")
    return angles, accuracy


def load_reference(source: ExperimentConfig | str | Path) -> ReferenceCurves:
    """Load the complete published tensor and fail on missing or extra curve CSVs."""

    raw_data = _raw_data_directory(source)
    if not raw_data.is_dir():
        raise FileNotFoundError(
            f"reference data not found at {raw_data}; run `rotation-patterns "
            "fetch-reference --config ...` first"
        )
    csv_paths = sorted(raw_data.glob("*.csv"))
    expected_names = {
        f"plot_{row}_{column}.csv"
        for row in range(1, EXPECTED_SHAPE[0] + 1)
        for column in range(1, EXPECTED_SHAPE[1] + 1)
    }
    actual_names = {path.name for path in csv_paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "reference directory must contain exactly 256 curve CSVs; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    tensor = np.empty(EXPECTED_SHAPE, dtype=np.float64)
    common_angles: np.ndarray | None = None
    for path in csv_paths:
        match = _FILE_PATTERN.fullmatch(path.name)
        if match is None:  # Protected by the exact-name test above.
            raise ValueError(f"unexpected reference filename: {path.name}")
        row, column = (int(value) - 1 for value in match.groups())
        angles, accuracy = _read_curve(path)
        if common_angles is None:
            common_angles = angles
        elif not np.array_equal(angles, common_angles):
            raise ValueError(f"angle grid differs in {path}")
        tensor[row, column] = accuracy

    if tensor.shape != EXPECTED_SHAPE or common_angles is None:
        raise AssertionError("reference tensor construction failed")
    return ReferenceCurves(common_angles, tensor)


def load_reference_tensor(source: ExperimentConfig | str | Path) -> np.ndarray:
    """Convenience API returning only the validated ``16 x 16 x 3600`` tensor."""

    return load_reference(source).tensor


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    selected: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for info in archive.infolist():
        # Reject dangerous paths before considering a member for extraction.
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe path in reference archive: {info.filename!r}")
        match = _MEMBER_PATTERN.fullmatch(info.filename)
        if match is None:
            continue
        if info.is_dir() or info.external_attr >> 16 & 0o170000 == 0o120000:
            raise ValueError(f"reference CSV is not a regular file: {info.filename!r}")
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted reference member is not allowed: {info.filename!r}")
        if info.file_size > MAX_MEMBER_BYTES or info.compress_size > MAX_MEMBER_BYTES:
            raise ValueError(f"reference member exceeds the size limit: {info.filename!r}")
        total_size += info.file_size
        if total_size > MAX_REFERENCE_BYTES:
            raise ValueError("reference CSVs exceed the total uncompressed size limit")
        filename = f"plot_{int(match.group(1))}_{int(match.group(2))}.csv"
        if filename in selected:
            raise ValueError(f"duplicate reference member for {filename}")
        selected[filename] = info
    if len(selected) != EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]:
        raise ValueError(
            f"pinned archive contains {len(selected)} reference CSVs; expected 256"
        )
    return selected


def fetch_reference(config: ExperimentConfig) -> Path:
    """Fetch only the 256 published CSVs, install atomically, and validate them."""

    destination = config.output_root / "reference"
    try:
        load_reference(destination)
        with (destination / "metadata.json").open(encoding="utf-8") as handle:
            installed_metadata = json.load(handle)
        if installed_metadata.get("commit") == REFERENCE_COMMIT:
            return destination
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, AttributeError):
        pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reference-", dir=destination.parent) as temp_name:
        temporary = Path(temp_name)
        archive_path = temporary / "reference.zip"
        request = Request(
            REFERENCE_URL, headers={"User-Agent": f"rotation-patterns/{__version__}"}
        )
        try:
            with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > MAX_ARCHIVE_BYTES:
                    raise ValueError("reference archive exceeds the download size limit")
                copied = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > MAX_ARCHIVE_BYTES:
                        raise ValueError("reference archive exceeds the download size limit")
                    output.write(chunk)
        except OSError as exc:
            raise RuntimeError(f"failed to fetch reference archive from {REFERENCE_URL}") from exc

        staged = temporary / "staged"
        raw_data = staged / "raw_data"
        raw_data.mkdir(parents=True)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = _archive_members(archive)
                for filename, info in sorted(members.items()):
                    target = raw_data / filename
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError("downloaded reference archive is not a valid ZIP file") from exc

        load_reference(staged)
        metadata = {
            "source": "https://github.com/paulyan678/thesis",
            "commit": REFERENCE_COMMIT,
            "archive_url": REFERENCE_URL,
            "csv_count": 256,
            "tensor_shape": list(EXPECTED_SHAPE),
            "mapping": reference_mapping(),
        }
        with (staged / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        # The directory is owned by this command.  os.replace keeps valid prior data
        # untouched until the newly staged reference has passed every validation.
        backup = temporary / "previous"
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staged, destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise

    load_reference(destination)
    return destination
