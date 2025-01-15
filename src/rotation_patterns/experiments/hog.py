"""Grouped HOG/RBF-SVM shortcut experiment."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage import __version__ as skimage_version
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.svm import SVC

from ..config import ExperimentConfig
from ..data import FixedRotationPairDataset, build_image_dataset
from ..reproducibility import (
    PROTOCOL_VERSION,
    config_digest,
    provenance,
    seed_everything,
    stable_job_id,
)


def _job_identity(job: dict[str, Any]) -> dict[str, Any]:
    required = ("experiment", "dataset", "method", "encoder", "angle", "seed")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"HOG job is missing keys: {missing}")
    if job["experiment"] != "hog":
        raise ValueError(f"expected a HOG job, received {job['experiment']!r}")
    if job["method"] != "hog_svm" or job["encoder"] != "hog":
        raise ValueError("the HOG protocol requires method='hog_svm' and encoder='hog'")
    return {
        "experiment": "hog",
        "dataset": str(job["dataset"]),
        "method": "hog_svm",
        "encoder": "hog",
        "angle": float(job["angle"]),
        "seed": int(job["seed"]),
    }


def _group_digest(groups: np.ndarray) -> str:
    payload = json.dumps(sorted(int(value) for value in np.unique(groups)), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _unnormalized_grayscale(image: torch.Tensor) -> np.ndarray:
    """Undo ``prepare_view`` normalization and produce one HOG intensity plane."""

    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        shape = getattr(image, "shape", None)
        raise ValueError(f"prepared HOG images must be CHW tensors, received {shape}")
    value = image.detach().to(device="cpu", dtype=torch.float32)
    channels = int(value.shape[0])
    if channels == 3:
        mean = value.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = value.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    else:
        mean = value.new_full((channels, 1, 1), 0.5)
        std = value.new_full((channels, 1, 1), 0.5)
    array = (value * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    if channels == 3:
        grayscale = rgb2gray(array)
    elif channels == 1:
        grayscale = array[:, :, 0]
    else:
        # BraTS has four MRI modalities rather than RGB channels.  The paper
        # leaves this conversion unspecified; an equal channel mean is the
        # deterministic analogue of grayscale conversion used here.
        grayscale = array.mean(axis=2)
    result = np.asarray(grayscale, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("grayscale conversion produced an invalid HOG image")
    return result


def _hog_features(image: torch.Tensor, settings: dict[str, Any]) -> np.ndarray:
    features = hog(
        _unnormalized_grayscale(image),
        orientations=int(settings["orientations"]),
        pixels_per_cell=tuple(int(value) for value in settings["pixels_per_cell"]),
        cells_per_block=tuple(int(value) for value in settings["cells_per_block"]),
        block_norm="L2-Hys",
        visualize=False,
        transform_sqrt=False,
        feature_vector=True,
        channel_axis=None,
    )
    result = np.asarray(features, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("HOG extraction produced an empty or non-finite feature vector")
    return result


def _extract_grouped_features(
    pair_dataset: FixedRotationPairDataset,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    expected_width: int | None = None
    for source_index in range(len(pair_dataset)):
        original, rotated = pair_dataset[source_index]
        for image, label in ((original, 0), (rotated, 1)):
            features = _hog_features(image, settings)
            if expected_width is None:
                expected_width = int(features.size)
            elif features.size != expected_width:
                raise ValueError("HOG feature width changed between source images")
            rows.append(features)
            labels.append(label)
            groups.append(source_index)
    if not rows:
        raise ValueError("cannot extract HOG features from an empty dataset")
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


def _json_parameter(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_hog_job(config: ExperimentConfig, job: dict[str, Any]) -> dict[str, Any]:
    """Fit a grouped cross-validated RBF SVM and score held-out source groups."""

    identity = _job_identity(job)
    section = config.section("hog")
    seed = identity["seed"]
    deterministic = bool(config.raw.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    phase_start = time.perf_counter()
    base_dataset = build_image_dataset(config, identity["dataset"], "hog")
    if len(base_dataset) < 3:
        raise ValueError(
            "HOG requires at least three source images for grouped train/test and "
            "cross-validation partitions"
        )
    pair_dataset = FixedRotationPairDataset(
        base_dataset, int(section["image_size"]), identity["angle"]
    )
    features, labels, groups = _extract_grouped_features(pair_dataset, section)
    timings["dataset_and_hog_extraction"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    test_fraction = float(section.get("test_fraction", 0.2))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("HOG test_fraction must be strictly between zero and one")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + 101)
    train_rows, test_rows = next(splitter.split(features, labels, groups))
    train_groups = groups[train_rows]
    test_groups = groups[test_rows]
    if set(np.unique(train_groups).tolist()) & set(np.unique(test_groups).tolist()):
        raise AssertionError("HOG source groups leaked across train and held-out test sets")
    if set(np.unique(labels[train_rows]).tolist()) != {0, 1}:
        raise ValueError("HOG training split does not contain both original and rotated labels")
    if set(np.unique(labels[test_rows]).tolist()) != {0, 1}:
        raise ValueError("HOG test split does not contain both original and rotated labels")

    requested_folds = int(section["cv_folds"])
    unique_train_groups = int(np.unique(train_groups).size)
    effective_folds = min(requested_folds, unique_train_groups)
    if effective_folds < 2:
        raise ValueError("HOG training split needs at least two source groups for GridSearchCV")
    cross_validation = GroupKFold(n_splits=effective_folds)
    search = GridSearchCV(
        estimator=SVC(kernel="rbf"),
        param_grid={"C": list(section["c_grid"]), "gamma": list(section["gamma_grid"])},
        scoring="accuracy",
        cv=cross_validation,
        refit=True,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(features[train_rows], labels[train_rows], groups=train_groups)
    timings["grouped_grid_search"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    predictions = search.predict(features[test_rows])
    accuracy = float(accuracy_score(labels[test_rows], predictions))
    correct = int(np.sum(predictions == labels[test_rows]))
    timings["held_out_test_evaluation"] = time.perf_counter() - phase_start
    timings["total"] = time.perf_counter() - total_start

    best_parameters = {
        key: _json_parameter(value) for key, value in sorted(search.best_params_.items())
    }
    source_provenance = provenance()
    source_provenance.update(
        {
            "config_path": str(Path(config.path).resolve()),
            "config_sha256": config_digest(config.raw),
            "profile": config.raw.get("profile"),
            "deterministic": deterministic,
            "dataset_class": type(base_dataset).__name__,
            "scikit_image": skimage_version,
            "scikit_learn": sklearn_version,
            "train_source_digest": _group_digest(train_groups),
            "test_source_digest": _group_digest(test_groups),
        }
    )
    fingerprint = config_digest(config.raw)
    canonical_identity = {
        **identity,
        "config_sha256": fingerprint,
        "protocol_version": PROTOCOL_VERSION,
    }
    return {
        "job_id": str(job.get("job_id") or stable_job_id(canonical_identity)),
        "config_sha256": fingerprint,
        "protocol_version": PROTOCOL_VERSION,
        **identity,
        "metric_name": "hog_svm_accuracy",
        "metric_value": accuracy,
        "accuracy": accuracy,
        "best_cv_accuracy": float(search.best_score_),
        "best_parameters": best_parameters,
        "counts": {
            "source_images": len(pair_dataset),
            "feature_rows": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "train_source_images": unique_train_groups,
            "test_source_images": int(np.unique(test_groups).size),
            "train_rows": int(train_rows.size),
            "test_rows": int(test_rows.size),
            "test_correct": correct,
            "cv_candidates": len(search.cv_results_["params"]),
            "requested_cv_folds": requested_folds,
            "effective_cv_folds": effective_folds,
        },
        "timing_seconds": timings,
        "provenance": source_provenance,
        "settings": {
            "image_size": int(section["image_size"]),
            "orientations": int(section["orientations"]),
            "pixels_per_cell": [int(value) for value in section["pixels_per_cell"]],
            "cells_per_block": [int(value) for value in section["cells_per_block"]],
            "block_norm": "L2-Hys",
            "grayscale_conversion": "rgb2gray_for_rgb_channel_mean_otherwise",
            "test_fraction": test_fraction,
            "cv_folds": requested_folds,
            "c_grid": list(section["c_grid"]),
            "gamma_grid": list(section["gamma_grid"]),
            "kernel": "rbf",
            "split": "grouped_by_source_image",
        },
    }
