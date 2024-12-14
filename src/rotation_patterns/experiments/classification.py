"""Fixed-rotation contrastive pre-training and downstream classification."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..data import FixedRotationPairDataset, build_image_dataset
from ..models import SixLayerProbe, build_contrastive_model, query_encoder
from ..reproducibility import (
    PROTOCOL_VERSION,
    config_digest,
    provenance,
    resolve_device,
    seed_everything,
    stable_job_id,
)
from ..training import (
    build_balanced_pair_dataset,
    deterministic_loader,
    evaluate_binary_probe,
    extract_pair_features,
    train_binary_probe,
    train_contrastive,
)


def _device(requested: str) -> torch.device:
    resolved = resolve_device(requested)
    try:
        device = torch.device(resolved)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device setting: {resolved!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {resolved!r} was requested but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
    if device.type == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()


def _elapsed(start: float, device: torch.device) -> float:
    _synchronize(device)
    return time.perf_counter() - start


def _source_split(
    length: int, train_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if length < 4:
        raise ValueError(
            "classification requires at least four source images so both train and test "
            "partitions can construct non-self negative pairs"
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("probe split must be a training fraction strictly between zero and one")
    train_size = int(round(length * train_fraction))
    train_size = min(max(train_size, 2), length - 2)
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(length, generator=generator)
    return order[:train_size], order[train_size:]


def _indices_digest(indices: torch.Tensor) -> str:
    payload = json.dumps(sorted(int(value) for value in indices.tolist()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _job_identity(job: dict[str, Any]) -> dict[str, Any]:
    required = ("experiment", "dataset", "method", "encoder", "angle", "seed")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"classification job is missing keys: {missing}")
    if job["experiment"] != "classification":
        raise ValueError(f"expected a classification job, received {job['experiment']!r}")
    return {
        "experiment": "classification",
        "dataset": str(job["dataset"]),
        "method": str(job["method"]),
        "encoder": str(job["encoder"]),
        "angle": float(job["angle"]),
        "seed": int(job["seed"]),
    }


def run_classification_job(config: ExperimentConfig, job: dict[str, Any]) -> dict[str, Any]:
    """Run one encoder/dataset/method/angle point and return an atomic result row."""

    identity = _job_identity(job)
    section = config.section("classification")
    pretrain_settings = dict(section["pretrain"])
    probe_settings = dict(section["probe"])
    seed = identity["seed"]
    deterministic = bool(config.raw.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    device = _device(str(config.raw.get("device", "auto")))
    workers = int(config.raw.get("num_workers", 0))
    image_size = int(section["image_size"])
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    phase_start = time.perf_counter()
    base_dataset = build_image_dataset(config, identity["dataset"], "classification")
    pair_dataset = FixedRotationPairDataset(base_dataset, image_size, identity["angle"])
    if len(pair_dataset) < 4:
        raise ValueError(
            f"dataset {identity['dataset']!r} has {len(pair_dataset)} image(s); "
            "classification requires at least four"
        )
    first_view, _ = pair_dataset[0]
    if first_view.ndim != 3:
        raise ValueError(f"prepared images must be CHW tensors, received {tuple(first_view.shape)}")
    in_channels = int(first_view.shape[0])
    timings["dataset_setup"] = _elapsed(phase_start, device)

    # A full last batch is preferable for contrastive encoders containing batch
    # normalization.  Small fixture datasets still retain their single batch.
    pretrain_batch_size = int(pretrain_settings["batch_size"])
    drop_last = len(pair_dataset) >= pretrain_batch_size
    pretrain_loader = deterministic_loader(
        pair_dataset,
        batch_size=pretrain_batch_size,
        shuffle=True,
        seed=seed + 101,
        num_workers=workers,
        device=device,
        drop_last=drop_last,
    )
    model = build_contrastive_model(
        identity["method"], identity["encoder"], in_channels, pretrain_settings
    ).to(device)

    phase_start = time.perf_counter()
    contrastive_trace = train_contrastive(model, pretrain_loader, pretrain_settings, device)
    timings["contrastive_pretraining"] = _elapsed(phase_start, device)

    # Feature extraction is non-shuffled, so tensor row i always corresponds to
    # source image i.  Splitting those rows before pair construction is the key
    # leakage barrier for the downstream task.
    feature_loader = deterministic_loader(
        pair_dataset,
        batch_size=pretrain_batch_size,
        shuffle=False,
        seed=seed + 202,
        num_workers=workers,
        device=device,
        drop_last=False,
    )
    phase_start = time.perf_counter()
    encoder = query_encoder(model)
    # Retain only the frozen query encoder.  In particular, release MoCo's key
    # encoder and queue before materializing full-dataset feature tensors.
    del model
    original_features, rotated_features = extract_pair_features(encoder, feature_loader, device)
    timings["frozen_feature_extraction"] = _elapsed(phase_start, device)
    if original_features.shape[0] != len(pair_dataset):
        raise RuntimeError(
            "feature extraction changed the source count: "
            f"expected {len(pair_dataset)}, received {original_features.shape[0]}"
        )

    train_sources, test_sources = _source_split(
        len(pair_dataset), float(probe_settings["split"]), seed + 303
    )
    if set(train_sources.tolist()) & set(test_sources.tolist()):
        raise AssertionError("train/test source partitions overlap")
    train_pairs = build_balanced_pair_dataset(
        original_features, rotated_features, train_sources, seed=seed + 404
    )
    test_pairs = build_balanced_pair_dataset(
        original_features, rotated_features, test_sources, seed=seed + 505
    )
    probe_batch_size = int(probe_settings["batch_size"])
    train_loader = deterministic_loader(
        train_pairs,
        batch_size=probe_batch_size,
        shuffle=True,
        seed=seed + 606,
        # These features are already resident in memory.  Keeping pair assembly
        # in-process avoids replicating large feature tensors in spawned workers.
        num_workers=0,
        device=device,
    )
    test_loader = deterministic_loader(
        test_pairs,
        batch_size=probe_batch_size,
        shuffle=False,
        seed=seed + 707,
        num_workers=0,
        device=device,
    )

    # Give every probe the same initialization for a given job seed, independent
    # of how much random state the selected encoder consumed during pre-training.
    seed_everything(seed + 808, deterministic=deterministic)
    feature_dim = int(original_features.shape[1])
    probe = SixLayerProbe(2 * feature_dim, probe_settings["hidden_dims"]).to(device)
    phase_start = time.perf_counter()
    probe_trace = train_binary_probe(probe, train_loader, probe_settings, device)
    timings["probe_training"] = _elapsed(phase_start, device)

    phase_start = time.perf_counter()
    evaluation = evaluate_binary_probe(probe, test_loader, device)
    timings["probe_evaluation"] = _elapsed(phase_start, device)
    timings["total"] = _elapsed(total_start, device)

    source_provenance = provenance()
    source_provenance.update(
        {
            "config_path": str(Path(config.path).resolve()),
            "config_sha256": config_digest(config.raw),
            "profile": config.raw.get("profile"),
            "deterministic": deterministic,
            "requested_device": str(config.raw.get("device", "auto")),
            "resolved_device": str(device),
            "dataset_class": type(base_dataset).__name__,
            "pair_dataset_class": type(pair_dataset).__name__,
            "train_source_digest": _indices_digest(train_sources),
            "test_source_digest": _indices_digest(test_sources),
        }
    )
    losses = {
        "contrastive_final": contrastive_trace.final_loss,
        "contrastive_by_epoch": list(contrastive_trace.epoch_losses),
        "probe_train_final": probe_trace.final_loss,
        "probe_train_by_epoch": list(probe_trace.epoch_losses),
        "probe_test": evaluation.loss,
    }
    counts = {
        "source_images": len(pair_dataset),
        "train_source_images": int(train_sources.numel()),
        "test_source_images": int(test_sources.numel()),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "feature_dim": feature_dim,
        "contrastive_examples_seen": contrastive_trace.examples_seen,
        "contrastive_optimizer_steps": contrastive_trace.optimizer_steps,
        "probe_examples_seen": probe_trace.examples_seen,
        "probe_optimizer_steps": probe_trace.optimizer_steps,
        "test_correct": evaluation.correct,
    }
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
        "metric_name": "accuracy",
        "metric_value": evaluation.accuracy,
        "accuracy": evaluation.accuracy,
        "loss": evaluation.loss,
        "pretrain_loss": contrastive_trace.final_loss,
        "probe_train_loss": probe_trace.final_loss,
        "test_loss": evaluation.loss,
        "losses": losses,
        "counts": counts,
        "timing_seconds": timings,
        "provenance": source_provenance,
        "settings": {
            "image_size": image_size,
            "num_workers": workers,
            "pretrain": pretrain_settings,
            "probe": probe_settings,
            "negative_pairing": "seeded_single_cycle_derangement_within_source_split",
        },
    }
