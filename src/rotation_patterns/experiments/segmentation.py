"""Fixed-angle MoCo pre-training followed by supervised U-Net segmentation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ..config import ExperimentConfig
from ..data import SegmentationImageView, build_segmentation_dataset, fixed_split_indices
from ..models import (
    TinyEncoder,
    TinyFeaturePyramid,
    UNet,
    build_contrastive_model,
    macro_dice,
    query_encoder,
    soft_dice_loss,
)
from ..reproducibility import (
    PROTOCOL_VERSION,
    config_digest,
    provenance,
    resolve_device,
    seed_everything,
    stable_job_id,
)
from ..training import deterministic_loader, train_contrastive


def _device(requested: str) -> torch.device:
    resolved = resolve_device(requested)
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {resolved!r} was requested but CUDA is unavailable")
    if resolved == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    try:
        return torch.device(resolved)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device setting: {resolved!r}") from exc


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


def _job_identity(job: dict[str, Any]) -> dict[str, Any]:
    required = ("experiment", "dataset", "method", "encoder", "angle", "seed")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"segmentation job is missing keys: {missing}")
    if job["experiment"] != "segmentation":
        raise ValueError(f"expected a segmentation job, received {job['experiment']!r}")
    if job["method"] != "mocov2":
        raise ValueError("the segmentation protocol requires MoCo v2 pre-training")
    return {
        "experiment": "segmentation",
        "dataset": str(job["dataset"]),
        "method": "mocov2",
        "encoder": str(job["encoder"]),
        "angle": float(job["angle"]),
        "seed": int(job["seed"]),
    }


def _indices_digest(indices: list[int]) -> str:
    payload = json.dumps(sorted(int(value) for value in indices), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _validate_partitions(
    dataset_length: int,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if dataset_length < 3:
        raise ValueError(
            "segmentation requires at least three items for distinct train, validation, "
            "and held-out test partitions"
        )
    train, validation, test = fixed_split_indices(
        dataset_length, validation_fraction, test_fraction, seed
    )
    if not train or not validation or not test:
        raise ValueError(
            "the configured segmentation fractions produced an empty partition; increase the "
            "dataset size or the validation/test fractions"
        )
    partitions = [set(train), set(validation), set(test)]
    if any(
        partitions[left] & partitions[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError("segmentation partitions overlap")
    return train, validation, test


def _task_shape(dataset: Dataset, dataset_name: str) -> tuple[int, int]:
    image, mask = dataset[0]
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        shape = getattr(image, "shape", None)
        raise ValueError(f"segmentation images must be CHW tensors, received {shape}")
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
        shape = getattr(mask, "shape", None)
        raise ValueError(f"segmentation masks must be HW or 1HW tensors, received {shape}")
    if mask.ndim == 3 and mask.shape[0] != 1:
        raise ValueError(f"binary masks must have one channel, received {tuple(mask.shape)}")
    # BraTS labels are known a priori to be {0,1,2,3}; inferring the number of
    # classes from one middle slice would fail for slices without every tumour type.
    classes = 4 if dataset_name == "brats" else 1
    return int(image.shape[0]), classes


def _copy_matching_state(
    source: nn.Module,
    target: nn.Module,
) -> dict[str, Any]:
    """Copy compatible query-encoder tensors into the U-Net feature encoder.

    timm classification and ``features_only`` wrappers normally retain identical
    backbone state keys, so those are copied by exact key and shape.  The smoke
    encoder and feature pyramid are intentionally different classes; its two
    learned stages have explicit semantic aliases below.
    """

    source_component = getattr(source, "model", source)
    target_component = getattr(target, "model", target)
    source_state = source_component.state_dict()
    target_state = target_component.state_dict()
    matches: dict[str, str] = {}
    for target_key, target_value in target_state.items():
        unwrapped = target_key.removeprefix("model.")
        candidates = [target_key, unwrapped]
        # timm's FeatureListNet flattens selected top-level Sequential
        # modules for ConvNeXt (stem_0 -> stem.0, stages_0 -> stages.0).
        for prefix in ("stem_", "stages_"):
            if unwrapped.startswith(prefix):
                candidates.append(unwrapped.replace(prefix, prefix[:-1] + ".", 1))
        for source_key in candidates:
            if (
                source_key in source_state
                and source_state[source_key].shape == target_value.shape
            ):
                matches[target_key] = source_key
                break

    transfer_mode = "exact_backbone_keys"
    if isinstance(source, TinyEncoder) and isinstance(target, TinyFeaturePyramid):
        transfer_mode = "tiny_semantic_stage_aliases"
        aliases = {
            "blocks.0.0.weight": "features.0.weight",
            "blocks.0.0.bias": "features.0.bias",
            "blocks.0.1.weight": "features.1.weight",
            "blocks.0.1.bias": "features.1.bias",
            "blocks.0.1.running_mean": "features.1.running_mean",
            "blocks.0.1.running_var": "features.1.running_var",
            "blocks.0.1.num_batches_tracked": "features.1.num_batches_tracked",
            "blocks.1.0.weight": "features.4.weight",
            "blocks.1.0.bias": "features.4.bias",
            "blocks.1.1.weight": "features.5.weight",
            "blocks.1.1.bias": "features.5.bias",
            "blocks.1.1.running_mean": "features.5.running_mean",
            "blocks.1.1.running_var": "features.5.running_var",
            "blocks.1.1.num_batches_tracked": "features.5.num_batches_tracked",
        }
        for target_key, source_key in aliases.items():
            if (
                target_key in target_state
                and source_key in source_state
                and target_state[target_key].shape == source_state[source_key].shape
            ):
                matches[target_key] = source_key

    if not matches:
        raise RuntimeError(
            "no shape-compatible query-encoder tensors matched the U-Net encoder; "
            "the selected backbone cannot be transferred safely"
        )

    updated_state = dict(target_state)
    for target_key, source_key in matches.items():
        target_value = target_state[target_key]
        updated_state[target_key] = source_state[source_key].detach().to(
            device=target_value.device, dtype=target_value.dtype
        )
    target_component.load_state_dict(updated_state, strict=True)

    matched_elements = sum(int(target_state[key].numel()) for key in matches)
    source_elements = sum(int(value.numel()) for value in source_state.values())
    target_elements = sum(int(value.numel()) for value in target_state.values())
    return {
        "mode": transfer_mode,
        "matched_tensors": len(matches),
        "source_tensors": len(source_state),
        "target_tensors": len(target_state),
        "matched_elements": matched_elements,
        "source_element_fraction": matched_elements / max(source_elements, 1),
        "target_element_fraction": matched_elements / max(target_elements, 1),
        "key_map": dict(sorted(matches.items())),
    }


def _train_segmentation_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int, int]:
    model.train()
    total_loss = 0.0
    examples = 0
    steps = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = soft_dice_loss(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite segmentation loss: {loss.detach().item()}")
        loss.backward()
        optimizer.step()
        count = int(images.shape[0])
        total_loss += float(loss.detach().item()) * count
        examples += count
        steps += 1
    if examples == 0:
        raise RuntimeError("segmentation training DataLoader produced no batches")
    return total_loss / examples, examples, steps


@torch.inference_mode()
def _evaluate_segmentation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    item_scores: list[float] = []
    examples = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("segmentation model produced non-finite logits")
        logits = logits.detach().cpu()
        masks = masks.detach().cpu()
        for index in range(int(images.shape[0])):
            item_scores.append(
                macro_dice(logits[index : index + 1], masks[index : index + 1])
            )
        examples += int(images.shape[0])
    if not item_scores:
        raise RuntimeError("segmentation evaluation DataLoader produced no batches")
    return float(np.mean(item_scores)), examples


def _denormalize_image(image: torch.Tensor) -> torch.Tensor:
    channels = int(image.shape[0])
    if channels == 3:
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    else:
        mean = image.new_full((channels, 1, 1), 0.5)
        std = image.new_full((channels, 1, 1), 0.5)
    return (image * std + mean).clamp(0, 1)


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, suffix=".npy", delete=False
    ) as temporary:
        np.save(temporary, value, allow_pickle=False)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


@torch.inference_mode()
def _save_brats_example(
    model: nn.Module,
    base_dataset: Dataset,
    test_index: int,
    image_size: int,
    angle: float,
    job_id: str,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Save the deterministic Figure 4 source and predictions as lossless arrays."""

    source_image, target = base_dataset[test_index]
    source_view = SegmentationImageView(
        Subset(base_dataset, [test_index]), image_size=image_size, angle=angle
    )
    original_image, rotated_image = source_view[0]
    images = torch.stack([original_image, rotated_image]).to(
        device, non_blocking=device.type == "cuda"
    )
    logits = model(images)
    if logits.shape[1] == 1:
        predictions = (torch.sigmoid(logits) >= 0.5).to(torch.uint8)[:, 0]
    else:
        predictions = logits.argmax(dim=1).to(torch.uint8)

    # The original source returned by BraTSSliceDataset is already normalized;
    # use the prepared view so all saved images have the configured dimensions.
    del source_image
    arrays: dict[str, np.ndarray] = {
        "original_image": _denormalize_image(original_image).cpu().numpy().astype(np.float32),
        "ground_truth_mask": target.cpu().numpy().astype(np.uint8),
        "predicted_mask": predictions[0].cpu().numpy(),
        "rotated_image": _denormalize_image(rotated_image).cpu().numpy().astype(np.float32),
        "rotated_ground_truth_mask": TF.rotate(
            target.unsqueeze(0).float() if target.ndim == 2 else target.float(),
            float(angle),
            interpolation=InterpolationMode.NEAREST,
            expand=False,
            fill=0,
        )
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8),
        "rotated_predicted_mask": predictions[1].cpu().numpy(),
    }
    example_root = output_root / "examples"
    paths: dict[str, str] = {}
    for name, array in arrays.items():
        path = example_root / f"{job_id}_brats_angle_95_{name}.npy"
        _atomic_save_npy(path, array)
        paths[name] = str(path)
    return {"source_index": int(test_index), "paths": paths}


def run_segmentation_job(config: ExperimentConfig, job: dict[str, Any]) -> dict[str, Any]:
    """Run one medical dataset/encoder/angle point and return one result row."""

    identity = _job_identity(job)
    section = config.section("segmentation")
    pretrain_settings = dict(section["pretrain"])
    supervised_settings = dict(section["supervised"])
    seed = identity["seed"]
    deterministic = bool(config.raw.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    device = _device(str(config.raw.get("device", "auto")))
    workers = int(config.raw.get("num_workers", 0))
    image_size = int(section["image_size"])
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    phase_start = time.perf_counter()
    base_dataset = build_segmentation_dataset(config, identity["dataset"])
    if len(base_dataset) == 0:
        raise ValueError(f"dataset {identity['dataset']!r} contains no items")
    in_channels, classes = _task_shape(base_dataset, identity["dataset"])
    train_indices, validation_indices, test_indices = _validate_partitions(
        len(base_dataset),
        float(supervised_settings["validation_fraction"]),
        float(supervised_settings["test_fraction"]),
        seed + 101,
    )
    timings["dataset_and_split_setup"] = _elapsed(phase_start, device)

    # Self-supervised fitting sees only the training source items.  Validation
    # and test masks are never used for model fitting or model selection.
    train_dataset = Subset(base_dataset, train_indices)
    validation_dataset = Subset(base_dataset, validation_indices)
    test_dataset = Subset(base_dataset, test_indices)
    contrastive_dataset = SegmentationImageView(
        train_dataset, image_size=image_size, angle=identity["angle"]
    )
    pretrain_batch_size = int(pretrain_settings["batch_size"])
    pretrain_loader = deterministic_loader(
        contrastive_dataset,
        batch_size=pretrain_batch_size,
        shuffle=True,
        seed=seed + 202,
        num_workers=workers,
        device=device,
        drop_last=len(contrastive_dataset) >= pretrain_batch_size,
    )
    contrastive_model = build_contrastive_model(
        "mocov2", identity["encoder"], in_channels, pretrain_settings
    ).to(device)
    phase_start = time.perf_counter()
    contrastive_trace = train_contrastive(
        contrastive_model, pretrain_loader, pretrain_settings, device
    )
    timings["contrastive_pretraining"] = _elapsed(phase_start, device)

    phase_start = time.perf_counter()
    unet = UNet(
        identity["encoder"],
        in_channels,
        classes,
        imagenet_initialized=bool(pretrain_settings.get("imagenet_initialized", True)),
    ).to(device)
    transfer = _copy_matching_state(query_encoder(contrastive_model), unet.encoder)
    timings["unet_construction_and_transfer"] = _elapsed(phase_start, device)

    supervised_batch_size = int(supervised_settings["batch_size"])
    train_loader = deterministic_loader(
        train_dataset,
        batch_size=supervised_batch_size,
        shuffle=True,
        seed=seed + 303,
        num_workers=workers,
        device=device,
    )
    validation_loader = deterministic_loader(
        validation_dataset,
        batch_size=supervised_batch_size,
        shuffle=False,
        seed=seed + 404,
        num_workers=workers,
        device=device,
    )
    test_loader = deterministic_loader(
        test_dataset,
        batch_size=supervised_batch_size,
        shuffle=False,
        seed=seed + 505,
        num_workers=workers,
        device=device,
    )

    epochs = int(supervised_settings["epochs"])
    patience = int(supervised_settings["early_stopping_patience"])
    if epochs <= 0:
        raise ValueError("supervised segmentation epochs must be positive")
    if patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    optimizer = torch.optim.Adam(
        unet.parameters(),
        lr=float(supervised_settings["learning_rate"]),
        weight_decay=float(supervised_settings.get("weight_decay", 0.0)),
    )
    best_validation_dice = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    train_loss_by_epoch: list[float] = []
    validation_dice_by_epoch: list[float] = []
    supervised_examples_seen = 0
    supervised_optimizer_steps = 0

    phase_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, seen, steps = _train_segmentation_epoch(
            unet, train_loader, optimizer, device
        )
        validation_dice, _ = _evaluate_segmentation(unet, validation_loader, device)
        train_loss_by_epoch.append(train_loss)
        validation_dice_by_epoch.append(validation_dice)
        supervised_examples_seen += seen
        supervised_optimizer_steps += steps
        if validation_dice > best_validation_dice + 1e-12:
            best_validation_dice = validation_dice
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in unet.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    timings["supervised_training"] = _elapsed(phase_start, device)
    if best_state is None:
        raise RuntimeError("early stopping did not capture a valid U-Net state")
    unet.load_state_dict(best_state, strict=True)

    phase_start = time.perf_counter()
    test_dice, test_examples = _evaluate_segmentation(unet, test_loader, device)
    timings["held_out_test_evaluation"] = _elapsed(phase_start, device)
    fingerprint = config_digest(config.raw)
    canonical_identity = {
        **identity,
        "config_sha256": fingerprint,
        "protocol_version": PROTOCOL_VERSION,
    }
    job_id = str(job.get("job_id") or stable_job_id(canonical_identity))
    example_artifacts: dict[str, Any] | None = None
    if identity["dataset"] == "brats" and abs(identity["angle"] - 95.0) < 1e-9:
        phase_start = time.perf_counter()
        example_artifacts = _save_brats_example(
            unet,
            base_dataset,
            test_indices[0],
            image_size,
            identity["angle"],
            job_id,
            config.output_root,
            device,
        )
        timings["qualitative_example_export"] = _elapsed(phase_start, device)
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
            "train_index_digest": _indices_digest(train_indices),
            "validation_index_digest": _indices_digest(validation_indices),
            "test_index_digest": _indices_digest(test_indices),
        }
    )
    result = {
        "job_id": job_id,
        "config_sha256": fingerprint,
        "protocol_version": PROTOCOL_VERSION,
        **identity,
        "metric_name": "segmentation_macro_dice",
        "metric_value": test_dice,
        "macro_dice": test_dice,
        "best_validation_macro_dice": best_validation_dice,
        "best_epoch": best_epoch,
        "loss": train_loss_by_epoch[-1],
        "pretrain_loss": contrastive_trace.final_loss,
        "losses": {
            "contrastive_final": contrastive_trace.final_loss,
            "contrastive_by_epoch": list(contrastive_trace.epoch_losses),
            "segmentation_train_final": train_loss_by_epoch[-1],
            "segmentation_train_by_epoch": train_loss_by_epoch,
            "validation_macro_dice_by_epoch": validation_dice_by_epoch,
        },
        "counts": {
            "source_items": len(base_dataset),
            "train_items": len(train_dataset),
            "validation_items": len(validation_dataset),
            "test_items": len(test_dataset),
            "test_examples": test_examples,
            "input_channels": in_channels,
            "output_classes": classes,
            "contrastive_examples_seen": contrastive_trace.examples_seen,
            "contrastive_optimizer_steps": contrastive_trace.optimizer_steps,
            "supervised_examples_seen": supervised_examples_seen,
            "supervised_optimizer_steps": supervised_optimizer_steps,
            "supervised_epochs_completed": len(train_loss_by_epoch),
        },
        "encoder_transfer": transfer,
        "timing_seconds": timings,
        "provenance": source_provenance,
        "settings": {
            "image_size": image_size,
            "num_workers": workers,
            "pretrain": pretrain_settings,
            "supervised": supervised_settings,
            "pretraining_partition": "train_only",
            "model_selection_metric": "validation_macro_dice",
            "reported_partition": "held_out_test",
            "dice_reduction": "mean_of_per_item_macro_class_dice_including_background",
        },
    }
    if example_artifacts is not None:
        result["example_artifacts"] = example_artifacts
    return result
