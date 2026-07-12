"""Deterministic training utilities shared by the experiment entry points.

The helpers in this module deliberately keep policy decisions (dataset splits,
pair construction, and result schemas) out of the training loops.  This makes
the relatively expensive pre-training and probe stages independently testable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainingTrace:
    """JSON-friendly summary of an optimization run."""

    epoch_losses: tuple[float, ...]
    examples_seen: int
    optimizer_steps: int

    @property
    def final_loss(self) -> float:
        if not self.epoch_losses:
            raise RuntimeError("training trace does not contain an epoch loss")
        return self.epoch_losses[-1]


@dataclass(frozen=True)
class ProbeEvaluation:
    """Aggregate binary-classification metrics."""

    accuracy: float
    loss: float
    correct: int
    examples: int


def _seed_worker(_: int) -> None:
    """Seed libraries used by dataset workers from PyTorch's worker seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def deterministic_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader whose ordering and worker RNGs are reproducible."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
        persistent_workers=False,
    )


def _move(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, non_blocking=device.type == "cuda")


def train_contrastive(
    model: nn.Module,
    loader: DataLoader,
    settings: dict[str, Any],
    device: torch.device,
) -> TrainingTrace:
    """Optimize a SimCLR or MoCo model through its common ``loss`` API."""

    epochs = int(settings["epochs"])
    if epochs <= 0:
        raise ValueError("contrastive epochs must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("contrastive model has no trainable parameters")
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(settings["learning_rate"]),
        momentum=float(settings.get("momentum", 0.0)),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    loss_method = getattr(model, "loss", None)
    if not callable(loss_method):
        raise TypeError("contrastive model must expose a callable loss(first, second) method")

    model.train()
    epoch_losses: list[float] = []
    examples_seen = 0
    optimizer_steps = 0
    for _ in range(epochs):
        running_loss = 0.0
        running_examples = 0
        for first, second in loader:
            first = _move(first, device)
            second = _move(second, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_method(first, second)
            if loss.ndim != 0:
                loss = loss.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite contrastive loss: {loss.detach().item()}")
            loss.backward()
            optimizer.step()

            count = int(first.shape[0])
            running_loss += float(loss.detach().item()) * count
            running_examples += count
            examples_seen += count
            optimizer_steps += 1
        if running_examples == 0:
            raise RuntimeError("contrastive DataLoader produced no batches")
        epoch_losses.append(running_loss / running_examples)
    return TrainingTrace(tuple(epoch_losses), examples_seen, optimizer_steps)


@torch.inference_mode()
def extract_pair_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze an encoder and extract original/rotated features in source order."""

    encoder.eval()
    encoder.requires_grad_(False)
    expected = len(loader.dataset)
    originals: torch.Tensor | None = None
    rotated: torch.Tensor | None = None
    offset = 0
    for first, second in loader:
        batch_size = int(first.shape[0])
        images = torch.cat([first, second], dim=0)
        features = encoder(_move(images, device))
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        if features.ndim != 2 or features.shape[0] != 2 * batch_size:
            raise ValueError(
                "encoder must return one flat feature vector per image; "
                f"received shape {tuple(features.shape)}"
            )
        original_batch, rotated_batch = features.split(batch_size, dim=0)
        original_batch = original_batch.detach().to(device="cpu", dtype=torch.float32)
        rotated_batch = rotated_batch.detach().to(device="cpu", dtype=torch.float32)
        if originals is None:
            feature_dim = int(original_batch.shape[1])
            originals = torch.empty((expected, feature_dim), dtype=torch.float32)
            rotated = torch.empty_like(originals)
        end = offset + batch_size
        if end > expected:
            raise RuntimeError("feature DataLoader yielded more examples than its dataset length")
        originals[offset:end].copy_(original_batch)
        assert rotated is not None
        rotated[offset:end].copy_(rotated_batch)
        offset = end
    if originals is None or rotated is None:
        raise RuntimeError("feature DataLoader produced no batches")
    if offset != expected:
        raise RuntimeError(
            f"feature DataLoader yielded {offset} examples for a dataset of length {expected}"
        )
    return originals, rotated


def single_cycle_derangement(length: int, *, seed: int) -> torch.Tensor:
    """Return a seeded permutation with no fixed points.

    Mapping a random ordering to its one-position rotation produces a single
    cycle.  Consequently every source occurs exactly once as a negative target
    and no source can be paired with itself.
    """

    if length < 2:
        raise ValueError("a derangement requires at least two elements")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(length, generator=generator)
    permutation = torch.empty(length, dtype=torch.long)
    permutation[order] = order.roll(shifts=-1)
    if bool(torch.any(permutation == torch.arange(length))):
        raise AssertionError("internal error: generated permutation is not a derangement")
    return permutation


class BalancedFeaturePairDataset(Dataset):
    """Lazy, balanced positive/negative pairs from one source partition."""

    def __init__(
        self,
        original_features: torch.Tensor,
        rotated_features: torch.Tensor,
        source_indices: torch.Tensor,
        *,
        seed: int,
    ):
        if original_features.ndim != 2 or rotated_features.ndim != 2:
            raise ValueError("feature tensors must be two-dimensional")
        if original_features.shape != rotated_features.shape:
            raise ValueError("original and rotated feature tensors must have identical shapes")
        indices = torch.as_tensor(source_indices, dtype=torch.long, device="cpu").flatten()
        if indices.numel() < 2:
            raise ValueError("each source split needs at least two images")
        if int(indices.min()) < 0 or int(indices.max()) >= original_features.shape[0]:
            raise IndexError("source index is outside the extracted feature tensors")
        if int(torch.unique(indices).numel()) != int(indices.numel()):
            raise ValueError("source_indices must not contain duplicates")

        self.original_features = original_features
        self.rotated_features = rotated_features
        self.source_indices = indices
        self.negative_targets = single_cycle_derangement(len(indices), seed=seed)

    def __len__(self) -> int:
        return 2 * len(self.source_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_count = len(self.source_indices)
        if index < 0:
            index += 2 * source_count
        if index < 0 or index >= 2 * source_count:
            raise IndexError(index)
        local_anchor = index % source_count
        anchor = int(self.source_indices[local_anchor])
        if index < source_count:
            target = self.rotated_features[anchor]
            label = 1.0
        else:
            local_target = int(self.negative_targets[local_anchor])
            target = self.original_features[int(self.source_indices[local_target])]
            label = 0.0
        pair = torch.cat([self.original_features[anchor], target], dim=0)
        return pair, torch.tensor(label, dtype=torch.float32)


def build_balanced_pair_dataset(
    original_features: torch.Tensor,
    rotated_features: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    seed: int,
) -> BalancedFeaturePairDataset:
    """Construct equal positive and negative feature pairs within one split.

    Positives are ``(f(x), f(rotate(x)))``.  Negatives are ``(f(x), f(x'))``
    with ``x != x'`` guaranteed by a derangement.  ``source_indices`` should
    already belong to a single train or test partition; pairing after the split
    prevents source-image leakage between partitions.
    """

    return BalancedFeaturePairDataset(
        original_features,
        rotated_features,
        source_indices,
        seed=seed,
    )


def train_binary_probe(
    probe: nn.Module,
    loader: DataLoader,
    settings: dict[str, Any],
    device: torch.device,
) -> TrainingTrace:
    """Train the downstream six-layer probe with BCE and configured SGD."""

    epochs = int(settings["epochs"])
    if epochs <= 0:
        raise ValueError("probe epochs must be positive")
    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=float(settings["learning_rate"]),
        momentum=float(settings.get("momentum", 0.0)),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    criterion = nn.BCEWithLogitsLoss()
    probe.train()
    epoch_losses: list[float] = []
    examples_seen = 0
    optimizer_steps = 0
    for _ in range(epochs):
        running_loss = 0.0
        running_examples = 0
        for features, labels in loader:
            features = _move(features, device)
            labels = _move(labels, device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(features)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite probe loss: {loss.detach().item()}")
            loss.backward()
            optimizer.step()

            count = int(labels.numel())
            running_loss += float(loss.detach().item()) * count
            running_examples += count
            examples_seen += count
            optimizer_steps += 1
        if running_examples == 0:
            raise RuntimeError("probe DataLoader produced no batches")
        epoch_losses.append(running_loss / running_examples)
    return TrainingTrace(tuple(epoch_losses), examples_seen, optimizer_steps)


@torch.inference_mode()
def evaluate_binary_probe(
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ProbeEvaluation:
    """Evaluate BCE loss and thresholded sigmoid accuracy."""

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    probe.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    for features, labels in loader:
        features = _move(features, device)
        labels = _move(labels, device)
        logits = probe(features)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("probe produced non-finite logits")
        total_loss += float(criterion(logits, labels).item())
        predictions = logits >= 0.0
        correct += int((predictions == labels.bool()).sum().item())
        examples += int(labels.numel())
    if examples == 0:
        raise RuntimeError("evaluation DataLoader produced no batches")
    return ProbeEvaluation(correct / examples, total_loss / examples, correct, examples)
