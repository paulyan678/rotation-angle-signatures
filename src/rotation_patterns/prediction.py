"""Appendix C signature-curve prediction on the published Figure 1 curves.

Appendix C forms logits from a *similarity* and then predicts with ``argmax``.  The paper
also calls one option "L2 distance" without specifying the required conversion.  Here L2
is therefore implemented as **negative Euclidean distance**, so larger values consistently
mean more similar.  This explicit assumption is also recorded in ``summary.json``.

Equation (6) uses ``a_i`` as the class-specific component of one shared vector ``a``.  We
fit that vector on a balanced batch containing every target class.  Fitting a different
vector after being told the unknown target ``r`` would leak the evaluation label and make
the classification task degenerate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .reference import ReferenceCurves, load_reference
from .reproducibility import PROTOCOL_VERSION, config_digest


# Red numeric labels printed in the published Figure 2.  The paper does not release the
# trial data behind these values; retaining them here makes the numerical mismatch of the
# Appendix C reconstruction machine-readable instead of silently presenting it as a match.
PUBLISHED_FIGURE2_MEANS: dict[str, dict[str, tuple[float, ...]]] = {
    "row": {
        "cosine": (0.62, 0.51, 0.66, 0.65, 0.44, 0.51, 0.59, 0.57,
                   0.52, 0.53, 0.65, 0.57, 0.51, 0.53, 0.57, 0.59),
        "l2": (0.80, 0.77, 0.70, 0.34, 0.50, 0.68, 0.69, 0.50,
               0.49, 0.54, 0.48, 0.82, 0.47, 0.59, 0.75, 0.63),
    },
    "column": {
        "cosine": (0.59, 0.62, 0.66, 0.66, 0.73, 0.78, 0.49, 0.65,
                   0.55, 0.72, 0.54, 0.79, 0.69, 0.48, 0.68, 0.61),
        "l2": (0.75, 0.43, 0.66, 0.45, 0.67, 0.70, 0.53, 0.77,
               0.74, 0.65, 0.51, 0.67, 0.47, 0.46, 0.53, 0.63),
    },
}


def _pairwise_similarities(curves: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the two Appendix C similarities once for all 256 curve pairs."""

    flat = np.ascontiguousarray(curves.reshape(-1, curves.shape[-1]), dtype=np.float64)
    # Some BLAS builds emit spurious overflow warnings during a finite blocked matmul;
    # the explicit finiteness check below remains authoritative.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gram = flat @ flat.T
    squared_norm = np.einsum("ij,ij->i", flat, flat)
    norms = np.sqrt(squared_norm)
    if np.any(norms == 0.0):
        raise ValueError("cosine similarity is undefined for an all-zero reference curve")
    cosine = gram / np.outer(norms, norms)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * gram
    negative_l2 = -np.sqrt(np.maximum(squared_distance, 0.0))
    if not np.isfinite(cosine).all() or not np.isfinite(negative_l2).all():
        raise FloatingPointError("non-finite pairwise curve similarity")
    return {"cosine": cosine, "l2": negative_l2}


def _flat_index(axis: str, class_index: int, sample_index: int, size: int) -> int:
    if axis == "row":
        return class_index * size + sample_index
    if axis == "column":
        return sample_index * size + class_index
    raise ValueError(f"unknown prediction axis: {axis}")


def _feature_cube(
    similarity: np.ndarray,
    axis: str,
    sample_indices: Iterable[int],
    size: int,
) -> np.ndarray:
    """Return Appendix C mean similarities for every target/query/candidate triple."""

    indices = tuple(int(value) for value in sample_indices)
    if len(indices) < 2:
        raise ValueError("Appendix C leave-one-curve-out similarity needs at least two curves")
    features = np.empty((size, len(indices), size), dtype=np.float64)
    for target in range(size):
        for query_offset, query_sample in enumerate(indices):
            query = _flat_index(axis, target, query_sample, size)
            for candidate in range(size):
                candidates = [
                    _flat_index(axis, candidate, sample, size)
                    for sample in indices
                    if candidate != target or sample != query_sample
                ]
                # For the true class, Equation (4) excludes the query curve itself.
                # For all other classes it averages over the entire opposite-axis split.
                features[target, query_offset, candidate] = float(
                    np.mean(similarity[query, candidates])
                )
    return features


def _adam_weights(
    train_features: np.ndarray,
    *,
    epochs: int,
    sample_size: int,
    learning_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float]]:
    """Fit the single per-class weight vector in Equations (4) and (6)."""

    classes, available, candidate_classes = train_features.shape
    if candidate_classes != classes:
        raise ValueError("prediction feature cube must have the same number of target classes")
    if epochs <= 0 or sample_size <= 0 or learning_rate <= 0:
        raise ValueError("prediction epochs, sample_size, and learning_rate must be positive")

    weights = np.ones(classes, dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    losses: list[float] = []

    for step in range(1, epochs + 1):
        # Appendix C samples with replacement in every epoch.  Each target class supplies
        # the stated mini-batch size, yielding a balanced optimization batch.
        sampled = rng.integers(0, available, size=(classes, sample_size))
        batch = np.concatenate(
            [train_features[target, sampled[target]] for target in range(classes)], axis=0
        )
        labels = np.repeat(np.arange(classes), sample_size)
        logits = batch * weights[None, :]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponential = np.exp(shifted)
        probabilities = exponential / np.sum(exponential, axis=1, keepdims=True)
        loss = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)).mean()
        losses.append(float(loss))

        derivative = probabilities
        derivative[np.arange(len(labels)), labels] -= 1.0
        gradient = np.mean(derivative * batch, axis=0)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_second = second_moment / (1.0 - beta2**step)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

    if not np.isfinite(weights).all():
        raise FloatingPointError("Adam produced non-finite Appendix C weights")
    return weights, losses


def _evaluate_trials(
    test_features: np.ndarray,
    weights: np.ndarray,
    *,
    draws_per_repetition: int,
    repetitions: int,
    test_indices: tuple[int, ...],
    rng: np.random.Generator,
    axis: str,
    metric: str,
    names: tuple[str, ...],
) -> list[dict[str, Any]]:
    if draws_per_repetition <= 0 or repetitions <= 0:
        raise ValueError("prediction draws and repetitions must be positive")
    rows: list[dict[str, Any]] = []
    for target, target_name in enumerate(names):
        for repetition in range(repetitions):
            sampled_offsets = rng.integers(
                0, len(test_indices), size=draws_per_repetition
            )
            predictions = np.argmax(
                test_features[target, sampled_offsets] * weights[None, :], axis=1
            )
            correct_count = int(np.sum(predictions == target))
            rows.append(
                {
                    "axis": axis,
                    "metric": metric,
                    "target_index": target,
                    "target_name": target_name,
                    "trial": repetition,
                    "repetition": repetition,
                    "draws": draws_per_repetition,
                    "correct_count": correct_count,
                    "accuracy": correct_count / draws_per_repetition,
                }
            )
    return rows


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = handle.name
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def run_prediction_experiment(config: ExperimentConfig) -> tuple[Path, Path]:
    """Run row and column prediction and save trial-level CSV plus summary JSON."""

    reference: ReferenceCurves = load_reference(config)
    if reference.tensor.shape[0] != reference.tensor.shape[1]:
        raise ValueError("Appendix C implementation requires equal row and column counts")
    size = reference.tensor.shape[0]
    settings = dict(config.section("prediction"))
    train_size = int(settings.get("train_size", 10))
    sample_size = int(settings.get("sample_size", 20))
    epochs = int(settings.get("epochs", 200))
    learning_rate = float(settings.get("learning_rate", 0.01))
    draws_per_repetition = int(settings.get("trials", 100))
    repetitions = int(settings.get("evaluation_repetitions", settings.get("repetitions", 100)))
    metrics = tuple(str(metric).lower() for metric in settings.get("metrics", ("cosine", "l2")))
    unknown = set(metrics) - {"cosine", "l2"}
    if unknown:
        raise ValueError(f"unsupported prediction metrics: {sorted(unknown)}")
    if not 1 < train_size < size - 1:
        raise ValueError(f"prediction train_size must leave at least two test curves, got {train_size}")
    train_indices = tuple(range(train_size))
    test_indices = tuple(range(train_size, size))

    pairwise = _pairwise_similarities(reference.tensor)
    all_trials: list[dict[str, Any]] = []
    weights_summary: dict[str, dict[str, list[float]]] = {}
    loss_summary: dict[str, dict[str, dict[str, float]]] = {}
    seed_sequence = np.random.SeedSequence(config.seed)
    child_seeds = iter(seed_sequence.spawn(2 * len(metrics)))

    for axis, names in (("row", reference.datasets), ("column", reference.encoder_methods)):
        weights_summary[axis] = {}
        loss_summary[axis] = {}
        for metric in metrics:
            rng = np.random.default_rng(next(child_seeds))
            train_features = _feature_cube(pairwise[metric], axis, train_indices, size)
            test_features = _feature_cube(pairwise[metric], axis, test_indices, size)
            weights, losses = _adam_weights(
                train_features,
                epochs=epochs,
                sample_size=sample_size,
                learning_rate=learning_rate,
                rng=rng,
            )
            weights_summary[axis][metric] = weights.tolist()
            loss_summary[axis][metric] = {
                "initial": losses[0],
                "final": losses[-1],
            }
            all_trials.extend(
                _evaluate_trials(
                    test_features,
                    weights,
                    draws_per_repetition=draws_per_repetition,
                    repetitions=repetitions,
                    test_indices=test_indices,
                    rng=rng,
                    axis=axis,
                    metric=metric,
                    names=names,
                )
            )

    frame = pd.DataFrame(all_trials).sort_values(
        ["axis", "metric", "target_index", "repetition"], ignore_index=True
    )
    grouped = (
        frame.groupby(["axis", "metric", "target_index", "target_name"], sort=True)["accuracy"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
    overall = (
        frame.groupby(["axis", "metric"], sort=True)["accuracy"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
    published_comparison: list[dict[str, Any]] = []
    for axis in ("row", "column"):
        for metric in metrics:
            published = PUBLISHED_FIGURE2_MEANS[axis][metric]
            reconstructed = grouped[
                (grouped["axis"] == axis) & (grouped["metric"] == metric)
            ].sort_values("target_index")["accuracy"].to_numpy(dtype=float)
            published_comparison.append(
                {
                    "axis": axis,
                    "metric": metric,
                    "published_macro_mean": float(np.mean(published)),
                    "reconstructed_macro_mean": float(np.mean(reconstructed)),
                    "mean_absolute_class_difference": float(
                        np.mean(np.abs(reconstructed - np.asarray(published)))
                    ),
                }
            )
    summary: dict[str, Any] = {
        "protocol": "Appendix C held-out-axis weighted similarity classifier",
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": config_digest(config.raw),
        "seed": config.seed,
        "reference_commit": "bf9d88733752448c193b7b43356a7e083b021a7b",
        "tensor_shape": list(reference.tensor.shape),
        "train_indices_zero_based": list(train_indices),
        "test_indices_zero_based": list(test_indices),
        "epochs": epochs,
        "sample_size": sample_size,
        "draws_per_evaluation_repetition": draws_per_repetition,
        "evaluation_repetitions_per_class": repetitions,
        "learning_rate": learning_rate,
        "l2_assumption": (
            "L2 is negative Euclidean distance because Equation (6) uses argmax; "
            "larger values must mean greater similarity."
        ),
        "weight_interpretation": (
            "One shared class-weight vector is fit on balanced examples from all classes. "
            "Equation (6) denotes its i-th component a_i; target-specific vectors would "
            "condition on and leak the label being predicted."
        ),
        "published_result_boundary": (
            "The paper does not release Figure 2 trial data. The reconstructed accuracies "
            "do not reproduce the red means printed in Figure 2; those digitized labels "
            "are retained below for an explicit comparison."
        ),
        "published_figure2_red_means": PUBLISHED_FIGURE2_MEANS,
        "published_comparison": published_comparison,
        "weights": weights_summary,
        "loss": loss_summary,
        "overall_accuracy": overall.to_dict(orient="records"),
        "per_class_accuracy": grouped.to_dict(orient="records"),
    }
    output = config.output_root / "prediction"
    trials_path = output / "trials.csv"
    summary_path = output / "summary.json"
    _atomic_csv(frame, trials_path)
    _atomic_json(summary, summary_path)
    return trials_path, summary_path
