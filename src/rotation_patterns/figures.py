"""Generate thesis Figures 1--4 from validated research artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .reference import load_reference


def _save(figure: plt.Figure, output_root: Path, stem: str) -> tuple[Path, Path]:
    destination = output_root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / f"{stem}.png"
    pdf = destination / f"{stem}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _classification_aggregate(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]] | None:
    path = config.output_root / "aggregate" / "classification.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"dataset", "method", "encoder", "angle", "metric_value"}
    if not required <= set(frame.columns):
        warnings.warn(f"ignoring malformed classification aggregate {path}", stacklevel=2)
        return None
    datasets = tuple(str(value) for value in config.raw["datasets"])
    methods = tuple(str(value) for value in config.raw["methods"])
    encoders = tuple(str(value) for value in config.raw["encoders"])
    angles = np.asarray(config.angles("classification"), dtype=np.float64)
    columns = tuple(f"{method} {encoder}" for method in methods for encoder in encoders)
    tensor = np.empty((len(datasets), len(columns), len(angles)), dtype=np.float64)
    for row, dataset in enumerate(datasets):
        for method_index, method in enumerate(methods):
            for encoder_index, encoder in enumerate(encoders):
                column = method_index * len(encoders) + encoder_index
                subset = frame[
                    (frame["dataset"] == dataset)
                    & (frame["method"] == method)
                    & (frame["encoder"] == encoder)
                ]
                values = subset.groupby("angle", as_index=True)["metric_value"].mean()
                try:
                    tensor[row, column] = values.loc[angles].to_numpy(dtype=np.float64)
                except KeyError:
                    warnings.warn(
                        f"classification aggregate is incomplete for {dataset}/{method}/{encoder}; "
                        "falling back to reference curves",
                        stacklevel=2,
                    )
                    return None
    return angles, tensor, datasets, columns


def create_figure1(config: ExperimentConfig) -> tuple[Path, Path]:
    """Render the accuracy-versus-angle grid from complete new output or reference data."""

    aggregate = _classification_aggregate(config)
    if aggregate is None:
        reference = load_reference(config)
        angles, tensor = reference.angles, reference.tensor
        datasets, columns = reference.datasets, reference.encoder_methods
    else:
        angles, tensor, datasets, columns = aggregate

    rows, cols = tensor.shape[:2]
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(18.0, cols * 1.55), max(14.0, rows * 1.25)),
        sharex=True,
        squeeze=False,
    )
    for row in range(rows):
        for column in range(cols):
            axis = axes[row, column]
            axis.plot(angles, tensor[row, column], color="#e1812c", linewidth=0.65)
            axis.set_xlim(float(angles[0]), float(angles[-1]))
            axis.tick_params(axis="both", labelsize=5, length=2, pad=1)
            if row == rows - 1:
                axis.set_xticks([0, 100, 200, 300])
            if column == 0:
                axis.set_ylabel(datasets[row], fontsize=7, rotation=0, ha="right", va="center")
            if row == 0:
                axis.set_title(columns[column], fontsize=7, rotation=35, ha="left", pad=8)
    figure.supxlabel("Rotation angle (degrees)", fontsize=10)
    figure.supylabel("Dataset / downstream accuracy", fontsize=10, x=0.02)
    figure.suptitle("Figure 1 - Rotation-angle response signatures", fontsize=14, y=1.005)
    figure.subplots_adjust(wspace=0.24, hspace=0.22)
    return _save(figure, config.output_root, "figure1")


def create_figure2(config: ExperimentConfig) -> tuple[Path, Path] | None:
    """Render the 2x2 trial-level violin grid from Appendix C output."""

    path = config.output_root / "prediction" / "trials.csv"
    if not path.exists():
        warnings.warn(f"Figure 2 skipped: no prediction trials at {path}", stacklevel=2)
        return None
    frame = pd.read_csv(path)
    required = {"axis", "metric", "target_index", "accuracy"}
    if not required <= set(frame.columns):
        raise ValueError(f"prediction trials are missing columns: {sorted(required-set(frame))}")

    figure, axes = plt.subplots(2, 2, figsize=(18, 9), sharey=True, squeeze=False)
    for row, prediction_axis in enumerate(("row", "column")):
        for column, metric in enumerate(("cosine", "l2")):
            axis = axes[row, column]
            subset = frame[(frame["axis"] == prediction_axis) & (frame["metric"] == metric)]
            targets = sorted(int(value) for value in subset["target_index"].unique())
            for position, target in enumerate(targets, start=1):
                values = subset.loc[subset["target_index"] == target, "accuracy"].to_numpy(float)
                mean = float(np.mean(values))
                if np.unique(values).size > 1:
                    parts = axis.violinplot(
                        values, positions=[position], widths=0.8, showextrema=False
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor("#4c78a8")
                        body.set_edgecolor("#2f4b66")
                        body.set_alpha(0.7)
                else:
                    axis.plot(
                        [position - 0.25, position + 0.25],
                        [values[0], values[0]],
                        color="#4c78a8",
                        linewidth=3,
                    )
                axis.text(position, min(0.98, mean + 0.045), f"{mean:.2f}", color="red", ha="center", fontsize=7)
            prefix = "Row" if prediction_axis == "row" else "Col"
            axis.set_xticks(range(1, len(targets) + 1), [f"{prefix} {x}" for x in targets])
            axis.tick_params(axis="x", labelrotation=45, labelsize=7)
            axis.set_ylim(-0.05, 1.05)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(
                f"{prediction_axis.title()} prediction - "
                f"{'Cosine similarity' if metric == 'cosine' else 'Negative L2 similarity'}"
            )
            if column == 0:
                axis.set_ylabel("Trial accuracy")
    figure.suptitle("Figure 2 - Signature prediction on the published curves", fontsize=14)
    figure.text(
        0.5,
        0.002,
        "Configured Appendix C interpretation; published means are retained in summary.json.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout()
    return _save(figure, config.output_root, "figure2")


def create_figure3(config: ExperimentConfig) -> tuple[Path, Path] | None:
    """Render the three medical Dice columns and the HoG/SVM comparison column."""

    aggregate = config.output_root / "aggregate"
    segmentation_path, hog_path = aggregate / "segmentation.csv", aggregate / "hog.csv"
    if not segmentation_path.exists() or not hog_path.exists():
        warnings.warn(
            "Figure 3 skipped: both aggregate/segmentation.csv and aggregate/hog.csv are required",
            stacklevel=2,
        )
        return None
    segmentation, hog = pd.read_csv(segmentation_path), pd.read_csv(hog_path)
    required = {"dataset", "encoder", "angle", "metric_value"}
    if not required <= set(segmentation) or not {"dataset", "angle", "metric_value"} <= set(hog):
        raise ValueError("Figure 3 aggregate CSV schema is incomplete")
    settings = config.section("segmentation")
    datasets = tuple(str(value) for value in settings["datasets"])
    encoders = tuple(str(value) for value in settings["encoders"])
    segmentation_expected = {
        (dataset, encoder, float(angle))
        for dataset in datasets
        for encoder in encoders
        for angle in config.angles("segmentation")
    }
    segmentation_actual = set(
        segmentation[["dataset", "encoder", "angle"]].itertuples(index=False, name=None)
    )
    hog_expected = {
        (str(dataset), float(angle))
        for dataset in config.section("hog")["datasets"]
        for angle in config.angles("hog")
    }
    hog_actual = set(hog[["dataset", "angle"]].itertuples(index=False, name=None))
    if segmentation_actual != segmentation_expected or hog_actual != hog_expected:
        raise ValueError(
            "Figure 3 requires complete configured aggregates: "
            f"segmentation missing={len(segmentation_expected - segmentation_actual)}, "
            f"extra={len(segmentation_actual - segmentation_expected)}; "
            f"hog missing={len(hog_expected - hog_actual)}, "
            f"extra={len(hog_actual - hog_expected)}"
        )
    figure, axes = plt.subplots(len(datasets), 4, figsize=(16, 3.5 * len(datasets)), squeeze=False)
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for row, dataset in enumerate(datasets):
        for column, encoder in enumerate(encoders[:3]):
            axis = axes[row, column]
            values = segmentation[
                (segmentation["dataset"] == dataset) & (segmentation["encoder"] == encoder)
            ].sort_values("angle")
            axis.plot(values["angle"], values["metric_value"], color=colors[column], linewidth=1.2)
            axis.set_title(f"MoCo {encoder}" if row == 0 else "")
            axis.set_ylabel(f"{dataset}\nMacro Dice" if column == 0 else "Macro Dice")
            axis.grid(alpha=0.2)
        svm = hog[hog["dataset"] == dataset].sort_values("angle")
        axis = axes[row, 3]
        axis.plot(svm["angle"], svm["metric_value"], color="#b279a2", linewidth=1.2)
        axis.set_title("HoG / RBF-SVM" if row == 0 else "")
        axis.set_ylabel("Accuracy (SVM)")
        axis.grid(alpha=0.2)
        for column in range(4):
            axes[row, column].set_xlim(0, 360)
            axes[row, column].set_xticks(range(0, 361, 50))
            if row == len(datasets) - 1:
                axes[row, column].set_xlabel("Rotation angle (degrees)")
    figure.suptitle(
        "Figure 3 - Held-out U-Net Dice and HoG shortcut test", fontsize=14
    )
    figure.tight_layout()
    return _save(figure, config.output_root, "figure3")


@dataclass
class _Example:
    image: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    angle: float | None
    path: Path


def _first_key(data: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _load_examples(root: Path) -> list[_Example]:
    examples: list[_Example] = []
    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                image = _first_key(data, ("image", "input", "source", "original"))
                target = _first_key(data, ("target", "mask", "ground_truth", "true_mask"))
                prediction = _first_key(data, ("prediction", "pred", "predicted_mask"))
                if image is None or target is None or prediction is None:
                    continue
                angle_value = _first_key(data, ("angle", "rotation_angle"))
                angle = float(np.ravel(angle_value)[0]) if angle_value is not None else None
        except (OSError, ValueError):
            continue
        if angle is None:
            match = re.search(r"(?:angle|rotation)[_-]?(\d+(?:\.\d+)?)", path.stem)
            angle = float(match.group(1)) if match else None
        examples.append(_Example(image, target, prediction, angle, path))

    # The segmentation runner saves Figure 4 as six separately atomic arrays.  Group
    # bundles by the prefix preceding ``_original_image.npy`` so partially written or
    # unrelated arrays are never mistaken for a complete qualitative example.
    for original_path in sorted((root / "examples").glob("*_original_image.npy")):
        prefix = original_path.name[: -len("original_image.npy")]
        paths = {
            name: original_path.parent / f"{prefix}{name}.npy"
            for name in (
                "original_image",
                "ground_truth_mask",
                "predicted_mask",
                "rotated_image",
                "rotated_ground_truth_mask",
                "rotated_predicted_mask",
            )
        }
        if not all(path.exists() for path in paths.values()):
            continue
        try:
            arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
        except (OSError, ValueError):
            continue
        match = re.search(r"angle[_-]?(\d+(?:\.\d+)?)", original_path.stem)
        angle = float(match.group(1)) if match else 95.0
        examples.extend(
            (
                _Example(
                    arrays["original_image"],
                    arrays["ground_truth_mask"],
                    arrays["predicted_mask"],
                    0.0,
                    original_path,
                ),
                _Example(
                    arrays["rotated_image"],
                    arrays["rotated_ground_truth_mask"],
                    arrays["rotated_predicted_mask"],
                    angle,
                    paths["rotated_image"],
                ),
            )
        )
    return examples


def _display_image(value: np.ndarray, *, mask: bool = False) -> np.ndarray:
    array = np.squeeze(np.asarray(value))
    if not mask and array.ndim == 3 and array.shape[0] == 4 and array.shape[-1] > 4:
        # BraTS inputs are (FLAIR, T1, T1ce, T2).  Show FLAIR as a defined grayscale
        # modality instead of falsely coloring three unrelated MRI modalities as RGB.
        array = array[0]
    elif not mask and array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[0] <= 3 and array.shape[-1] > 4:
        array = np.moveaxis(array, 0, -1)
    if mask and array.ndim == 3:
        class_axis = 0 if array.shape[0] <= 4 and array.shape[-1] > 4 else -1
        array = np.argmax(array, axis=class_axis)
    if not mask and array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    return array


def create_figure4(config: ExperimentConfig) -> tuple[Path, Path] | None:
    """Render saved qualitative segmentation examples, or explicitly skip if absent."""

    examples = _load_examples(config.output_root)
    if not examples:
        warnings.warn(
            "Figure 4 skipped: no NPZ with image, target/mask, and prediction arrays was saved",
            stacklevel=2,
        )
        return None
    selected: list[_Example] | None = None
    for original in examples:
        if original.angle != 0.0:
            continue
        original_bundle = re.sub(r"_(?:original|rotated)_image\.npy$", "", original.path.name)
        for rotated in examples:
            rotated_bundle = re.sub(r"_(?:original|rotated)_image\.npy$", "", rotated.path.name)
            if (
                rotated.path.parent == original.path.parent
                and rotated_bundle == original_bundle
                and rotated.angle not in (None, 0.0)
            ):
                selected = [original, rotated]
                break
        if selected is not None:
            break
    if selected is None:
        ordered = sorted(examples, key=lambda item: (item.angle is None, item.angle or 0.0))
        selected = ordered[:2]
    figure, axes = plt.subplots(len(selected), 3, figsize=(10, 3.5 * len(selected)), squeeze=False)
    for row, example in enumerate(selected):
        panels = (example.image, example.target, example.prediction)
        titles = ("Original image", "True segmentation", "Predicted segmentation")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            array = _display_image(panel, mask=column > 0)
            axes[row, column].imshow(array, cmap="gray" if array.ndim == 2 else None)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)
        angle_label = "unknown angle" if example.angle is None else f"{example.angle:g} degrees"
        axes[row, 0].set_ylabel(angle_label)
    figure.suptitle("Figure 4 - Qualitative segmentation examples", fontsize=14)
    figure.tight_layout()
    return _save(figure, config.output_root, "figure4")


def create_all_figures(config: ExperimentConfig, only: str = "all") -> list[Path]:
    """Create one requested paper figure or every figure with available inputs."""

    creators = {
        "figure1": create_figure1,
        "figure2": create_figure2,
        "figure3": create_figure3,
        "figure4": create_figure4,
    }
    if only != "all" and only not in creators:
        raise ValueError(f"unknown figure selection: {only!r}")
    requested = tuple(creators) if only == "all" else (only,)
    outputs: list[Path] = []
    for name in requested:
        result = creators[name](config)
        if result is not None:
            outputs.extend(result)
    return outputs


# Backwards-friendly aliases for callers that prefer underscored figure numbers.
create_figure_1 = create_figure1
create_figure_2 = create_figure2
create_figure_3 = create_figure3
create_figure_4 = create_figure4
