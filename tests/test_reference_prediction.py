import numpy as np
import pytest

from rotation_patterns.config import ExperimentConfig, load_config
from rotation_patterns.prediction import _feature_cube, _pairwise_similarities
from rotation_patterns.reference import (
    _read_curve,
    fetch_reference,
    load_bundled_reference,
    load_reference,
)


def test_reference_curve_parser_checks_paper_grid(tmp_path) -> None:
    path = tmp_path / "plot_1_1.csv"
    angles = np.arange(3600, dtype=np.float64) / 10
    values = np.column_stack([angles, np.full(3600, 0.75)])
    np.savetxt(path, values, delimiter=",", header="RotationAngle,Accuracy", comments="")
    parsed_angles, accuracy = _read_curve(path)
    assert np.array_equal(parsed_angles, angles)
    assert np.all(accuracy == 0.75)


def test_bundled_reference_is_complete_and_stages_without_network(tmp_path) -> None:
    bundled = load_bundled_reference()
    assert bundled.tensor.shape == (16, 16, 3600)
    base = load_config("configs/smoke.yaml")
    config = ExperimentConfig(base.path, {**base.raw, "output_root": str(tmp_path)})
    destination = fetch_reference(config)
    assert (destination / "figure1_curves.npz").is_file()
    assert np.array_equal(load_reference(config).tensor, bundled.tensor)

    with (destination / "figure1_curves.npz").open("ab") as artifact:
        artifact.write(b"tampered")
    with pytest.raises(ValueError, match="artifact checksum"):
        load_reference(config)
    fetch_reference(config)
    assert np.array_equal(load_reference(config).tensor, bundled.tensor)


def test_prediction_similarities_and_leave_one_out_features() -> None:
    curves = np.arange(4 * 4 * 5, dtype=np.float64).reshape(4, 4, 5) + 1
    similarities = _pairwise_similarities(curves)
    assert similarities["cosine"].shape == (16, 16)
    assert np.allclose(np.diag(similarities["cosine"]), 1)
    assert np.allclose(np.diag(similarities["l2"]), 0)
    features = _feature_cube(similarities["cosine"], "row", (0, 1), 4)
    assert features.shape == (4, 2, 4)
    assert np.isfinite(features).all()
