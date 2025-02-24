import zipfile

import numpy as np
import pytest

from rotation_patterns.prediction import _feature_cube, _pairwise_similarities
from rotation_patterns.reference import _archive_members, _read_curve


def test_reference_curve_parser_checks_paper_grid(tmp_path) -> None:
    path = tmp_path / "plot_1_1.csv"
    angles = np.arange(3600, dtype=np.float64) / 10
    values = np.column_stack([angles, np.full(3600, 0.75)])
    np.savetxt(path, values, delimiter=",", header="RotationAngle,Accuracy", comments="")
    parsed_angles, accuracy = _read_curve(path)
    assert np.array_equal(parsed_angles, angles)
    assert np.all(accuracy == 0.75)


def test_reference_archive_rejects_traversal(tmp_path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../raw_data/plot_1_1.csv", "bad")
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="unsafe path"):
        _archive_members(archive)


def test_prediction_similarities_and_leave_one_out_features() -> None:
    curves = np.arange(4 * 4 * 5, dtype=np.float64).reshape(4, 4, 5) + 1
    similarities = _pairwise_similarities(curves)
    assert similarities["cosine"].shape == (16, 16)
    assert np.allclose(np.diag(similarities["cosine"]), 1)
    assert np.allclose(np.diag(similarities["l2"]), 0)
    features = _feature_cube(similarities["cosine"], "row", (0, 1), 4)
    assert features.shape == (4, 2, 4)
    assert np.isfinite(features).all()

