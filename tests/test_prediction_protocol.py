import json
from pathlib import Path

import numpy as np
import pytest

from rotation_patterns.analysis_receipt import reproduce
from rotation_patterns.prediction import _adam_weights, _feature_cube, _pairwise_similarities


def test_known_answer_similarities_and_query_exclusion():
    curves = np.array([[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [-1.0, 0.0]]])
    similarities = _pairwise_similarities(curves)
    cosine = _feature_cube(similarities["cosine"], "row", (0, 1), 2)
    distance = _feature_cube(similarities["l2"], "row", (0, 1), 2)
    np.testing.assert_allclose(cosine[0, 0], [1 / np.sqrt(2), -0.5])
    np.testing.assert_allclose(distance[0, 0], [-1, -(np.sqrt(2) + 2) / 2])


def test_balanced_shared_weights_learn_without_target_specific_parameters():
    features = np.repeat(np.eye(3)[:, None, :], 2, axis=1)
    weights, losses = _adam_weights(
        features, epochs=30, sample_size=4, learning_rate=0.01, rng=np.random.default_rng(4)
    )
    assert weights.shape == (3,)
    np.testing.assert_allclose(weights, weights[0])
    assert losses[-1] < losses[0] and np.all(weights > 1)


def test_axis_pools_do_not_use_curves_outside_the_selected_split():
    rng = np.random.default_rng(1)
    curves = rng.uniform(0.1, 1, size=(4, 4, 5))
    before = _feature_cube(_pairwise_similarities(curves)["cosine"], "row", (0, 1), 4)
    curves[:, 2:] = rng.uniform(1, 10, size=(4, 2, 5))
    after = _feature_cube(_pairwise_similarities(curves)["cosine"], "row", (0, 1), 4)
    np.testing.assert_allclose(before, after)


def test_current_bundled_analysis_has_a_reproducible_disclosed_baseline(tmp_path):
    receipt = reproduce(Path("configs/paper.yaml"), tmp_path)
    payload = json.loads(receipt.read_text())
    actual = {
        (r["axis"], r["metric"]): r["reconstructed_macro_mean"] for r in payload["comparison"]
    }
    expected = {
        ("row", "cosine"): 0.13546875,
        ("row", "l2"): 0.1142875,
        ("column", "cosine"): 0.0516375,
        ("column", "l2"): 0.2911375,
    }
    # This protects the current interpretation, not the much higher published labels.
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, abs=1e-10)
    assert payload["kind"].endswith("not-original-paper-reproduction")
    assert payload["source_files"] and payload["reference"]["measurement_count"] == 921600
    with pytest.raises(ValueError, match="new or empty"):
        reproduce(Path("configs/paper.yaml"), tmp_path)
