import numpy as np

from rotation_patterns.figures import _load_examples


def test_figure4_six_array_bundle_loads_as_original_and_rotated(tmp_path) -> None:
    root = tmp_path / "examples"
    root.mkdir()
    prefix = "abc_brats_angle_95_"
    arrays = {
        "original_image": np.zeros((4, 8, 8), dtype=np.float32),
        "ground_truth_mask": np.zeros((8, 8), dtype=np.uint8),
        "predicted_mask": np.zeros((8, 8), dtype=np.uint8),
        "rotated_image": np.ones((4, 8, 8), dtype=np.float32),
        "rotated_ground_truth_mask": np.ones((8, 8), dtype=np.uint8),
        "rotated_predicted_mask": np.ones((8, 8), dtype=np.uint8),
    }
    for name, value in arrays.items():
        np.save(root / f"{prefix}{name}.npy", value, allow_pickle=False)
    examples = _load_examples(tmp_path)
    assert len(examples) == 2
    assert {example.angle for example in examples} == {0.0, 95.0}

