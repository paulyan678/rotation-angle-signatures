from pathlib import Path

from rotation_patterns.config import load_config
from rotation_patterns.manifest import expected_job_counts, read_job, write_manifests


def test_paper_angle_cardinality() -> None:
    config = load_config("configs/paper.yaml")
    assert len(config.angles("classification")) == 3600
    assert config.angles("classification")[-1] == 359.9
    assert len(config.angles("segmentation")) == 361
    assert config.angles("segmentation")[-1] == 360.0


def test_paper_job_counts() -> None:
    config = load_config("configs/paper.yaml")
    assert expected_job_counts(config) == {
        "classification": 921_600,
        "segmentation": 3_249,
        "hog": 1_083,
    }


def test_smoke_manifests_are_complete_and_indexable(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    assert expected_job_counts(config) == {
        "classification": 64,
        "segmentation": 2,
        "hog": 3,
    }
    paths = write_manifests(config, "classification", tmp_path, shard_size=13)
    assert len(paths) == 5
    first = read_job(paths[0], 0)
    assert first["experiment"] == "classification"
    assert first["angle"] == 0.0
    assert len(first["job_id"]) == 16
    assert len(first["config_sha256"]) == 64
    assert first["protocol_version"] == "0.2.0"


def test_manifest_rewrite_removes_stale_shards(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    assert len(write_manifests(config, "classification", tmp_path, shard_size=10)) == 7
    assert len(write_manifests(config, "classification", tmp_path, shard_size=64)) == 1
    assert [path.name for path in tmp_path.glob("classification-*.csv")] == [
        "classification-0000.csv"
    ]
