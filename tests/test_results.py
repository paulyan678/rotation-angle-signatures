import json
from pathlib import Path

import pytest

from rotation_patterns.reproducibility import stable_job_id
from rotation_patterns.results import aggregate_results, result_path, validate_result


def _result(job_id: str, angle: float, value: float) -> dict:
    return {
        "job_id": job_id,
        "experiment": "hog",
        "dataset": "synthetic_a",
        "method": "hog_svm",
        "encoder": "hog",
        "angle": angle,
        "seed": 2025,
        "metric_name": "accuracy",
        "metric_value": value,
    }


def test_result_schema_and_aggregation(tmp_path: Path) -> None:
    directory = tmp_path / "jobs" / "hog"
    directory.mkdir(parents=True)
    rows = [_result("a", 0.0, 0.5), _result("b", 45.0, 0.75)]
    for row in rows:
        validate_result(row)
        (directory / f"{row['job_id']}.json").write_text(json.dumps(row), encoding="utf-8")
    frame = aggregate_results(tmp_path, "hog")
    assert frame["metric_value"].tolist() == [0.5, 0.75]
    assert (tmp_path / "aggregate" / "hog.csv").exists()


def test_result_path_rejects_forged_manifest_id(tmp_path: Path) -> None:
    payload = {
        "experiment": "hog",
        "dataset": "synthetic_a",
        "method": "hog_svm",
        "encoder": "hog",
        "angle": 0.0,
        "seed": 2025,
        "config_sha256": "a" * 64,
        "protocol_version": "0.1.0",
    }
    valid = {**payload, "job_id": stable_job_id(payload)}
    assert result_path(tmp_path, valid).name.endswith(".json")
    with pytest.raises(ValueError, match="job_id"):
        result_path(tmp_path, {**valid, "job_id": "../overwrite"})
