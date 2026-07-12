"""Result paths, schema validation, and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .reproducibility import stable_job_id


REQUIRED_RESULT_KEYS = {
    "job_id",
    "experiment",
    "dataset",
    "method",
    "encoder",
    "angle",
    "seed",
    "metric_name",
    "metric_value",
}
JOB_KEYS = (
    "experiment",
    "dataset",
    "method",
    "encoder",
    "angle",
    "seed",
    "config_sha256",
    "protocol_version",
)
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def result_path(output_root: Path, job: dict[str, Any]) -> Path:
    if job.get("experiment") not in {"classification", "segmentation", "hog"}:
        raise ValueError(f"invalid result experiment: {job.get('experiment')!r}")
    missing = [key for key in JOB_KEYS if key not in job]
    if missing:
        raise ValueError(f"job is missing identity keys: {missing}")
    payload = {key: job[key] for key in JOB_KEYS}
    expected = stable_job_id(payload)
    job_id = str(job.get("job_id", expected))
    if not JOB_ID_PATTERN.fullmatch(job_id) or job_id != expected:
        raise ValueError("job_id does not match the canonical manifest payload")
    return output_root / "jobs" / str(job["experiment"]) / f"{job_id}.json"


def validate_result(result: dict[str, Any]) -> None:
    missing = REQUIRED_RESULT_KEYS - result.keys()
    if missing:
        raise ValueError(f"result is missing required keys: {sorted(missing)}")
    value = float(result["metric_value"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"metric_value must be in [0, 1], got {value}")


def aggregate_results(
    output_root: Path, experiment: str, config_sha256: str | None = None
) -> pd.DataFrame:
    rows = []
    job_dir = output_root / "jobs" / experiment
    for path in sorted(job_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        validate_result(row)
        fingerprint = row.get("config_sha256") or row.get("provenance", {}).get(
            "config_sha256"
        )
        if config_sha256 is not None and fingerprint != config_sha256:
            continue
        if config_sha256 is not None:
            missing_identity = [key for key in JOB_KEYS if key not in row]
            if missing_identity:
                raise ValueError(f"result {path} is missing identity keys: {missing_identity}")
            expected_id = stable_job_id({key: row[key] for key in JOB_KEYS})
            if row["job_id"] != expected_id or path.stem != expected_id:
                raise ValueError(f"result identity or filename is not canonical: {path}")
        if fingerprint is not None:
            row["config_sha256"] = fingerprint
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no {experiment} result JSON files under {job_dir}")
    frame = pd.DataFrame(rows)
    identity = ["dataset", "method", "encoder", "angle", "seed"]
    if frame.duplicated(identity).any():
        raise ValueError(f"duplicate {experiment} result identities under {job_dir}")
    frame = frame.sort_values(["dataset", "method", "encoder", "angle"])
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(aggregate_dir / f"{experiment}.csv", index=False)
    return frame
