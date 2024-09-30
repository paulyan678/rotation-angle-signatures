"""Resumable array-job manifests for the large experiment grid."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator

from .config import ExperimentConfig
from .reproducibility import PROTOCOL_VERSION, config_digest, stable_job_id

FIELDS = (
    "job_id",
    "experiment",
    "dataset",
    "method",
    "encoder",
    "angle",
    "seed",
    "config_sha256",
    "protocol_version",
)


def write_manifests(
    config: ExperimentConfig,
    experiment: str,
    destination: Path,
    shard_size: int = 10_000,
) -> list[Path]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = config_digest(config.raw)
    with tempfile.TemporaryDirectory(prefix=f".{experiment}-", dir=destination) as temp_name:
        staged_root = Path(temp_name)
        staged_paths: list[Path] = []
        handle = None
        writer = None
        count = 0
        try:
            for count, job in enumerate(config.jobs(experiment), start=1):
                offset = count - 1
                if offset % shard_size == 0:
                    if handle is not None:
                        handle.close()
                    path = staged_root / f"{experiment}-{len(staged_paths):04d}.csv"
                    staged_paths.append(path)
                    handle = path.open("w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                assert writer is not None
                payload = {
                    **job,
                    "config_sha256": fingerprint,
                    "protocol_version": PROTOCOL_VERSION,
                }
                writer.writerow({"job_id": stable_job_id(payload), **payload})
        finally:
            if handle is not None:
                handle.close()
        summary = {
            "experiment": experiment,
            "jobs": count,
            "shard_size": shard_size,
            "config_sha256": fingerprint,
            "protocol_version": PROTOCOL_VERSION,
            "shards": [path.name for path in staged_paths],
        }
        summary_name = f"{experiment}-manifest.json"
        (staged_root / summary_name).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for old_path in destination.glob(f"{experiment}-*.csv"):
            old_path.unlink()
        for path in staged_paths:
            os.replace(path, destination / path.name)
        os.replace(staged_root / summary_name, destination / summary_name)
        return [destination / path.name for path in staged_paths]


def read_job(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise IndexError("manifest index must be non-negative")
    with path.open(newline="", encoding="utf-8") as handle:
        for current, row in enumerate(csv.DictReader(handle)):
            if current == index:
                row["angle"] = float(row["angle"])
                row["seed"] = int(row["seed"])
                return row
    raise IndexError(f"manifest index {index} is out of range for {path}")


def iter_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["angle"] = float(row["angle"])
                row["seed"] = int(row["seed"])
                yield row


def expected_job_counts(config: ExperimentConfig) -> dict[str, int]:
    classification = (
        len(config.raw["datasets"])
        * len(config.raw["encoders"])
        * len(config.raw["methods"])
        * len(config.angles("classification"))
    )
    segmentation = config.section("segmentation")
    seg_count = (
        len(segmentation["datasets"])
        * len(segmentation["encoders"])
        * len(config.angles("segmentation"))
    )
    hog = config.section("hog")
    hog_count = len(hog["datasets"]) * len(config.angles("hog"))
    return {"classification": classification, "segmentation": seg_count, "hog": hog_count}


def slurm_array_bounds(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8") as handle:
        rows = max(sum(1 for _ in handle) - 1, 0)
    if rows == 0:
        raise ValueError(f"manifest has no jobs: {path}")
    return 0, rows - 1
