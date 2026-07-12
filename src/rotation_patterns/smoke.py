"""End-to-end execution of the bounded synthetic profile."""

from __future__ import annotations

from .config import ExperimentConfig
from .manifest import write_manifests
from .reproducibility import config_digest
from .results import aggregate_results


def run_smoke_profile(config: ExperimentConfig) -> None:
    if config.raw.get("profile") != "smoke":
        raise ValueError("the smoke command requires a configuration with profile: smoke")
    from .cli import _run_task

    manifests = config.output_root / "manifests"
    for experiment in ("classification", "segmentation", "hog"):
        paths = write_manifests(config, experiment, manifests, shard_size=50)
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                row_count = sum(1 for _ in handle) - 1
            for index in range(row_count):
                _run_task(config.path, path, index, force=False)
        frame = aggregate_results(
            config.output_root, experiment, config_sha256=config_digest(config.raw)
        )
        print(f"smoke {experiment}: {len(frame)} completed jobs")
