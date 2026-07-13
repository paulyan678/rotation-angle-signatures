"""Command-line entry point for every experiment in the paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .manifest import expected_job_counts, read_job, write_manifests
from .results import aggregate_results, result_path, validate_result
from .reproducibility import PROTOCOL_VERSION, config_digest


def _run_task(config_path: Path, manifest_path: Path, index: int, force: bool) -> Path:
    config = load_config(config_path)
    job = read_job(manifest_path, index)
    if job.get("config_sha256") != config_digest(config.raw):
        raise ValueError("manifest configuration fingerprint does not match --config")
    if job.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("manifest protocol version does not match this installation")
    destination = result_path(config.output_root, job)
    if destination.exists() and not force:
        try:
            with destination.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            validate_result(existing)
            if (
                existing.get("job_id") == job["job_id"]
                and existing.get("protocol_version") == PROTOCOL_VERSION
                and existing.get("provenance", {}).get("config_sha256")
                == job["config_sha256"]
            ):
                print(f"skip completed job: {destination}")
                return destination
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        print(f"recompute invalid or stale result: {destination}")

    experiment = job["experiment"]
    if experiment == "classification":
        from .experiments.classification import run_classification_job

        result = run_classification_job(config, job)
    elif experiment == "segmentation":
        from .experiments.segmentation import run_segmentation_job

        result = run_segmentation_job(config, job)
    elif experiment == "hog":
        from .experiments.hog import run_hog_job

        result = run_hog_job(config, job)
    else:
        raise ValueError(f"unknown experiment: {experiment}")

    from .reproducibility import atomic_write_json

    atomic_write_json(destination, result)
    print(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rotation-patterns",
        description="Run the rotation-angle signature thesis experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifests = subparsers.add_parser("make-manifests", help="write sharded job manifests")
    manifests.add_argument("--config", type=Path, required=True)
    manifests.add_argument(
        "--experiment", choices=["classification", "segmentation", "hog", "all"], default="all"
    )
    manifests.add_argument("--shard-size", type=int, default=10_000)

    run_task = subparsers.add_parser("run-task", help="run one manifest row")
    run_task.add_argument("--config", type=Path, required=True)
    run_task.add_argument("--manifest", type=Path, required=True)
    run_task.add_argument("--index", type=int, required=True)
    run_task.add_argument("--force", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="combine atomic job results")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--experiment", choices=["classification", "segmentation", "hog"], required=True)

    figures = subparsers.add_parser("figures", help="generate Figures 1 through 4")
    figures.add_argument("--config", type=Path, required=True)
    figures.add_argument(
        "--only",
        choices=["all", "figure1", "figure2", "figure3", "figure4"],
        default="all",
    )

    reference = subparsers.add_parser(
        "fetch-reference", help="fetch and validate my published curve CSVs"
    )
    reference.add_argument("--config", type=Path, required=True)

    estimator = subparsers.add_parser("estimate", help="report configured job and fit counts")
    estimator.add_argument("--config", type=Path, required=True)

    prediction = subparsers.add_parser("prediction", help="run the Figure 2 prediction experiment")
    prediction.add_argument("--config", type=Path, required=True)

    smoke = subparsers.add_parser("smoke", help="run all smoke-profile experiments")
    smoke.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "make-manifests":
        config = load_config(args.config)
        experiments = (
            ["classification", "segmentation", "hog"]
            if args.experiment == "all"
            else [args.experiment]
        )
        destination = config.output_root / "manifests"
        for experiment in experiments:
            paths = write_manifests(config, experiment, destination, args.shard_size)
            print(f"{experiment}: {len(paths)} shard(s)")
        print(json.dumps(expected_job_counts(config), sort_keys=True))
        return 0
    if args.command == "run-task":
        _run_task(args.config, args.manifest, args.index, args.force)
        return 0
    if args.command == "aggregate":
        config = load_config(args.config)
        frame = aggregate_results(
            config.output_root, args.experiment, config_sha256=config_digest(config.raw)
        )
        print(f"aggregated {len(frame)} rows")
        return 0
    if args.command == "prediction":
        from .prediction import run_prediction_experiment

        run_prediction_experiment(load_config(args.config))
        return 0
    if args.command == "figures":
        from .figures import create_all_figures

        create_all_figures(load_config(args.config), only=args.only)
        return 0
    if args.command == "fetch-reference":
        from .reference import fetch_reference

        destination = fetch_reference(load_config(args.config))
        print(destination)
        return 0
    if args.command == "estimate":
        from .compute import estimate

        print(json.dumps(estimate(load_config(args.config)), indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        from .smoke import run_smoke_profile

        run_smoke_profile(load_config(args.config))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
