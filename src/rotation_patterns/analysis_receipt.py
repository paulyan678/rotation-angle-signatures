"""Reproduce the current curve analysis without training or changing historical results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import yaml

from .config import ExperimentConfig, load_config
from .prediction import run_prediction_experiment
from .reference import bundled_reference_metadata, fetch_reference


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(config_path: Path, output: Path) -> Path:
    if output.exists() and any(output.iterdir()):
        raise ValueError("analysis output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(config_path)
    config = ExperimentConfig(base.path, {**base.raw, "output_root": str(output)})
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.raw), encoding="utf-8")
    fetch_reference(config)
    trials, summary_path = run_prediction_experiment(config)
    summary = json.loads(summary_path.read_text())
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        {
            *root.glob("src/**/*.py"),
            *root.glob("configs/*.yaml"),
            *root.glob("requirements*.txt"),
            root / "pyproject.toml",
        }
    )
    source = {str(p.relative_to(root)): digest(p) for p in paths if p.is_file()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "PyYAML",
        "torch",
        "torchvision",
        "timm",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "Pillow",
        "pytest",
        "ruff",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    receipt = {
        "schema_version": 1,
        "kind": "current-explicit-interpretation-not-original-paper-reproduction",
        "git_commit": commit,
        "git_dirty": dirty,
        "source_files": source,
        "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "reference": bundled_reference_metadata(),
        "config_sha256": summary["config_sha256"],
        "train_axis_indices": summary["train_indices_zero_based"],
        "test_axis_indices": summary["test_indices_zero_based"],
        "comparison": summary["published_comparison"],
        "artifacts": {
            str(p.resolve().relative_to(output.resolve())): digest(p)
            for p in (trials, summary_path, (output / "resolved_config.yaml").resolve())
        },
        "boundary": "No source-image training, original classifier recovery, or Figure 3 replication.",
    }
    receipt_path = output / "receipt.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(reproduce(args.config, args.output))


if __name__ == "__main__":
    main()
