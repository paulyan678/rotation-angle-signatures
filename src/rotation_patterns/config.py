"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is malformed."""


@dataclass(frozen=True)
class AngleGrid:
    start: Decimal
    stop: Decimal
    step: Decimal
    inclusive: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "AngleGrid":
        try:
            grid = cls(
                start=Decimal(str(value["start"])),
                stop=Decimal(str(value["stop"])),
                step=Decimal(str(value["step"])),
                inclusive=bool(value.get("inclusive", True)),
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError("angle grid requires start, stop, and step") from exc
        if grid.step <= 0:
            raise ConfigError("angle step must be positive")
        if grid.stop < grid.start:
            raise ConfigError("angle stop must not be less than start")
        return grid

    def values(self) -> list[float]:
        values: list[float] = []
        current = self.start
        compare = (lambda x: x <= self.stop) if self.inclusive else (lambda x: x < self.stop)
        while compare(current):
            values.append(float(current))
            current += self.step
        return values


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output_root"])

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data_root"])

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"missing mapping section: {name}")
        return value

    def angles(self, section: str) -> list[float]:
        return AngleGrid.from_mapping(self.section(section)["angles"]).values()

    def jobs(self, experiment: str) -> Iterator[dict[str, Any]]:
        if experiment == "classification":
            for dataset in self.raw["datasets"]:
                for method in self.raw["methods"]:
                    for encoder in self.raw["encoders"]:
                        for angle in self.angles("classification"):
                            yield {
                                "experiment": experiment,
                                "dataset": dataset,
                                "method": method,
                                "encoder": encoder,
                                "angle": angle,
                                "seed": self.seed,
                            }
        elif experiment == "segmentation":
            section = self.section("segmentation")
            for dataset in section["datasets"]:
                for encoder in section["encoders"]:
                    for angle in self.angles("segmentation"):
                        yield {
                            "experiment": experiment,
                            "dataset": dataset,
                            "method": "mocov2",
                            "encoder": encoder,
                            "angle": angle,
                            "seed": self.seed,
                        }
        elif experiment == "hog":
            section = self.section("hog")
            for dataset in section["datasets"]:
                for angle in self.angles("hog"):
                    yield {
                        "experiment": experiment,
                        "dataset": dataset,
                        "method": "hog_svm",
                        "encoder": "hog",
                        "angle": angle,
                        "seed": self.seed,
                    }
        else:
            raise ConfigError(f"unknown experiment: {experiment}")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")
    for key in (
        "seed",
        "data_revision",
        "data_root",
        "output_root",
        "datasets",
        "encoders",
        "methods",
    ):
        if key not in raw:
            raise ConfigError(f"missing required configuration key: {key}")
    for key in ("datasets", "encoders", "methods"):
        if not isinstance(raw[key], list) or not raw[key]:
            raise ConfigError(f"{key} must be a non-empty list")
    unknown_methods = set(raw["methods"]) - {"mocov2", "simclr"}
    if unknown_methods:
        raise ConfigError(f"unsupported contrastive methods: {sorted(unknown_methods)}")
    config = ExperimentConfig(config_path, raw)
    for section in ("classification", "segmentation", "hog"):
        config.angles(section)
    config.section("prediction")
    return config
