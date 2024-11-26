"""Transparent scale estimates for experiment planning."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .manifest import expected_job_counts


def estimate(config: ExperimentConfig) -> dict[str, Any]:
    jobs = expected_job_counts(config)
    neural_fits = jobs["classification"] * 2 + jobs["segmentation"] * 2
    svm_fits = jobs["hog"]
    return {
        "jobs": jobs,
        "total_jobs": sum(jobs.values()),
        "neural_model_fits": neural_fits,
        "cross_validated_svm_jobs": svm_fits,
        "note": "SVM CV folds/grid candidates each perform additional internal fits.",
    }

