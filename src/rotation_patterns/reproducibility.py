"""Determinism, atomic results, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import PIL

from . import __version__

PROTOCOL_VERSION = __version__


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def stable_job_id(job: dict[str, Any]) -> str:
    payload = json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def config_digest(raw: dict[str, Any]) -> str:
    """Hash the fully parsed configuration rather than its incidental YAML layout."""

    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, sort_keys=True, indent=2)
        tmp.write("\n")
        temp_name = tmp.name
    os.replace(temp_name, path)


def provenance() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        commit = completed.stdout.strip() if completed.returncode == 0 else ""
    except OSError:
        commit = ""
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit or None,
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
            }
        )
        try:
            import torchvision

            result["torchvision"] = torchvision.__version__
        except ImportError:
            result["torchvision"] = None
        try:
            import timm

            result["timm"] = timm.__version__
        except ImportError:
            result["timm"] = None
    except ImportError:
        result["torch"] = None
    return result


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
