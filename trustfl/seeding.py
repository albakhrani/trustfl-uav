"""Reproducibility by construction: seeding + run metadata capture.

Feeds the AAAI reproducibility checklist items on seed methodology and
computing-infrastructure reporting.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch (CPU+CUDA). Called once per run.

    Per-client generators should be derived via `spawn_generator` so that
    changing the number of clients does not silently reshuffle everything.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cudnn determinism costs a little speed; acceptable at our scale.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def spawn_generator(base_seed: int, stream_id: int) -> np.random.Generator:
    """Independent, reproducible RNG stream (one per UAV / per module)."""
    return np.random.default_rng(np.random.SeedSequence([base_seed, stream_id]))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "not-a-git-repo"


@dataclass
class RunMeta:
    """Everything needed to state 'this exact run' in the paper."""
    seed: int
    config_dump: dict
    started_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    git_commit: str = field(default_factory=_git_commit)
    python: str = field(default_factory=platform.python_version)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_available: bool = field(default_factory=torch.cuda.is_available)
    gpu_name: str = field(default_factory=lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    hostname: str = field(default_factory=platform.node)
    os_name: str = field(default_factory=platform.platform)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
