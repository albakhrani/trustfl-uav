"""Read-only source identity and strict JSON helpers for prospective attempts."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_imported_source(expected=None):
    expected = Path(expected or Path.cwd()).resolve()
    if Path.cwd().resolve() != expected:
        raise RuntimeError("Working directory must equal the intended isolated source")
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("PYTHONPATH is set; report the conflict without altering the shared environment")
    for name in ("trustfl", "trustfl.fl.server", "trustfl.fl.client", "trustfl.fl.aggregation",
                 "trustfl.uav.network", "trustfl.attacks.attacks", "trustfl.data", "trustfl.models"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(expected / "trustfl"):
            raise RuntimeError(f"Imported-source conflict for {name}: {path}")
    return expected


def source_identity(root=None):
    root = Path(root or Path.cwd()).resolve()
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), *args], text=True)
    paths = sorted((root / "trustfl").rglob("*.py"))
    files = {}
    for path in paths:
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            "sha256_raw": hashlib.sha256(raw).hexdigest(),
            "sha256_lf_normalized": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        }
    return {"git_head": git("rev-parse", "HEAD").strip(), "files": files,
            "dirty_scientific_diff": git("diff", "HEAD", "--", "trustfl"),
            "untracked_scientific_files": git("ls-files", "--others", "--exclude-standard", "--", "trustfl").splitlines()}


def environment():
    import numpy as np
    import torch
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "command": sys.argv, "cwd": str(Path.cwd().resolve()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def tensor_state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        h.update(value.numpy().tobytes(order="C"))
    return h.hexdigest()


def safe_new_target(path, protected=()):
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError("An absolute target without traversal is required")
    for parent in [raw, *raw.parents]:
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("Symlink/junction targets are not supported")
    resolved = raw.resolve()
    for parent in protected:
        protected_path = Path(parent).resolve()
        if resolved == protected_path or resolved.is_relative_to(protected_path):
            raise ValueError("Target intersects a protected source/data/historical root")
    if raw.exists():
        raise FileExistsError(str(raw))
    if not raw.parent.is_dir():
        raise ValueError("Target parent must already exist")
    return raw
