"""Versioned scientific-only precision scope; no import-time numerical dependency.

This policy is separate from model/operator and RNG identities.  It is not a
training authorization and does not affect any native entry point.  All settings
use the measured legacy allow_tf32 family, never the newer fp32_precision API.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os


POLICY_ID = "scientific-precision-v3-2-strict-legacy-no-tf32-v1"
_SETTINGS = {
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": False,
    "cuda_matmul_allow_tf32": False,
    "cublas_workspace_config": ":4096:8",
}
_ACTIVE = []
# A later scope may reuse CUDA initialized by this helper in the same process.
# Pre-existing CUDA initialization without this observation is rejected, even if
# the environment currently happens to contain the expected cuBLAS value.
_CUDA_ESTABLISHED = []


def precision_identity():
    """Return the exact versioned policy, without importing Torch."""
    value = {"policy_id": POLICY_ID, "api_family": "legacy_allow_tf32",
             "settings": copy.deepcopy(_SETTINGS),
             "scope": "all_scientific_methods_entire_numerical_execution"}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    value["identity_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def require_precision_identity(value):
    """Reject old/missing/mutated identities before numerical module import."""
    # JSON canonical equality also distinguishes bools from integer substitutes.
    try:
        actual = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid scientific precision execution identity") from exc
    expected = json.dumps(precision_identity(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if actual != expected:
        raise ValueError("Missing/mismatched scientific precision identity; prepare a fresh scenario")
    return precision_identity()


def _torch(module):
    if module is None:
        import torch
        return torch
    return module


def read_precision(torch_module):
    """Read supported legacy settings only; no CUDA initialization is requested."""
    t = torch_module
    for name in ("use_deterministic_algorithms", "are_deterministic_algorithms_enabled",
                 "is_deterministic_algorithms_warn_only_enabled"):
        if not callable(getattr(t, name, None)):
            raise RuntimeError("Required strict precision API unavailable: " + name)
    if not callable(getattr(t.cuda, "is_initialized", None)):
        raise RuntimeError("Required CUDA initialization state API unavailable")
    result = {
        "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": t.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": t.backends.cudnn.deterministic,
        "cudnn_benchmark": t.backends.cudnn.benchmark,
        "cudnn_allow_tf32": t.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": t.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if any(type(result[key]) is not bool for key in _SETTINGS if key != "cublas_workspace_config"):
        raise RuntimeError("Legacy precision API did not return exact boolean settings")
    return result


def require_precision_active(expected_identity, *, torch_module=None):
    require_precision_identity(expected_identity)
    t = _torch(torch_module)
    if not any(item is t for item in _ACTIVE):
        raise RuntimeError("Scientific numerical dispatch requires the shared precision scope")
    if read_precision(t) != _SETTINGS:
        raise RuntimeError("Scientific precision state changed during execution")
    return True


class PrecisionRestorationError(RuntimeError):
    """A failed restoration prevents a clean execution certificate."""


@contextmanager
def precision_policy_scope(expected_identity, *, evidence=None, torch_module=None):
    """Apply/read back policy and independently restore every prior setting.

    Identity validation precedes lazy Torch import.  If CUDA was already
    initialized outside this helper, fail before dispatch: an environment value
    seen after initialization cannot establish historical cuBLAS configuration.
    Caller-owned ``evidence`` remains available after exceptions/restoration.
    Fake-state fixtures may inject a tiny module-shaped object; no model API is
    required or used by this helper.
    """
    identity = require_precision_identity(expected_identity)
    audit = {} if evidence is None else evidence
    if not isinstance(audit, dict) or audit:
        raise ValueError("Precision evidence must be a fresh empty dict")
    audit.update(identity=identity, requested=copy.deepcopy(_SETTINGS), status="entering")
    t = _torch(torch_module)
    before = read_precision(t)
    initialized = t.cuda.is_initialized()
    if type(initialized) is not bool:
        raise RuntimeError("CUDA initialization readback must be boolean")
    audit.update(before=before, cuda_initialized_before=initialized,
                 cuda_initialization_provenance="prior_shared_scope" if initialized else "uninitialized_before_policy")
    if initialized and not any(item is t for item in _CUDA_ESTABLISHED + _ACTIVE):
        audit["status"] = "rejected_preexisting_cuda_initialization"
        raise RuntimeError("CUDA was initialized before shared precision policy; use a fresh process")
    registered = False
    body_error = None
    try:
        # CUDA initialization is prohibited above until the exact environment is
        # in place.  Neither these setters nor the readback initializes CUDA.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _SETTINGS["cublas_workspace_config"]
        t.use_deterministic_algorithms(True, warn_only=False)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.allow_tf32 = False
        t.backends.cuda.matmul.allow_tf32 = False
        effective = read_precision(t)
        audit["effective"] = effective
        if effective != _SETTINGS:
            raise RuntimeError("Scientific precision setting readback mismatch before dispatch")
        _ACTIVE.append(t)
        registered = True
        audit["status"] = "active"
        yield audit
        require_precision_active(identity, torch_module=t)
        audit["status"] = "body_complete"
    except BaseException as exc:
        body_error = exc
        audit.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if registered:
            # Pop our own nested activation, without removing an outer scope.
            for index in range(len(_ACTIVE) - 1, -1, -1):
                if _ACTIVE[index] is t:
                    del _ACTIVE[index]
                    break
        errors = []
        try:
            audit["cuda_initialized_after_body"] = t.cuda.is_initialized()
            if audit["cuda_initialized_after_body"] and not any(item is t for item in _CUDA_ESTABLISHED):
                _CUDA_ESTABLISHED.append(t)
        except BaseException as exc:
            errors.append({"setting": "cuda_initialization_readback", "error": repr(exc)})
        restorers = (
            ("deterministic_algorithms_and_warn_only", lambda: t.use_deterministic_algorithms(
                before["deterministic_algorithms"], warn_only=before["deterministic_warn_only"])),
            ("cudnn_deterministic", lambda: setattr(t.backends.cudnn, "deterministic", before["cudnn_deterministic"])),
            ("cudnn_benchmark", lambda: setattr(t.backends.cudnn, "benchmark", before["cudnn_benchmark"])),
            ("cudnn_allow_tf32", lambda: setattr(t.backends.cudnn, "allow_tf32", before["cudnn_allow_tf32"])),
            ("cuda_matmul_allow_tf32", lambda: setattr(t.backends.cuda.matmul, "allow_tf32", before["cuda_matmul_allow_tf32"])),
        )
        attempted = []
        for name, restore in restorers:
            attempted.append(name)
            try:
                restore()
            except BaseException as exc:
                errors.append({"setting": name, "error": repr(exc)})
        attempted.append("cublas_workspace_config")
        try:
            if before["cublas_workspace_config"] is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = before["cublas_workspace_config"]
        except BaseException as exc:
            errors.append({"setting": "cublas_workspace_config", "error": repr(exc)})
        # Read every setting independently so one failed getter cannot conceal
        # whether the remaining flags/environment were restored.
        getters = (
            ("deterministic_algorithms", t.are_deterministic_algorithms_enabled),
            ("deterministic_warn_only", t.is_deterministic_algorithms_warn_only_enabled),
            ("cudnn_deterministic", lambda: t.backends.cudnn.deterministic),
            ("cudnn_benchmark", lambda: t.backends.cudnn.benchmark),
            ("cudnn_allow_tf32", lambda: t.backends.cudnn.allow_tf32),
            ("cuda_matmul_allow_tf32", lambda: t.backends.cuda.matmul.allow_tf32),
            ("cublas_workspace_config", lambda: os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
        )
        after = {}
        for name, getter in getters:
            try:
                after[name] = getter()
                if type(after[name]) is not type(before[name]) or after[name] != before[name]:
                    errors.append({"setting": name, "error": "restored value differs from prior state"})
            except BaseException as exc:
                after[name] = {"read_error": repr(exc)}
                errors.append({"setting": name, "error": repr(exc)})
        audit.update(after=after, restoration={"attempted": attempted, "errors": errors,
                                               "status": "FAIL" if errors else "PASS"})
        if errors:
            audit["status"] = "restoration_failed"
            raise PrecisionRestorationError("Scientific precision restoration failed; inspect precision evidence") from body_error
        if body_error is None:
            audit["status"] = "complete"
