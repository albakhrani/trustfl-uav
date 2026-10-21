"""Strict, engineering-only common-policy protocol; native config is untouched."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "cp_protocol_v1"
BASE_COMMIT = "a0cd40ce4939a661fa17bc7b94f4c18f59f4360b"
SUPPORTED_OPERATIONS = {
    "runner_datasets": ["synthetic"],
    "runner_models": ["mlp"],
    "methods": ["trustfl", "fedavg"],
    "devices": ["cpu", "cuda"],
    "workers": 0,
    "execution": "sequential",
    "transforms": "EventDataset scopes sample access; real-dataset runner support is gated",
    "scope": "AE3/R3.4 engineering fixtures; claim_eligible=false",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def fixture_protocol(condition):
    if condition not in ("benign", "adaptive"):
        raise ValueError("Only the approved benign/adaptive engineering fixtures exist")
    adaptive = condition == "adaptive"
    return {
        "schema": SCHEMA, "rng_version": "cp_rng_v1", "engineering_only": True,
        "condition": condition, "master_seed": 31001,
        "prefix": {"train": 320, "test": 80},
        "scientific": {
            "name": "common_policy_engineering_" + condition, "seed": 31001,
            "data": {"dataset": "synthetic", "partition": "iid", "server_val_fraction": 0.01},
            "model": {"name": "mlp", "num_classes": 5},
            "train": {"rounds": 6 if adaptive else 3, "local_epochs": 1, "batch_size": 16,
                      "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "fedprox_mu": 0.0,
                      "eval_every": 1, "device": "cpu"},
            "uav": {"enabled": True, "num_uavs": 4, "battery_j": 200000.0,
                    "e_min_j": 10000.0, "idle_energy_drain": True,
                    "high_mobility_fraction": 0.5, "f_max_hz": 2000000000.0,
                    "p_tx_min_w": 0.05, "p_tx_max_w": 1.0},
            "trust": {"k_max": 2, "weight_mode": "learnable"},
            "attack": {"attack_type": "adaptive_system" if adaptive else "none",
                       "malicious_fraction": 0.25 if adaptive else 0.0,
                       "adaptive_probe_rounds": 5, "adaptive_trust_target": 0.6},
        },
    }


def validate_protocol(value):
    if not isinstance(value, dict) or canonical(value) != canonical(fixture_protocol(value.get("condition"))):
        raise ValueError("Protocol differs from the approved bounded engineering schema; no implicit grid/override")
    return copy.deepcopy(value)


def load_protocol(path):
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))
