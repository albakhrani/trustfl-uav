"""Prepare and validate NEW immutable engineering scenarios; never historical recovery."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from .protocol import identity, validate_protocol, SUPPORTED_OPERATIONS
from .provenance import sha256_file, source_identity, tensor_state_hash, safe_new_target


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def potential_seed_blocks(rng, cfg, indices):
    """Enumerate bounded potential operation identities without drawing tensors."""
    rows = []
    def add(event, t, uid, shape, dtype, engine, **counters):
        digest = rng.digest(event, t, uid, **counters)
        raw = bytes.fromhex(digest)
        rows.append({"key": rng.key(event, t, uid, **counters), "digest": digest,
                     "torch_seed": int.from_bytes(raw[:8], "big") & ((1 << 63) - 1),
                     "numpy_seed_words_big_endian": [int.from_bytes(raw[i:i+4], "big") for i in range(0, 32, 4)],
                     "shape": shape, "dtype": dtype, "engine": engine})
    for t in range(cfg.train.rounds):
        for uid, samples in enumerate(indices["client_base"]):
            add("attack_random_delta", t, uid, [2437], "float32", "Torch CPU payload generator")
            for ep in range(cfg.train.local_epochs):
                add("minibatch_permutation", t, uid, [], "int64", "Torch CPU loader base seed", epoch=ep, operation="loader")
                for ident in samples:
                    add("per_sample_transform", t, uid, [32], "float32", "scoped CPU Torch/Python/NumPy; synthetic identity transform",
                        epoch=ep, sample=ident["index"], occurrence=0, operation="client_transform|"+ident["namespace"])
                for batch, offset in enumerate(range(0, len(samples), cfg.train.batch_size)):
                    size = min(cfg.train.batch_size, len(samples)-offset)
                    add("model_forward_stochastic_operations", t, uid, [size, 5], "float32", "scoped selected-device Torch; current MLP has no stochastic forward operator",
                        epoch=ep, batch=batch, operation="local")
                    add("attack_label_mask", t, uid, [size], "float64", "NumPy PCG64 uniform threshold input", epoch=ep, batch=batch)
        for purpose, samples, shape in (("root_validation", indices["root_base"], [32]),
                                        ("test_evaluation", indices["test_base"], [32])):
            add("server_validation_and_evaluation", t, -1, [], "int64", "Torch CPU loader base seed", operation="loader:"+purpose)
            add("server_validation_and_evaluation", t, -1, [len(samples), 5], "float32", "scoped selected-device Torch", operation=purpose)
            for ident in samples:
                add("per_sample_transform", t, -1, shape, "float32", "scoped CPU Torch/Python/NumPy; synthetic identity transform",
                    epoch=0, sample=ident["index"], occurrence=0, operation=purpose+"|"+ident["namespace"])
    return rows


def prepare(protocol, data_root, output):
    """Initialization/data preparation only; this function contains no training."""
    import numpy as np
    import torch
    from ..config import ExperimentConfig
    from ..data import load_dataset, split_server_val, iid_partition
    from ..models import build_model
    from ..uav.network import UAVNetwork
    from ..attacks.attacks import AttackController
    from .randomness import EventRandomness
    from .datasets import SavedTensorDataset, expand_indices, build_views
    from .records import write_json
    from .provenance import assert_imported_source, environment

    root = assert_imported_source()
    protocol = validate_protocol(protocol)
    data_root = Path(data_root)
    if not data_root.is_absolute() or not data_root.is_dir():
        raise ValueError("Existing absolute dataset root required; no downloads")
    target = safe_new_target(output, protected=[root, data_root])
    cfg = ExperimentConfig.from_dict(protocol["scientific"])
    cfg.data.data_dir = str(data_root.resolve())
    # Generated content cannot be an input to its own seed namespace.
    recipe = {"generator_sha256": sha256_file(root / "trustfl/data.py"),
              "generator": "existing _SyntheticBlobs 4000/1000 then saved320/80 prefix",
              "protocol": protocol, "schema": "cp_scenario_v1"}
    namespace = identity(recipe)
    rng = EventRandomness(protocol["master_seed"], namespace)
    data_seed = rng.torch_seed("dataset_samples_and_centers")
    train, test = load_dataset(cfg.data, data_seed)
    tx, ty = train.x[:320].clone(), train.y[:320].clone()
    vx, vy = test.x[:80].clone(), test.y[:80].clone()
    train_view = SavedTensorDataset(tx, ty, "synthetic/train")
    test_view = SavedTensorDataset(vx, vy, "synthetic/test")
    counts = np.bincount(ty.numpy(), minlength=5)
    if len(counts) != 5 or np.any(counts < 10):
        raise ValueError("Prefix must contain >=10 examples of each of5 classes; no reseed")
    rest, val = split_server_val(train_view, cfg.data.server_val_fraction,
                                 rng.torch_seed("root_and_client_partition", operation="root"))
    parts = iid_partition(rest, cfg.uav.num_uavs, rng.torch_seed("root_and_client_partition", operation="client"))
    if len(val) != 50 or [len(p) for p in parts] != [68, 68, 67, 67]:
        raise ValueError("Unexpected bounded fixture partition counts")
    indices = {
        "train_namespace": "synthetic/train", "test_namespace": "synthetic/test",
        "client_local": [p.tolist() for p in parts], "root_local": list(val.indices),
        "test_local": list(range(80)),
        "client_base": [expand_indices(train_view, p, "synthetic/train") for p in parts],
        "root_base": expand_indices(train_view, val.indices, "synthetic/train"),
        "test_base": expand_indices(test_view, range(80), "synthetic/test"),
    }
    build_views(train_view, test_view, indices)
    with rng.scope("initialization"):
        model = build_model(cfg.model, cfg.data.dataset)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    network = UAVNetwork(cfg.uav, rng.torch_seed("uav_initial_state_and_memberships"))
    attack = AttackController(cfg.attack, cfg.uav.num_uavs, rng.torch_seed("uav_initial_state_and_memberships", operation="malicious"))
    attack.apply_static_effects(network)
    rounds, clients = cfg.train.rounds, cfg.uav.num_uavs
    schedule, mobility, channel, coins, permutations = [], {}, {}, {}, {}
    for t in range(rounds):
        schedule.append(sorted(rng.numpy("invitation_schedule", t).choice(clients, cfg.trust.k_max, replace=False).tolist()))
        mobility[str(t)], channel[str(t)], coins[str(t)], permutations[str(t)] = {}, {}, {}, {}
        for uid in range(clients):
            mobility[str(t)][str(uid)] = rng.numpy("mobility_innovation", t, uid).normal(size=3).tolist()
            channel[str(t)][str(uid)] = float(rng.numpy("channel_unit_exponential_fading", t, uid).exponential(1.0))
            coins[str(t)][str(uid)] = {kind: float(rng.numpy(kind, t, uid).random())
                                      for kind in ("attack_participation", "attack_delay_branch")}
            permutations[str(t)][str(uid)] = {
                str(ep): rng.numpy("minibatch_permutation", t, uid, epoch=ep).permutation(len(parts[uid])).tolist()
                for ep in range(cfg.train.local_epochs)}
    hm = sorted(u.uid for u in network.uavs if u.high_mobility)
    malicious = sorted(attack.malicious)
    source = source_identity(root)
    bundle = {
        "schema": "cp_scenario_v1", "prospective": True, "engineering_only": True,
        "protocol": protocol, "protocol_hash": identity(protocol),
        "rng": {"master_seed": protocol["master_seed"], "namespace": namespace},
        "generation_recipe": recipe, "resolved_config": cfg.to_dict(),
        "indices": indices, "schedule": schedule,
        "traces": {"mobility_normal": mobility, "channel_exponential": channel, "attack_coins": coins},
        "permutations": permutations,
        "initial_uav": [_plain(asdict(u)) for u in network.uavs],
        "gm_mean": _plain(network._gm_mean),
        "memberships": {"malicious": malicious, "high_mobility": hm,
                        "overlap": sorted(set(malicious) & set(hm))},
        "model_tensor_hash": tensor_state_hash(initial_state),
        "dataset_tensor_hash": tensor_state_hash({"train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy}),
        "seed_registry": {"version": "cp_rng_v1", "source_identity_hash": identity(source["files"]),
                          "bounds": {"rounds": rounds, "clients": clients, "epochs": 1, "max_batches": 5},
                          "supported_operations": SUPPORTED_OPERATIONS,
                          "used_payload_hashes": "run-local event ledger only; never mutate this bundle"},
    }
    # Exclusive creation precedes every write. Preparation errors after this point
    # leave a visible, incomplete target which must never be reused automatically.
    target.mkdir()
    write_json(target / "preparation_status.json", {"status": "preparing", "prospective": True})
    try:
        torch.save({"train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy}, target / "data.pt")
        torch.save(initial_state, target / "initial_model.pt")
        write_json(target / "source_identity.json", source)
        write_json(target / "generation_environment.json", environment())
        write_json(target / "potential_seed_blocks.json", potential_seed_blocks(rng, cfg, indices))
        write_json(target / "preparation_events.json", rng.ledger)
        bundle["payloads"] = {p.name: sha256_file(p) for p in sorted(target.iterdir())
                              if p.name != "preparation_status.json"}
        bundle["bundle_id"] = identity(bundle)
        write_json(target / "scenario.json", bundle)
        write_json(target / "preparation_status.json", {"status": "complete", "bundle_id": bundle["bundle_id"]})
    except BaseException as exc:
        write_json(target / "preparation_status.json", {"status": "failed", "error": repr(exc)})
        raise
    return load_bundle(target, check_source=True)


def validate_payloads(directory, payloads):
    directory = Path(directory)
    for name, digest in payloads.items():
        if Path(name).name != name or (directory / name).is_symlink() or sha256_file(directory / name) != digest:
            raise ValueError("Scenario payload mismatch: " + name)


def validate_model_state(model, state, expected_hash=None):
    import torch
    reference = model.state_dict()
    if set(reference) != set(state):
        raise ValueError("Initial model tensor names mismatch")
    for name, tensor in state.items():
        if tensor.shape != reference[name].shape or tensor.dtype != reference[name].dtype:
            raise ValueError("Initial model shape/dtype mismatch: " + name)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite initial model")
    if expected_hash is not None and tensor_state_hash(state) != expected_hash:
        raise ValueError("Initial model logical identity mismatch")


def preflight_tensors(bundle, directory):
    """Validate consumed sample paths and model tensors before reserving a run."""
    import torch
    from ..config import ExperimentConfig
    from ..models import build_model
    from .datasets import SavedTensorDataset, build_views
    from .randomness import EventRandomness
    directory = Path(directory)
    inputs = torch.load(directory / "data.pt", map_location="cpu", weights_only=True)
    if set(inputs) != {"train_x", "train_y", "test_x", "test_y"} or tensor_state_hash(inputs) != bundle["dataset_tensor_hash"]:
        raise ValueError("Dataset tensor fields/hash mismatch")
    for split, size in (("train", 320), ("test", 80)):
        x, y = inputs[split + "_x"], inputs[split + "_y"]
        if x.shape != (size, 32) or y.shape != (size,) or x.dtype != torch.float32 or y.dtype != torch.int64:
            raise ValueError("Dataset tensor shape/dtype mismatch")
        if not bool(torch.isfinite(x).all()) or not bool(((y >= 0) & (y < 5)).all()):
            raise ValueError("Dataset contains invalid numeric values")
    train = SavedTensorDataset(inputs["train_x"], inputs["train_y"], "synthetic/train")
    test = SavedTensorDataset(inputs["test_x"], inputs["test_y"], "synthetic/test")
    build_views(train, test, bundle["indices"])
    cfg = ExperimentConfig.from_dict(bundle["resolved_config"])
    rng = EventRandomness(bundle["rng"]["master_seed"], bundle["rng"]["namespace"])
    with rng.scope("initialization", operation="validation"):
        model = build_model(cfg.model, cfg.data.dataset)
    initial = torch.load(directory / "initial_model.pt", map_location="cpu", weights_only=True)
    validate_model_state(model, initial, bundle["model_tensor_hash"])
    return inputs, initial


def load_bundle(path, check_source=False):
    directory = Path(path).resolve()
    bundle = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
    expected = bundle.pop("bundle_id")
    if identity(bundle) != expected:
        raise ValueError("Scenario manifest identity mismatch")
    bundle["bundle_id"] = expected
    if bundle["schema"] != "cp_scenario_v1" or bundle["protocol_hash"] != identity(validate_protocol(bundle["protocol"])):
        raise ValueError("Scenario schema/protocol mismatch")
    if identity(bundle["generation_recipe"]) != bundle["rng"]["namespace"]:
        raise ValueError("Scenario namespace mismatch")
    status = json.loads((directory / "preparation_status.json").read_text())
    if status != {"status": "complete", "bundle_id": expected}:
        raise ValueError("Incomplete scenario preparation")
    validate_payloads(directory, bundle["payloads"])
    cfg = bundle["resolved_config"]
    # Resolve through the unchanged native schema, then permit only the explicit
    # physical dataset path inserted by prepare; no hidden scientific override.
    from ..config import ExperimentConfig
    expected_cfg = ExperimentConfig.from_dict(bundle["protocol"]["scientific"]).to_dict()
    expected_cfg["data"]["data_dir"] = cfg["data"]["data_dir"]
    # Compare canonical JSON values: native tuple fields become lists on disk.
    if identity(cfg) != identity(expected_cfg):
        raise ValueError("Resolved scientific configuration differs from locked protocol")
    total, count, k = cfg["train"]["rounds"], cfg["uav"]["num_uavs"], cfg["trust"]["k_max"]
    if len(bundle["schedule"]) != total:
        raise ValueError("Wrong schedule length")
    for t, ids in enumerate(bundle["schedule"]):
        if len(ids) != k or ids != sorted(set(ids)) or any(type(i) is not int or i < 0 or i >= count for i in ids):
            raise ValueError("Malformed schedule")
        for uid in range(count):
            key = str(uid)
            noise = bundle["traces"]["mobility_normal"][str(t)][key]
            fade = bundle["traces"]["channel_exponential"][str(t)][key]
            if len(noise) != 3 or not all(math.isfinite(x) for x in noise) or not math.isfinite(fade) or fade < 0:
                raise ValueError("Invalid physical trace")
            coins = bundle["traces"]["attack_coins"][str(t)][key]
            if set(coins) != {"attack_participation", "attack_delay_branch"} or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1 for v in coins.values()):
                raise ValueError("Missing or invalid frozen attack coin")
            n = len(bundle["indices"]["client_local"][uid])
            for ep in range(cfg["train"]["local_epochs"]):
                perm = bundle["permutations"][str(t)][key][str(ep)]
                if sorted(perm) != list(range(n)):
                    raise ValueError("Malformed minibatch permutation")
    groups = bundle["memberships"]
    for key in ("malicious", "high_mobility"):
        if groups[key] != sorted(set(groups[key])) or any(type(i) is not int or not 0 <= i < count for i in groups[key]):
            raise ValueError("Malformed memberships")
    if groups["overlap"] != sorted(set(groups["malicious"]) & set(groups["high_mobility"])):
        raise ValueError("Wrong overlap")
    if len(groups["malicious"]) != round(count * cfg["attack"]["malicious_fraction"]) or len(groups["high_mobility"]) != round(count * cfg["uav"]["high_mobility_fraction"]):
        raise ValueError("Membership count disagrees with resolved configuration")
    if len(bundle["initial_uav"]) != count or set(bundle["gm_mean"]) != {str(i) for i in range(count)}:
        raise ValueError("Initial physical state count mismatch")
    for uid, state in enumerate(bundle["initial_uav"]):
        if state["uid"] != uid or bool(state["high_mobility"]) != (uid in groups["high_mobility"]):
            raise ValueError("Initial physical membership mismatch")
        for vector in (state["pos"], state["vel"], bundle["gm_mean"][str(uid)]):
            if len(vector) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
                raise ValueError("Invalid initial physical vector")
        for name in ("energy_j", "f_actual_hz", "f_reported_hz"):
            if type(state[name]) not in (int, float) or not math.isfinite(state[name]) or state[name] <= 0:
                raise ValueError("Invalid initial physical scalar")
        if not cfg["uav"]["f_min_hz"] <= state["f_actual_hz"] <= cfg["uav"]["f_max_hz"]:
            raise ValueError("Initial true hardware cap outside configured range")
    if check_source:
        saved = json.loads((directory / "source_identity.json").read_text())
        if saved["files"] != source_identity()["files"]:
            raise ValueError("Scenario producing source differs from current imported source")
    from .randomness import EventRandomness
    registry_rng = EventRandomness(bundle["rng"]["master_seed"], bundle["rng"]["namespace"])
    registered = json.loads((directory / "potential_seed_blocks.json").read_text())
    if registered != potential_seed_blocks(registry_rng, ExperimentConfig.from_dict(cfg), bundle["indices"]):
        raise ValueError("Potential stochastic operation registry mismatch")
    return bundle
