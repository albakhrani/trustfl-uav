"""Strict prospective real-data protocol. Legacy/engineering schemas are untouched.

The two public profiles are immutable recipes, not free-form ExperimentConfig
input. Scientific execution remains a separately invoked, user-authorized step.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from ..comparison.protocol import identity, canonical

SCHEMA = "scientific_protocol_v1"
V5_SCHEMA = "scientific_protocol_v5"
V5_PROFILE = "sampled_clients_reserved_root_v5"
V5_SEEDS = {"emnist_byclass": tuple(range(41000,41010)), "eurosat": tuple(range(42000,42006))}
RNG_VERSION = "scientific_rng_v1"
METHODS = ("trustfl", "fedavg", "fltrust_declared_root_fallback", "fedgreed_algorithm1", "fd_distance_validation_local")
POLICIES = ("common", "prospective_native_policy")
DATASETS = ("mnist", "emnist_byclass", "eurosat")
LEGACY_METHODS = dict(zip(METHODS, ("trustfl", "fedavg", "fltrust", "fedgreed", "fedaa")))
DATASET_SPECS = {
 "mnist": {"legacy_key":"mnist", "model":"cnn_mnist", "classes":10, "train_count":60000, "test_count":10000, "root_count":600, "rounds":100, "epochs":2, "parameters":1663370},
 "emnist_byclass": {"legacy_key":"femnist", "model":"cnn_femnist", "classes":62, "train_count":697932, "test_count":116323, "root_count":6944, "rounds":60, "epochs":2, "parameters":1690046},
 "eurosat": {"legacy_key":"eurosat", "model":"cnn_eurosat", "classes":10, "train_count":21600, "test_count":5400, "root_count":210, "rounds":60, "epochs":4, "parameters":1664970},
}
# Literal submitted-simulator defaults and declared protocol overrides, pinned
# from scientific-design-v1. Uncalibrated simulation values, not hardware facts.
DEFAULTS = {'data': {'dataset': 'mnist',
          'data_dir': 'REQUIRES_CONFIRMED_ABSOLUTE_WINDOWS_PATH',
          'partition': 'dirichlet',
          'dirichlet_alpha': 0.2,
          'server_val_fraction': 0.01,
          'min_samples_per_client': 10},
 'model': {'name': 'auto', 'num_classes': 0},
 'train': {'rounds': 100,
           'local_epochs': 2,
           'batch_size': 64,
           'lr': 0.01,
           'momentum': 0.9,
           'weight_decay': 0.0,
           'fedprox_mu': 0.0,
           'device': 'cuda',
           'eval_every': 5},
 'uav': {'enabled': True,
         'num_uavs': 30,
         'area_m': 1000.0,
         'alt_min_m': 50.0,
         'alt_max_m': 150.0,
         'v_max': 20.0,
         'a_max': 5.0,
         'gm_alpha': 0.85,
         'round_duration_s': 10.0,
         'high_mobility_fraction': 0.3,
         'high_mobility_speed_factor': 3.0,
         'battery_j': 500000.0,
         'kappa': 1e-27,
         'p0_hover_w': 80.0,
         'p1_speed_w': 0.15,
         'f_min_hz': 500000000.0,
         'f_max_hz': 2000000000.0,
         'cycles_per_sample': 2000000.0,
         'server_pos': [500.0, 500.0, 0.0],
         'bandwidth_hz': 5000000.0,
         'beta0_db': -50.0,
         'pathloss_alpha': 2.2,
         'noise_dbm_per_hz': -174.0,
         'p_tx_min_w': 0.05,
         'p_tx_max_w': 1.0,
         'bits_per_param': 32.0,
         'comm_deadline_s': 5.0,
         'comm_slack': 0.0,
         'e_min_j': 10000.0,
         'idle_energy_drain': True},
 'trust': {'alpha_beh': 0.7,
           'alpha_comp': 0.7,
           'alpha_comm': 0.7,
           'w_participation': 0.2,
           'w_commitment': 0.2,
           'w_gradient': 0.6,
           'weight_mode': 'learnable',
           'w_beh': 0.34,
           'w_comp': 0.33,
           'w_comm': 0.33,
           'learnable_lr': 0.5,
           'asymmetry': 3.0,
           'init_trust': 0.5,
           't_min': 0.3,
           'aggregation_gate': True,
           'round_consistency_gate': True,
           'k_max': 12,
           'staleness_beta': 0.0,
           'explore_slots': 1},
 'resources': {'enabled': True,
               'lambda1': 1.0,
               'lambda2': 1.0,
               'lambda3': 0.0001,
               'tau_max_s': 8.0,
               't_max_s': 5.0,
               'dual_step': 0.1,
               'max_iters': 50,
               'tol': 0.001,
               'compute_deadline': True,
               'compute_slack': 1.5},
 'attack': {'attack_type': 'none',
            'malicious_fraction': 0.0,
            'label_flip_prob': 1.0,
            'grad_scale_factor': 5.0,
            'random_update_std': 1.0,
            'misreport_factor': 2.0,
            'delay_slack_s': 0.5,
            'selective_prob': 0.4,
            'adaptive_trust_target': 0.6,
            'adaptive_probe_rounds': 5,
            'evasive_align': 0.9,
            'evasive_poison_scale': 1.0},
 'method': {'name': 'PER_GRID_METHOD',
            'krum_f': 12,
            'multikrum_m': 8,
            'trimmed_mean_beta': 0.1,
            'fedgreed_k': 5}}


def make_protocol(dataset, seed, condition="honest", purpose="scientific", device="cpu"):
    if dataset not in DATASETS or condition not in ("honest", "adaptive40"):
        raise ValueError("Only the named offline datasets and honest/adaptive40 conditions are admitted")
    if purpose not in ("scientific", "v6", "v5") or device not in ("cpu", "cuda"):
        raise ValueError("Unknown explicit purpose/device; no device fallback")
    if type(seed) is not int or seed < 0:
        raise ValueError("scenario seed must be a nonnegative typed integer")
    if purpose == "v6" and (condition != "honest" or seed != 99001):
        raise ValueError("V6 is the fixed honest seed99001 fixture")
    if purpose == "v5" and (dataset not in V5_SEEDS or seed not in V5_SEEDS[dataset] or device != "cuda"):
        raise ValueError("V5 admits only fixed EMNIST/EuroSAT seeds on CUDA")
    cfg = copy.deepcopy(DEFAULTS)
    spec = DATASET_SPECS[dataset]
    cfg["data"]["dataset"] = spec["legacy_key"]
    cfg["data"]["data_dir"] = "EXPLICIT_OFFLINE_PATH_AT_EXECUTION"
    cfg["model"] = {"name": spec["model"], "num_classes": spec["classes"]}
    cfg["train"].update(rounds=spec["rounds"], local_epochs=spec["epochs"], device=device)
    cfg["method"]["name"] = "METHOD_AT_EXECUTION"
    cfg["attack"].update(attack_type="adaptive_system" if condition == "adaptive40" else "none",
                          malicious_fraction=0.4 if condition == "adaptive40" else 0.0)
    if purpose == "v6":
        cfg["train"].update(rounds=2, local_epochs=1, batch_size=16, eval_every=1)
        cfg["uav"].update(num_uavs=4, high_mobility_fraction=0.5)
        cfg["trust"]["k_max"] = 2
    return {"schema":V5_SCHEMA if purpose == "v5" else SCHEMA, "rng_version":RNG_VERSION, "dataset":dataset,
            "scenario_seed":seed, "condition":condition, "purpose":purpose,
            "claim_eligible":False, "device":device, "scientific":cfg,
            "data_profile":V5_PROFILE if purpose == "v5" else ("full" if purpose == "scientific" else "root16_client16_test16"),
            "workers":0, "execution":"sequential", "observer_scope":"oracle_simulation"}


def validate_protocol(value):
    if not isinstance(value, dict):
        raise ValueError("Scientific protocol must be an object")
    expected = make_protocol(value.get("dataset"), value.get("scenario_seed"),
                             value.get("condition"), value.get("purpose"), value.get("device"))
    if canonical(value) != canonical(expected):
        raise ValueError("Scientific protocol differs from a strict named recipe")
    return copy.deepcopy(value)


def load_protocol(path):
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def native_config_dict(protocol, data_root, method="trustfl"):
    protocol = validate_protocol(protocol)
    if method not in METHODS:
        raise ValueError("Scientific core method not supported")
    root = Path(data_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Existing absolute offline dataset root required")
    result = copy.deepcopy(protocol["scientific"])
    result.update(name="scientific_"+protocol["dataset"]+"_"+protocol["condition"], seed=protocol["scenario_seed"])
    result["data"]["data_dir"] = str(root.resolve())
    result["method"]["name"] = LEGACY_METHODS[method]
    return result


def pairing_recipe(protocol, content_id):
    """No method, policy, active-threat toggle, or execution-device identity.

    A common content/recipe/seed defines potential external inputs. A run's
    separate resolved configuration identifies active threat and device.
    """
    p = validate_protocol(protocol)
    return {"version":"scientific_pairing_v1", "rng_version":RNG_VERSION,
            "dataset":p["dataset"], "content_id":content_id, "scenario_seed":p["scenario_seed"],
            "data_profile":"full" if p["purpose"] == "v5" else p["data_profile"], "root_fraction":0.01,
            "root_floor":50, "dirichlet_alpha":0.2, "minimum_client_samples":10,
            "partition_attempt_limit":100, "eurosat_outer_seed":42,
            "clients":p["scientific"]["uav"]["num_uavs"],
            "rounds":p["scientific"]["train"]["rounds"],
            "model":p["scientific"]["model"], "transforms":"preserved_deterministic_v1"}
