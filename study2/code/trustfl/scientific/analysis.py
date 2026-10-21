"""Prospective scenario-level analysis. No training or historical-data repair.

The exact test treats the supplied finite binary64 differences as exact rational
numbers. Equal-extreme tails are included with exact integer/rational arithmetic;
there is no data-dependent tolerance. Paired inputs do not prove null sign
symmetry. Bootstrap intervals are approximate marginal intervals, not power or
simultaneous-coverage guarantees.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import math
import os
from pathlib import Path
import re
import statistics


ANALYSIS_VERSION = "scientific-analysis-v2"
ENDPOINTS = ("failure_adjusted_accuracy_reliability_score",
             "pooled_honest_HM_retention",
             "late_unconditional_malicious_member_coefficient_share")
COMPARATORS = ("fedavg", "fltrust_declared_root_fallback",
               "fedgreed_algorithm1", "fd_distance_validation_local")
FAILURE_TABLE = {
    "complete": {"score": "observed_finite_final_accuracy", "requires":
                 ["complete_certificate", "final_evaluation_finite", "full_expected_round_sequence"]},
    "numerical_aggregate": {"score": 0.0, "requires": ["nonfinite_aggregate_recorded"]},
    "numerical_model": {"score": 0.0, "requires": ["nonfinite_model_recorded"]},
    "numerical_evaluation": {"score": 0.0, "requires": ["nonfinite_evaluation_recorded"]},
    "budget_oom": {"score": 0.0, "requires": ["oom_exception_recorded", "declared_resource_budget",
                                                 "same_declared_execution_device"]},
    "external_interruption": {"score": None, "requires": ["interruption_recorded"]},
    "not_started": {"score": None, "requires": []},
    "missing_corrupt_evidence": {"score": None, "requires": ["evidence_integrity_failure_recorded"]},
    "ambiguous_cause": {"score": None, "requires": []},
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def failure_adjusted_score(kind, observed_accuracy=None, evidence=None):
    """Return a score separately from observed accuracy and evidential status.

    Evidence values must be literal true; a populated string is insufficient.
    Unknown or insufficiently evidenced classifications remain unavailable.
    This rule never imputes missing mechanistic telemetry.
    """
    evidence = evidence or {}
    if kind not in FAILURE_TABLE:
        raise ValueError("unknown failure classification")
    missing = [key for key in FAILURE_TABLE[kind]["requires"] if evidence.get(key) is not True]
    measured = float(observed_accuracy) if _finite(observed_accuracy) and 0 <= observed_accuracy <= 1 else None
    score = FAILURE_TABLE[kind]["score"]
    if kind == "complete":
        score = measured
        if measured is None:
            missing.append("finite_final_accuracy_in_unit_interval")
    if missing:
        score = None
    return {"failure_classification": kind, "observed_final_accuracy": measured,
            "failure_adjusted_accuracy_reliability_score": score,
            "score_is_observed_accuracy": kind == "complete" and score is not None,
            "classification_evidence_missing": missing,
            "mechanistic_missing_is_never_zero_imputed": True}


def exact_sign_flip(differences):
    """Exhaustive two-sided absolute paired mean, exact binary64 ties, n <= 20."""
    values = tuple(differences)
    if not values or len(values) > 20 or not all(_finite(x) for x in values):
        raise ValueError("require 1..20 finite paired differences")
    ratios = [Fraction.from_float(float(x)) for x in values]
    # Binary64 ratios have power-of-two denominators, so the largest is common.
    denominator = max(x.denominator for x in ratios)
    integers = [x.numerator * (denominator // x.denominator) for x in ratios]
    observed = abs(sum(integers))
    extreme = sum(abs(sum(s * x for s, x in zip(signs, integers))) >= observed
                  for signs in itertools.product((-1, 1), repeat=len(integers)))
    total = 1 << len(integers)
    return {"n": len(values), "extreme": extreme, "enumerated": total,
            "p_value": extreme / total, "observed_mean": statistics.mean(values),
            "tie_rule": "exact_binary64_rational_equal_extreme_inclusive",
            "assumption": "independent scenarios and null sign symmetry/exchangeability"}


def fixed_holm(slots, expected_ids, alpha=0.05):
    """Keep every frozen family slot; null raw p is not a measured p of one."""
    expected = tuple(expected_ids)
    if len(expected) != len(set(expected)) or set(slots) != set(expected):
        raise ValueError("Holm slots must exactly equal frozen unique family IDs")
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    for value in slots.values():
        if value is not None and (not _finite(value) or not 0 <= value <= 1):
            raise ValueError("p values must be finite [0,1] or unavailable")
    order = sorted(expected, key=lambda name: (1.0 if slots[name] is None else slots[name], name))
    results, previous = {}, 0.0
    for position, name in enumerate(order):
        raw = slots[name]
        bookkeeping = 1.0 if raw is None else float(raw)
        adjusted = min(1.0, max(previous, (len(order) - position) * bookkeeping))
        previous = adjusted
        results[name] = {"raw_p": raw, "holm_bookkeeping_p": bookkeeping,
                         "available": raw is not None, "family_size": len(order),
                         "holm_adjusted_p": adjusted if raw is not None else None,
                         "reject": raw is not None and adjusted <= alpha,
                         "bookkeeping_only": raw is None}
    return {key: results[key] for key in expected}


def bootstrap_indices(n, resamples=50000, seed=51001):
    """Versioned NumPy PCG64 indices, fixed protocol defaults, explicit indices."""
    import numpy as np
    if type(n) is not int or n <= 0 or type(resamples) is not int or not 1 <= resamples <= 50000:
        raise ValueError("invalid bootstrap dimensions")
    return np.random.Generator(np.random.PCG64(seed)).integers(0, n, size=(resamples, n), dtype=np.int64)


def paired_bootstrap(differences, resamples=50000, seed=51001, return_indices=False):
    import numpy as np
    values = tuple(differences)
    if not values or not all(_finite(x) for x in values):
        raise ValueError("finite paired differences required")
    indices = bootstrap_indices(len(values), resamples, seed)
    means = np.mean(np.asarray(values, dtype=np.float64)[indices], axis=1)
    ci = np.quantile(means, [0.025, 0.975], method="linear")
    result = {"n": len(values), "mean_difference": statistics.mean(values),
              "difference_sd": statistics.stdev(values) if len(values) > 1 else None,
              "ci95_marginal_percentile": ci.tolist(), "resamples": resamples, "seed": seed,
              "rng": "numpy.PCG64", "numpy_version": np.__version__,
              "indices_dtype": "little_endian_int64_C_order",
              "indices_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes(order="C")).hexdigest(),
              "quantile_rule": "numpy_linear", "scope": "approximate_marginal_not_simultaneous"}
    if return_indices:
        result["indices"] = indices
    return result


def mechanistic_endpoints(records, memberships, expected_rounds):
    """Pooled full-window honest-HM retention and fixed late-window alpha share.

    All rounds must be valid complete telemetry. Numerators count client updates,
    not samples. Alpha is the actual submitted-delta coefficient after any norm
    scaling; the endpoint is not poisoning damage or theorem leakage.
    """
    from .records import validate_round
    empty = {ENDPOINTS[1]: None, ENDPOINTS[2]: None, "mechanistic_eligible": False}
    if type(expected_rounds) is not int or expected_rounds < 1:
        raise ValueError("positive expected_rounds required")
    if len(records) != expected_rounds or [r.get("round") for r in records] != list(range(expected_rounds)):
        return dict(empty, unavailable_reason="incomplete_or_unordered_window")
    errors = {r["round"]: validate_round(r) for r in records}
    errors = {t: e for t, e in errors.items() if e}
    if errors:
        return dict(empty, unavailable_reason="invalid_mechanistic_telemetry", errors=errors)
    malicious = set(memberships["malicious"])
    honest_hm = set(memberships["high_mobility"]) - malicious
    all_ids = set(records[0]["client_ids"])
    if not malicious <= all_ids or not honest_hm <= all_ids:
        raise ValueError("membership IDs absent from scenario")
    retention_numerator = retention_denominator = scheduled_denominator = arrival_denominator = 0
    full_shares, masses, mal_masses = [], [], []
    for record in records:
        accepted = set(record["positive_contribution"])
        retention_numerator += len(accepted & honest_hm)
        retention_denominator += len(set(record["executable"]) & honest_hm)
        scheduled_denominator += len(set(record["scheduled"]) & honest_hm)
        arrival_denominator += len(set(record["arrived"]) & honest_hm)
        applied = record["aggregate_applied"]
        mass = math.fsum(record["alpha"].values()) if applied else 0.0
        mal_mass = math.fsum(v for uid, v in record["alpha"].items() if int(uid) in malicious) if applied else 0.0
        full_shares.append(mal_mass / mass if mass > 0 else 0.0)
        masses.append(mass)
        mal_masses.append(mal_mass)
    late_start = expected_rounds // 2
    late_shares = full_shares[late_start:]
    positive_late = [share for share, mass in zip(late_shares, masses[late_start:]) if mass > 0]
    return {ENDPOINTS[1]: retention_numerator / retention_denominator if retention_denominator else None,
            ENDPOINTS[2]: statistics.mean(late_shares), "mechanistic_eligible": True,
            "retention_numerator": retention_numerator, "retention_denominator": retention_denominator,
            "retention_unavailable_reason": None if retention_denominator else "no_honest_HM_executable_opportunity",
            "scheduled_retention": retention_numerator / scheduled_denominator if scheduled_denominator else None,
            "arrival_retention": retention_numerator / arrival_denominator if arrival_denominator else None,
            "late_window": [late_start, expected_rounds - 1], "late_round_count": len(late_shares),
            "full_window_share": statistics.mean(full_shares),
            "late_conditional_positive_mass_share": statistics.mean(positive_late) if positive_late else None,
            "late_positive_mass_rounds": len(positive_late), "applied_alpha_mass": masses,
            "malicious_member_applied_alpha_mass": mal_masses,
            "interpretation": "malicious-member coefficient share, not poisoning damage or theoretical leakage"}


def confirmatory_slots(plan):
    if plan not in ("primary", "fallback"):
        raise ValueError("freeze primary or fallback before analysis")
    datasets = ("mnist", "emnist_byclass") if plan == "primary" else ("emnist_byclass",)
    seeds = tuple(range(41000, 41012 if plan == "primary" else 41010))
    return [{"id": ":".join((dataset, comparator, endpoint)), "dataset": dataset,
             "comparator": comparator, "endpoint": endpoint, "seeds": seeds,
             "policy": "common", "condition": "adaptive40"}
            for dataset in datasets for comparator in COMPARATORS for endpoint in ENDPOINTS]


def _summary_endpoint_verified(summary, endpoint):
    if not summary or summary.get("certificate_verified") is not True or summary.get("endpoint_eligible", {}).get(endpoint) is not True:
        return False
    if endpoint != ENDPOINTS[0]:
        return summary.get("attempt_status") == "complete" and summary.get("failure_classification") == "complete"
    kind = summary.get("failure_classification")
    if kind not in FAILURE_TABLE:
        return False
    if summary.get("attempt_status") != ("complete" if kind == "complete" else "failed"):
        return False
    assessed = failure_adjusted_score(kind, summary.get("observed_final_accuracy"), summary.get("classification_evidence"))
    expected = assessed[ENDPOINTS[0]]
    supplied = summary.get("endpoints", {}).get(endpoint)
    return expected is not None and _finite(supplied) and expected == supplied


def runtime_pairing_identity(environment, selected_device):
    """Stable actual runtime fields; exclude commands, paths and platform text."""
    from .records import identity
    if selected_device not in ("cpu", "cuda") or environment.get("selected_device") != selected_device:
        raise ValueError("saved actual device differs from attempt device")
    version_names = ("python", "numpy", "torch", "torchvision")
    if any(not isinstance(environment.get(name), str) or not environment[name] for name in version_names):
        raise ValueError("missing actual runtime version")
    flags = ("cuda_available", "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark")
    if any(type(environment.get(name)) is not bool for name in flags):
        raise ValueError("missing actual determinism/device flag")
    extra = ("torch_cuda_build", "cublas_workspace_config", "cuda_selected", "cudnn_version", "torch_num_threads", "torch_num_interop_threads")
    if any(name not in environment for name in extra):
        raise ValueError("missing actual runtime identity field")
    for name in ("torch_num_threads", "torch_num_interop_threads"):
        if type(environment[name]) is not int or environment[name] < 1:
            raise ValueError("invalid actual Torch thread count")
    if environment["torch_cuda_build"] is not None and not isinstance(environment["torch_cuda_build"], str):
        raise ValueError("invalid recorded CUDA build")
    if environment["cudnn_version"] is not None and type(environment["cudnn_version"]) is not int:
        raise ValueError("invalid recorded cuDNN version")
    gpu = environment["cuda_selected"]
    if selected_device == "cuda":
        if environment["cuda_available"] is not True or not environment["torch_cuda_build"]:
            raise ValueError("CUDA execution lacks actual CUDA runtime evidence")
        required = ("logical_index", "name", "total_memory_bytes", "compute_capability", "uuid")
        if not isinstance(gpu, dict) or any(key not in gpu for key in required):
            raise ValueError("missing selected GPU identity")
        if type(gpu["logical_index"]) is not int or gpu["logical_index"] < 0 or not isinstance(gpu["name"], str) or not gpu["name"]:
            raise ValueError("invalid selected GPU identity")
        if type(gpu["total_memory_bytes"]) is not int or gpu["total_memory_bytes"] <= 0 or not isinstance(gpu["compute_capability"], list) or len(gpu["compute_capability"]) != 2 or any(type(v) is not int or v < 0 for v in gpu["compute_capability"]):
            raise ValueError("invalid selected GPU properties")
        if gpu["uuid"] is not None and (not isinstance(gpu["uuid"], str) or not gpu["uuid"]):
            raise ValueError("invalid GPU UUID evidence")
        if environment["cublas_workspace_config"] not in (":4096:8", ":16:8"):
            raise ValueError("missing declared CUDA deterministic workspace")
    elif gpu is not None:
        raise ValueError("CPU run has inconsistent selected GPU metadata")
    stable = {name: environment[name] for name in version_names + flags + extra}
    stable["selected_device"] = selected_device
    return {"identity": identity(stable), "fields": stable,
            "scope": "actual_versions_device_gpu_properties_and_determinism_not_path_or_command"}


def artifact_pairing_identity(reference, metadata):
    """Compare consumed content, model tensors and numeric arrays, not containers."""
    from .records import identity
    if reference.get("bundle_id") != metadata.get("scenario_id") or reference.get("rng", {}).get("namespace") != metadata.get("scenario_pairing_identity"):
        raise ValueError("saved scenario reference identity mismatch")
    required_arrays = {"indices_logical_manifest": {"train_ids", "test_ids", "root_ids", "client_values", "client_offsets", "train_labels", "test_labels"},
                       "traces_logical_manifest": {"priority", "mobility", "channel", "attack_coins"}}
    if metadata.get("purpose") == "v5":
        required_arrays["indices_logical_manifest"] |= {"full_root_reserve", "full_client_values", "full_client_offsets", "excluded_root_ids", "excluded_client_values", "excluded_client_offsets"}
    for name in ("data_content_id", "model_tensor_hash"):
        if not isinstance(reference.get(name), str) or re.fullmatch(r"[0-9a-f]{64}", reference[name]) is None:
            raise ValueError("missing logical artifact identity: " + name)
    if reference["data_content_id"] != metadata.get("data_content_id"):
        raise ValueError("data content identity mismatch")
    for name, names in required_arrays.items():
        manifest = reference.get(name)
        if not isinstance(manifest, dict) or set(manifest) != names:
            raise ValueError("missing logical numeric array manifest: " + name)
        for array_name, entry in manifest.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str) or re.fullmatch(r"[<|][biuf][1248]", entry["dtype"]) is None:
                raise ValueError("invalid canonical array dtype: " + array_name)
            if not isinstance(entry.get("shape"), list) or not entry["shape"] or any(type(n) is not int or n < (0 if metadata.get("purpose") == "v5" else 1) for n in entry["shape"]):
                raise ValueError("invalid numeric array shape: " + array_name)
            if not isinstance(entry.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
                raise ValueError("missing numeric array content hash: " + array_name)
    fields = {name: reference[name] for name in ("data_content_id", "model_tensor_hash", *required_arrays)}
    return {"identity": identity(fields), "fields": fields,
            "scope": "saved_actual_logical_arrays_and_model_tensor_identity_not_npz_or_torch_container_bytes"}


def load_attempt_summary(path, expected_source_identity=None):
    """Derive eligibility from immutable saved files, never caller-authored flags.

    Failure certificates are verified too. Prospective engineering attempts with
    claim_eligible=false remain ineligible for scientific confirmation. Hashes
    establish evidence consistency, not remote attestation or historical origin.
    """
    from .records import PRODUCTION_CONTRACT, strict_load, validate_attempt
    result = validate_attempt(path, expected_source_identity=expected_source_identity)
    metadata = result["metadata"]
    if not isinstance(metadata, dict) or metadata.get("record_contract") != PRODUCTION_CONTRACT:
        raise ValueError("scientific summary requires saved production-contract attempt metadata")
    required = ("dataset", "method", "policy", "condition", "scenario_seed")
    if any(key not in metadata for key in required):
        raise ValueError("corrupt metadata cannot identify the frozen scientific cell")
    summary = {key: metadata[key] for key in required}
    if metadata.get("purpose") == "v5":
        summary["campaign_protocol_sha256"] = metadata.get("campaign_v5", {}).get("protocol_sha256")
        summary["scientific_cell_protocol_id"] = metadata.get("protocol_id")
    summary.update(attempt_path=str(path), scenario_pairing_identity=metadata.get("scenario_pairing_identity"),
                   certificate_verified=result["verified"], verification_errors=result["errors"],
                   claim_eligible=metadata.get("claim_eligible") is True,
                   endpoints=dict.fromkeys(ENDPOINTS), endpoint_eligible=dict.fromkeys(ENDPOINTS, False),
                   observed_final_accuracy=None, classification_evidence={})
    if not result["verified"]:
        summary.update(attempt_status="unavailable", failure_classification="missing_corrupt_evidence")
        return summary
    try:
        runtime = runtime_pairing_identity(strict_load(Path(path) / "environment.json"), metadata.get("device"))
        artifacts = artifact_pairing_identity(strict_load(Path(path) / "scenario-reference.json"), metadata)
        if result["complete"] or (Path(path) / "initial-evaluation.json").exists():
            initial = strict_load(Path(path) / "initial-evaluation.json")
            if initial.get("model_tensor_hash") != artifacts["fields"]["model_tensor_hash"]:
                raise ValueError("actually initialized model differs from scenario tensor identity")
        summary["runtime_pairing"] = runtime
        summary["runtime_pairing_identity"] = runtime["identity"]
        summary["artifact_pairing"] = artifacts
        summary["artifact_pairing_identity"] = artifacts["identity"]
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        summary.update(attempt_status="unavailable", failure_classification="missing_corrupt_evidence", certificate_verified=False)
        summary["verification_errors"].append("runtime_or_logical_input_identity:" + str(exc))
        return summary
    certificate = result["certificate"]
    summary["source_identity_sha256"] = certificate["source_identity_sha256"]
    summary["attempt_status"] = certificate["status"]
    summary["failure_classification"] = certificate["failure_classification"]
    if result["complete"]:
        summary["observed_final_accuracy"] = result["final_evaluation"]["test_accuracy"]
        summary["classification_evidence"] = {"complete_certificate": True, "final_evaluation_finite": True,
                                               "full_expected_round_sequence": True}
        metrics = mechanistic_endpoints(result["records"], metadata["memberships"], metadata["planned_rounds"])
        summary["mechanistic_details"] = metrics
        for endpoint in ENDPOINTS[1:]:
            summary["endpoints"][endpoint] = metrics[endpoint]
            summary["endpoint_eligible"][endpoint] = summary["claim_eligible"] and metrics[endpoint] is not None
    else:
        summary["classification_evidence"] = certificate["evidence"]
    score = failure_adjusted_score(summary["failure_classification"], summary["observed_final_accuracy"], summary["classification_evidence"])
    summary["endpoints"][ENDPOINTS[0]] = score[ENDPOINTS[0]]
    summary["endpoint_eligible"][ENDPOINTS[0]] = summary["claim_eligible"] and score[ENDPOINTS[0]] is not None
    return summary


def analyze_confirmatory(plan, attempt_paths, expected_source_identity=None):
    """Public analysis entry: read/hash-verify saved attempt paths itself.

    Do not pass user-authored summary dictionaries. Absent planned cells remain
    unavailable in the fixed family. If supplied, expected_source_identity binds
    source records to an independently preserved source manifest.
    """
    from .records import _file_sha
    summaries = [load_attempt_summary(path, expected_source_identity) for path in attempt_paths]
    result = _analyze_verified_summaries(plan, summaries)
    result["input_evidence"] = []
    for summary in summaries:
        item = {key: summary.get(key) for key in ("attempt_path", "dataset", "policy", "condition", "method",
                "scenario_seed", "scenario_pairing_identity", "source_identity_sha256", "runtime_pairing",
                "artifact_pairing", "attempt_status", "failure_classification", "claim_eligible",
                "certificate_verified", "verification_errors")}
        path = Path(summary["attempt_path"])
        item["terminal_certificate_hashes"] = {name: _file_sha(path / name) for name in ("completion.json", "failure.json") if (path / name).is_file()}
        result["input_evidence"].append(item)
    return result


def save_analysis(plan, attempt_paths, fresh_output, expected_source_identity=None):
    """Explicit future analysis: immutable results plus exact shared resample IDs.

    This does not launch runs. It refuses any existing output directory and saves
    the fixed 50,000 x 12/10 index matrix once even when contrasts are unavailable.
    """
    import numpy as np
    from .records import _file_sha, _safe_path, identity, payload_inventory, write_json_once
    output = _safe_path(fresh_output)
    if output.exists():
        raise FileExistsError(str(output))
    result = analyze_confirmatory(plan, attempt_paths, expected_source_identity)
    n = len(confirmatory_slots(plan)[0]["seeds"])
    indices = bootstrap_indices(n, 50000, 51001).astype("<i8", copy=False)
    raw_sha = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
    for slot in result["contrasts"].values():
        if slot["bootstrap"] is not None and slot["bootstrap"]["indices_sha256"] != raw_sha:
            raise ValueError("analysis bootstrap index identity differs from fixed saved matrix")
    output.mkdir(exist_ok=False)
    try:
        with (output / "bootstrap-indices.npy").open("xb") as stream:
            np.save(stream, indices, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        result["saved_bootstrap_indices"] = {"file": "bootstrap-indices.npy", "file_sha256": _file_sha(output / "bootstrap-indices.npy"),
            "logical_sha256": raw_sha, "shape": [50000, n], "dtype": "<i8", "seed": 51001,
            "rng": "numpy.PCG64", "numpy_version": np.__version__, "shared_across_frozen_contrasts": True}
        result["expected_source_identity_hash"] = identity(expected_source_identity) if expected_source_identity is not None else None
        write_json_once(output / "analysis.json", result)
        certificate = {"schema": "scientific-analysis-completion-v2", "status": "complete", "plan": plan,
                       "family_size": result["family_size"], "payload_inventory": payload_inventory(output),
                       "scientific_runs_launched": 0, "optimizer_steps": 0,
                       "scope": "saved_evidence_analysis; completion_does_not_make_unavailable_contrasts_available"}
        write_json_once(output / "analysis-completion.json", certificate)
    except BaseException as exc:
        write_json_once(output / "analysis-failure.json", {"status": "failed", "error": type(exc).__name__ + ": " + str(exc),
                                                         "no_retry_or_overwrite": True})
        raise
    return certificate


def _analyze_verified_summaries(plan, attempt_summaries):
    """Use frozen IDs, never available-case rescue of a confirmatory contrast.

    Internal helper for verified loaders and fixed fabricated-number fixtures.
    Each summary must include dataset, policy, condition, method, scenario_seed,
    scenario_pairing_identity, endpoints, endpoint_eligible, certificate_verified,
    attempt_status, failure_classification and classification_evidence. An accuracy
    score is recomputed from its classification and observed_final_accuracy. The
    caller must obtain certificate_verified by reading the saved evidence, not
    merely trusting a supplied boolean. A numeric remnant
    from an ineligible attempt is unavailable. Duplicate cells and absent pairing
    identity fail. Extra honest/native/EuroSAT summaries remain descriptive only.
    """
    indexed = {}
    for summary in attempt_summaries:
        key = tuple(summary[name] for name in ("dataset", "policy", "condition", "method", "scenario_seed"))
        if key in indexed:
            raise ValueError("duplicate scientific cell, no implicit replacement")
        indexed[key] = summary
    definitions, results, pvalues = confirmatory_slots(plan), {}, {}
    for definition in definitions:
        endpoint, differences, paired, missing = definition["endpoint"], [], [], []
        for seed in definition["seeds"]:
            prefix = (definition["dataset"], "common", "adaptive40")
            first = indexed.get(prefix + ("trustfl", seed))
            second = indexed.get(prefix + (definition["comparator"], seed))
            identity_names = ("scenario_pairing_identity", "source_identity_sha256", "runtime_pairing_identity", "artifact_pairing_identity")
            identity_ok = first and second and all(first.get(name) and first[name] == second.get(name) for name in identity_names)
            left = first.get("endpoints", {}).get(endpoint) if first else None
            right = second.get("endpoints", {}).get(endpoint) if second else None
            eligible = _summary_endpoint_verified(first, endpoint) and _summary_endpoint_verified(second, endpoint)
            if not identity_ok or not eligible or not _finite(left) or not _finite(right):
                missing.append(seed)
                paired.append({"scenario_seed": seed, "trustfl": left, "comparator": right,
                               "available": False, "pairing_identity_valid": bool(identity_ok)})
            else:
                differences.append(float(left) - float(right))
                paired.append({"scenario_seed": seed, "trustfl": left, "comparator": right,
                               "available": True, "pairing_identity_valid": True})
        available = not missing
        result = {"definition": definition, "available": available, "paired_values": paired,
                  "missing_planned_seeds": missing, "available_pairs": len(differences),
                  "available_pair_descriptive_mean": statistics.mean(differences) if differences else None,
                  "confirmatory_test": exact_sign_flip(differences) if available else None,
                  "bootstrap": paired_bootstrap(differences) if available else None,
                  "unavailable_rule": "whole_fixed_contrast_unavailable_no_available_case_confirmatory_rescue"}
        pvalues[definition["id"]] = result["confirmatory_test"]["p_value"] if available else None
        results[definition["id"]] = result
    holm = fixed_holm(pvalues, [slot["id"] for slot in definitions])
    for name, result in results.items():
        result["multiplicity"] = holm[name]
    return {"version": ANALYSIS_VERSION, "plan": plan, "family_size": len(definitions),
            "unit": "scenario_seed", "effect_direction": "TrustFL minus comparator",
            "contrasts": results, "power_established": False,
            "sign_symmetry_established_by_pairing": False}
