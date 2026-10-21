"""Strict common-only records. This module never imports training code."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import tempfile


def json_safe(value):
    """Nonfinite values are explicit null; callers retain failure status/reason."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def reserve_directory(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute non-traversing output path required")
    for parent in (path, *path.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("output symlink/junction refused")
    # Parent creation belongs to the runner's protected-root preflight.
    path.mkdir(exist_ok=False)
    return path


def write_json(path, value):
    path = Path(path)
    encoded = json.dumps(json_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".cp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            out.write(encoded)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_round(path, value):
    with open(path, "a", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(json_safe(value), sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def fresh_round(t, network):
    return {"schema": "cp_round_v1", "round": t, "scheduled": [], "executable": [],
            "exclusions": {}, "clients": {}, "adaptive": {}, "events": [],
            "physical_arrivals": [], "quarantined": {}, "usable_candidates": [],
            "coefficients": {}, "coefficient_status": "not_called_no_candidates",
            "coefficient_round": t, "coefficient_dtype": None, "coefficient_mass": 0.0,
            "precursor_coefficients": {}, "aggregate_available": False,
            "aggregate_finite": None, "aggregate_application_attempted": False,
            "aggregate_application_completed": False, "post_model_finite": True,
            "model_changed": False, "positive_coefficient_candidate_ids": [],
            "accepted_valid_ids": [], "guarded_events": [], "errors": [],
            "energy": {str(u.uid): {"pre_j": float(u.energy_j), "post_j": None,
                                     "modeled_j": 0.0, "depletion_j": 0.0,
                                     "components": [], "alive_pre": bool(u.alive)}
                       for u in network.uavs},
            "query_counts": {"root_validation": 0, "test_evaluation": 0}}


def validate_round(record):
    errors = list(record.get("errors", []))
    t = record.get("round")
    scheduled, executable = record.get("scheduled", []), record.get("executable", [])
    if (scheduled != sorted(set(scheduled)) or executable != sorted(set(executable)) or
            not set(executable) <= set(scheduled)):
        errors.append("invalid_scheduled_executable_sets")
    if set(record.get("exclusions", {})) != {str(j) for j in set(scheduled) - set(executable)}:
        errors.append("missing_exclusion_reasons")
    arrivals = record.get("physical_arrivals", [])
    if not set(arrivals) <= set(executable):
        errors.append("arrival_without_executable_invitation")
    candidates = record.get("usable_candidates", [])
    if not set(candidates) <= set(arrivals):
        errors.append("candidate_without_physical_arrival")
    if set(map(str, candidates)) & set(record.get("quarantined", {})):
        errors.append("quarantined_candidate")
    coefficients = record.get("coefficients")
    if record.get("coefficient_round") != t:
        errors.append("stale_coefficient_round")
    if not isinstance(coefficients, dict):
        errors.append("missing_coefficients")
    elif set(coefficients) != {str(j) for j in candidates}:
        errors.append("coefficient_candidate_mismatch")
    elif any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
             for v in coefficients.values()):
        errors.append("nonfinite_or_negative_coefficients")
    else:
        positive = sorted(int(k) for k, v in coefficients.items() if v > 0)
        if positive != record.get("positive_coefficient_candidate_ids", []):
            errors.append("positive_coefficient_ids_mismatch")
        expected_accepted = positive if (record.get("aggregate_application_completed") is True
                                        and record.get("post_model_finite") is True) else []
        if record.get("accepted_valid_ids", []) != expected_accepted:
            errors.append("accepted_valid_ids_mismatch")
        if candidates and record.get("coefficient_status") != (
                "valid_positive_coefficients" if positive else "valid_zero_coefficients"):
            errors.append("coefficient_status_value_mismatch")
        mass = record.get("coefficient_mass")
        # Actual mass is a float32 tensor reduction; retain that rounding.
        if not isinstance(mass, (int, float)) or not math.isfinite(mass) or not math.isclose(
                mass, sum(coefficients.values()), rel_tol=1e-6, abs_tol=0.0):
            errors.append("coefficient_mass_mismatch")
    if record.get("aggregate_application_completed") and not (
            record.get("aggregate_application_attempted") and record.get("aggregate_finite") is True):
        errors.append("invalid_application_provenance")
    if candidates:
        if record.get("aggregate_available") is not True or type(record.get("aggregate_finite")) is not bool:
            errors.append("missing_aggregate_provenance")
        elif record["aggregate_finite"] and not (record.get("aggregate_application_attempted") is True
                                                and record.get("aggregate_application_completed") is True):
            errors.append("finite_aggregate_not_applied")
        elif not record["aggregate_finite"] and (record.get("aggregate_application_attempted") or
                record.get("aggregate_application_completed") or
                "nonfinite_aggregate" not in record.get("guarded_events", [])):
            errors.append("nonfinite_aggregate_provenance_mismatch")
    elif record.get("aggregate_available") or record.get("aggregate_application_attempted") or record.get(
            "aggregate_application_completed") or record.get("model_changed"):
        errors.append("stale_empty_aggregate_provenance")
    if not candidates and record.get("coefficient_status") != "not_called_no_candidates":
        errors.append("invalid_empty_coefficient_status")
    if candidates and record.get("coefficient_status") not in (
            "valid_zero_coefficients", "valid_positive_coefficients"):
        errors.append("invalid_candidate_coefficient_status")
    if record.get("post_model_finite") is not True:
        errors.append("nonfinite_model_after_application")
    for name in ("test_acc", "test_loss", "val_acc", "val_loss"):
        if record.get("evaluation_required") and name not in record:
            errors.append("missing_evaluation:" + name)
        if name in record and (not isinstance(record[name], (int, float)) or
                               not math.isfinite(record[name])):
            errors.append("nonfinite_evaluation:" + name)
    for uid, energy in record.get("energy", {}).items():
        pre, post, modeled, depletion = (energy.get(k) for k in (
            "pre_j", "post_j", "modeled_j", "depletion_j"))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                   for v in (pre, post, modeled, depletion)):
            errors.append("invalid_energy:" + uid)
        elif not math.isclose(pre - post, depletion, rel_tol=1e-10, abs_tol=1e-8):
            errors.append("energy_depletion_mismatch:" + uid)
        elif not math.isclose(depletion, min(pre, modeled), rel_tol=1e-10, abs_tol=1e-8):
            errors.append("modeled_energy_mismatch:" + uid)
        component_total = 0.0
        for components in energy.get("components", []):
            values = [components.get(k) for k in ("compute_j", "tx_j", "mobility_j")]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
                errors.append("invalid_energy_components:" + uid)
            else:
                component_total += sum(values)
        if isinstance(modeled, (int, float)) and not math.isclose(
                component_total, modeled, rel_tol=1e-10, abs_tol=1e-8):
            errors.append("energy_components_mismatch:" + uid)
    return sorted(set(errors))


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _loads_json(text):
    # Python's default decoder permits nonstandard NaN/Infinity tokens.
    def reject_constant(value):
        raise ValueError("nonstandard JSON constant: " + value)
    return json.loads(text, parse_constant=reject_constant)


def _load_json(path):
    return _loads_json(Path(path).read_text(encoding="utf-8"))


def _validate_provenance(path, status, errors):
    """Check linked saved identities, without importing a model or current source."""
    try:
        reference = _load_json(path / "scenario_reference.json")
        source = _load_json(path / "source_identity.json")
        environment = _load_json(path / "environment.json")
        if not environment.get("executable") or not environment.get("cwd") or not environment.get("command"):
            errors.append("missing_execution_environment_provenance")
        directory = Path(reference["path"])
        if not directory.is_absolute() or directory.is_symlink():
            raise ValueError("invalid scenario reference path")
        bundle = _load_json(directory / "scenario.json")
        bundle_id = bundle.pop("bundle_id")
        if _identity(bundle) != bundle_id:
            errors.append("scenario_manifest_hash_mismatch")
        if reference.get("bundle_id") != bundle_id or status.get("scenario_bundle_id") != bundle_id:
            errors.append("scenario_reference_identity_mismatch")
        if status.get("protocol_hash") != bundle.get("protocol_hash") or _identity(
                bundle.get("protocol")) != bundle.get("protocol_hash"):
            errors.append("protocol_identity_mismatch")
        if reference.get("payloads") != bundle.get("payloads") or reference.get("rng") != bundle.get("rng"):
            errors.append("scenario_reference_payload_mismatch")
        for name, expected_hash in bundle["payloads"].items():
            file = directory / name
            if Path(name).name != name or file.is_symlink():
                raise ValueError("invalid scenario payload path")
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected_hash:
                errors.append("scenario_payload_hash_mismatch:" + name)
        scenario_source = _load_json(directory / "source_identity.json")
        if not source.get("files") or not source.get("git_head"):
            errors.append("missing_source_identity")
        elif source["files"] != scenario_source.get("files") or source["git_head"] != scenario_source.get("git_head"):
            errors.append("run_scenario_source_identity_mismatch")
        if bundle.get("seed_registry", {}).get("source_identity_hash") != _identity(source.get("files")):
            errors.append("seed_registry_source_identity_mismatch")
        if status.get("expected_rounds") != bundle["resolved_config"]["train"]["rounds"]:
            errors.append("scenario_round_count_mismatch")
        initial = {str(u["uid"]): u["energy_j"] for u in bundle["initial_uav"]}
        return bundle, initial
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        errors.append("missing_or_invalid_source_scenario_provenance")
        return None, None


def validate_attempt(path, require_terminal=True):
    path = Path(path)
    errors, records = [], []
    try:
        status = _load_json(path / "status.json")
        if not isinstance(status, dict):
            raise ValueError("invalid status object")
    except (OSError, ValueError):
        return {"complete": False, "errors": ["missing_or_invalid_status"]}
    try:
        records = [_loads_json(line) for line in (path / "common_rounds.jsonl").read_text().splitlines()]
        if any(not isinstance(record, dict) for record in records):
            records = []
            raise ValueError("invalid round object")
    except (OSError, ValueError):
        errors.append("missing_or_invalid_round_records")
    expected = status.get("expected_rounds")
    if type(expected) is not int or [r.get("round") for r in records] != list(range(expected)):
        errors.append("incomplete_or_duplicate_round_sequence")
    if status.get("completed_rounds") != len(records):
        errors.append("completed_round_count_mismatch")
    bundle, scenario_initial = _validate_provenance(path, status, errors)
    try:
        initial = _load_json(path / "initial_battery.json")
        if not initial or not all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                  for v in initial.values()):
            raise ValueError("invalid initial battery")
        if scenario_initial is not None and initial != scenario_initial:
            errors.append("scenario_initial_battery_mismatch")
    except (OSError, ValueError, AttributeError):
        initial = {}
        errors.append("missing_or_invalid_initial_battery")
    preceding_energy = dict(initial)
    summed_depletion = dict.fromkeys(initial, 0.0)
    preceding_alive = None
    for record in records:
        try:
            errors.extend(f"round{record.get('round')}:{e}" for e in validate_round(record))
        except (KeyError, TypeError, ValueError, AttributeError):
            errors.append("malformed_round_schema")
            continue
        if record.get("evaluation_required") is not True:
            errors.append("missing_required_fixture_evaluation")
        energy = record.get("energy", {})
        if set(energy) != set(initial):
            errors.append("missing_or_extra_energy_clients")
        if bundle is not None:
            t = record.get("round")
            if type(t) is not int or not 0 <= t < len(bundle["schedule"]) or record.get("scheduled") != bundle["schedule"][t]:
                errors.append("recorded_schedule_mismatch")
        for uid in set(energy) & set(initial):
            values = energy[uid]
            pre, post, depletion = (values.get(key) for key in ("pre_j", "post_j", "depletion_j"))
            if not all(type(v) in (int, float) and math.isfinite(v) for v in (pre, post, depletion)):
                continue
            if not math.isclose(pre, preceding_energy[uid], rel_tol=1e-10, abs_tol=1e-8):
                errors.append("battery_round_continuity_mismatch:" + uid)
            if preceding_alive is not None and values.get("alive_pre") != preceding_alive.get(uid):
                errors.append("alive_round_continuity_mismatch:" + uid)
            if values.get("alive_pre") is False and values.get("components"):
                errors.append("dead_client_energy_charge:" + uid)
            preceding_energy[uid] = post
            summed_depletion[uid] += depletion
        preceding_alive = {uid: values.get("alive_post") for uid, values in energy.items()}
    for uid in initial:
        if not math.isclose(initial[uid] - preceding_energy[uid], summed_depletion[uid],
                            rel_tol=1e-10, abs_tol=1e-8):
            errors.append("battery_interval_reconciliation_mismatch:" + uid)
    if require_terminal and status.get("status") != "complete":
        errors.append("nonterminal_or_unsuccessful_status:" + str(status.get("status")))
    if not require_terminal and status.get("status") not in ("running", "complete"):
        errors.append("invalid_preterminal_status:" + str(status.get("status")))
    try:
        final = _load_json(path / "final_evaluation.json")
        for name in ("test_acc", "test_loss", "val_acc", "val_loss"):
            if type(final.get(name)) not in (int, float) or not math.isfinite(final[name]):
                errors.append("missing_or_nonfinite_final:" + name)
            elif not records or final[name] != records[-1].get(name):
                errors.append("final_evaluation_state_mismatch:" + name)
        if final.get("round") != expected - 1:
            errors.append("final_evaluation_round_mismatch")
    except (OSError, ValueError, TypeError):
        errors.append("missing_or_invalid_final_evaluation")
    guarded = [e for r in records for e in r.get("guarded_events", [])]
    return {"complete": not errors, "errors": sorted(set(errors)), "guarded_events": guarded,
            "engineering_pass": not errors and not guarded, "scientific_claim_eligible": False,
            "observed_rounds": len(records), "expected_rounds": expected}
