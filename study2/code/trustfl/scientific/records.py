"""Immutable prospective records and failure-aware completion certificates.

No simulator/model import occurs here. A durable invalid-round audit is separate
from accepted round records; guarded NaN/Inf is never recast as valid zero work.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


SCHEMA = "scientific-round-v2"
RECONSTRUCTION_ATOL = 1e-6
RECONSTRUCTION_RTOL = 1e-5
PRODUCTION_CONTRACT = "scientific-production-v2"
FIXTURE_CONTRACT = "scientific-fixture-v2"
QUERY_PURPOSES = ("test", "scheduled_root", "intrinsic_root_validation", "trustfl_learner", "shadow_attacker_feedback")
REQUIRED_TIMERS = ("selection", "resource_rule", "reference_consistency", "intrinsic_method_inclusive",
                   "intrinsic_root_training", "trust_update", "trustfl_learner_inclusive",
                   "shadow_attacker_feedback_inclusive", "scheduled_evaluation", "local_training",
                   "round_inclusive", "server_work_before_publication_excluding_local_and_scheduled_eval")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_valid(source):
    if not isinstance(source, dict) or not isinstance(source.get("git_head"), str) or not re.fullmatch(r"[0-9a-f]{40}", source["git_head"]):
        return False
    files = source.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for name, entry in files.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            return False
        if not isinstance(entry, dict) or not all(_sha(entry.get(key)) for key in ("sha256_raw", "sha256_lf_normalized")):
            return False
    return True


def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        return ["metadata_not_object"]
    errors = []
    required = ("record_contract", "planned_rounds", "protocol_id", "scenario_id", "source_identity", "claim_eligible", "client_ids", "initial_batteries_j")
    if any(key not in metadata for key in required):
        return ["missing_attempt_identity_metadata"]
    if metadata["record_contract"] not in (PRODUCTION_CONTRACT, FIXTURE_CONTRACT):
        errors.append("unknown_record_contract")
    if type(metadata["planned_rounds"]) is not int or metadata["planned_rounds"] < 1:
        errors.append("invalid_planned_rounds")
    if type(metadata["claim_eligible"]) is not bool:
        errors.append("invalid_claim_eligibility")
    ids = metadata["client_ids"]
    if not isinstance(ids, list) or not ids or any(type(uid) is not int or uid < 0 for uid in ids) or ids != sorted(set(ids)):
        return errors + ["invalid_metadata_client_ids"]
    batteries = metadata["initial_batteries_j"]
    if not isinstance(batteries, dict) or set(batteries) != {str(uid) for uid in ids} or not all(_nonnegative(v) for v in batteries.values()):
        errors.append("invalid_initial_batteries")
    if metadata["record_contract"] == FIXTURE_CONTRACT:
        if metadata["claim_eligible"] is not False:
            errors.append("fixture_cannot_be_claim_eligible")
        if not metadata["source_identity"] or not metadata["protocol_id"] or not metadata["scenario_id"]:
            errors.append("missing_fixture_identity")
        return errors
    if not _source_valid(metadata["source_identity"]):
        errors.append("invalid_production_source_identity")
    for name in ("protocol_id", "scenario_id", "scenario_pairing_identity"):
        if not _sha(metadata.get(name)):
            errors.append("invalid_identity:" + name)
    for name in ("dataset", "method", "policy", "condition"):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            errors.append("missing_scientific_cell:" + name)
    if type(metadata.get("scenario_seed")) is not int:
        errors.append("missing_scenario_seed")
    memberships = metadata.get("memberships")
    if not isinstance(memberships, dict) or not {"malicious", "high_mobility"} <= set(memberships):
        errors.append("missing_memberships")
    else:
        for name in ("malicious", "high_mobility"):
            members = memberships.get(name)
            if not isinstance(members, list) or any(type(x) is not int for x in members) or len(members) != len(set(members)) or not set(members) <= set(ids):
                errors.append("invalid_memberships:" + name)
        if metadata.get("malicious_ids") != memberships.get("malicious"):
            errors.append("malicious_membership_mismatch")
    return errors


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _nonnegative(value):
    return _finite(value) and value >= 0


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def identity(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_load(path):
    def reject(value):
        raise ValueError("nonstandard JSON constant: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject,
                      object_pairs_hook=unique)


def _safe_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute non-traversing output path required")
    for parent in (path, *path.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("output symlink/junction refused")
    return path


def write_json_once(path, value):
    """Publish a complete fsynced file by atomic no-replace hard-link creation.

    The output filesystem must support hard links (NTFS and APFS do). Failure
    refuses publication rather than falling back to overwrite-prone replace.
    """
    path = _safe_path(path)
    encoded = canonical_bytes(value)
    if path.exists():
        raise FileExistsError(str(path))
    fd, temporary = tempfile.mkstemp(prefix=".scientific-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(encoded)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(encoded).hexdigest()


def fresh_round(round_id, client_ids, initial_batteries=None):
    ids = sorted(client_ids)
    if type(round_id) is not int or round_id < 0 or len(ids) != len(set(ids)) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("nonnegative round and unique integer client IDs required")
    batteries = {str(k): v for k, v in (initial_batteries or {}).items()}
    return {"schema": SCHEMA, "round": round_id, "client_ids": ids,
            "scheduled": [], "executable": [], "participating": [], "arrived": [],
            "quarantined": {}, "usable": [], "positive_contribution": [], "exclusions": {},
            "alpha": {}, "q": {}, "root_term": {"coefficient": 0.0, "norm": 0.0, "fallback": False},
            "coefficient_round": round_id, "coefficient_status": "valid_no_candidates",
            "aggregate_applied": False, "aggregate_finite": None,
            "updates_finite": True, "post_model_finite": True,
            "reconstruction": {"residual_l2": 0.0, "scale": 0.0,
                               "bound": RECONSTRUCTION_ATOL, "passed": True},
            "energy": {str(uid): {"pre_j": batteries.get(str(uid)), "post_j": None,
                                  "modeled_j": 0.0, "depletion_j": 0.0,
                                  "compute_j": 0.0, "tx_j": 0.0, "mobility_j": 0.0}
                       for uid in ids},
            "evaluation_required": False, "evaluations": {}, "counters": {}, "timers": {},
            "adaptive": {}, "resources": {}}


def _uid_set(record, name, errors):
    values = record.get(name)
    if not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values) or len(values) != len(set(values)):
        errors.append("invalid_client_id_list:" + name)
        return set()
    return set(values)


def validate_round(record, metadata=None):
    """Return stable error codes; any nonempty result prevents completion."""
    errors = []
    if not isinstance(record, dict):
        return ["round_not_object"]
    if record.get("schema") != SCHEMA:
        errors.append("wrong_round_schema")
    t = record.get("round")
    if type(t) is not int or t < 0:
        errors.append("invalid_round_identifier")
    sets = {name: _uid_set(record, name, errors) for name in (
        "client_ids", "scheduled", "executable", "participating", "arrived", "usable", "positive_contribution")}
    chain = ("client_ids", "scheduled", "executable", "participating", "arrived", "usable", "positive_contribution")
    for parent, child in zip(chain, chain[1:]):
        if not sets[child] <= sets[parent]:
            errors.append(child + "_not_subset_of_" + parent)
    exclusions = record.get("exclusions")
    if not isinstance(exclusions, dict) or set(exclusions) != {str(x) for x in sets["scheduled"] - sets["executable"]}:
        errors.append("missing_exclusion_reasons")
    elif any(not isinstance(reason, str) or not reason for reason in exclusions.values()):
        errors.append("empty_exclusion_reason")
    quarantined = record.get("quarantined")
    if not isinstance(quarantined, dict):
        errors.append("missing_quarantine_map")
        quarantined = {}
    elif any(not isinstance(reason, str) or not reason for reason in quarantined.values()):
        errors.append("empty_quarantine_reason")
    if not set(quarantined) <= {str(x) for x in sets["arrived"]} or set(quarantined) & {str(x) for x in sets["usable"]}:
        errors.append("quarantine_state_mismatch")
    if {str(x) for x in sets["arrived"] - sets["usable"]} != set(quarantined):
        errors.append("unexplained_unusable_arrival")
    if record.get("coefficient_round") != t:
        errors.append("stale_coefficient_round")
    valid_maps = True
    for name in ("alpha", "q"):
        coefficients = record.get(name)
        if not isinstance(coefficients, dict):
            errors.append("missing_" + name)
            valid_maps = False
        elif set(coefficients) != {str(x) for x in sets["usable"]}:
            errors.append(name + "_candidate_mismatch")
            valid_maps = False
        elif not all(_nonnegative(v) for v in coefficients.values()):
            errors.append("nonfinite_or_negative_" + name)
            valid_maps = False
    root = record.get("root_term")
    if not isinstance(root, dict) or not _nonnegative(root.get("coefficient")) or not _nonnegative(root.get("norm")) or type(root.get("fallback")) is not bool:
        errors.append("invalid_root_term")
        root = {"coefficient": 0.0, "fallback": False}
    applied = record.get("aggregate_applied")
    if type(applied) is not bool:
        errors.append("missing_application_state")
    # This flag concerns usable candidates after quarantine. A nonfinite received
    # update may be quarantined without making the applied aggregate nonfinite.
    if record.get("updates_finite") is not True:
        errors.append("nonfinite_or_unknown_update")
    if record.get("post_model_finite") is not True:
        errors.append("nonfinite_or_unknown_post_model")
    if valid_maps:
        positive = {int(uid) for uid, value in record["alpha"].items() if value > 0}
        expected = positive if applied and record.get("post_model_finite") is True else set()
        if sets["positive_contribution"] != expected:
            errors.append("positive_contribution_mismatch")
        if not sets["usable"]:
            status = "valid_no_candidates"
        elif positive:
            status = "valid_positive"
        else:
            status = "valid_zero"
        if record.get("coefficient_status") != status:
            errors.append("coefficient_status_mismatch")
        if root.get("fallback") and (positive or root.get("coefficient") != 1.0):
            errors.append("root_fallback_coefficient_mismatch")
    if sets["usable"]:
        if record.get("aggregate_finite") is not True:
            errors.append("nonfinite_or_missing_aggregate")
        if applied is not True:
            errors.append("finite_candidate_aggregate_not_applied")
    elif record.get("aggregate_finite") is not None or applied or root.get("coefficient") != 0:
        errors.append("stale_empty_aggregate")
    recon = record.get("reconstruction")
    if not isinstance(recon, dict) or not all(_nonnegative(recon.get(k)) for k in ("residual_l2", "scale", "bound")):
        errors.append("missing_or_nonfinite_reconstruction")
    else:
        bound = RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * recon["scale"]
        # The fixed formula must be used exactly; tolerance is not widened after a failure.
        if recon["bound"] != bound:
            errors.append("reconstruction_bound_changed")
        if recon.get("passed") is not True or recon["residual_l2"] > bound:
            errors.append("aggregate_reconstruction_failed")
    energy = record.get("energy")
    if not isinstance(energy, dict) or set(energy) != {str(x) for x in sets["client_ids"]}:
        errors.append("incomplete_full_swarm_energy")
    else:
        for uid, values in energy.items():
            names = ("pre_j", "post_j", "modeled_j", "depletion_j", "compute_j", "tx_j", "mobility_j")
            if not isinstance(values, dict) or not all(_nonnegative(values.get(k)) for k in names):
                errors.append("invalid_energy:" + uid)
                continue
            close = lambda a, b: math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)
            if not close(values["pre_j"] - values["post_j"], values["depletion_j"]):
                errors.append("energy_depletion_mismatch:" + uid)
            if not close(values["depletion_j"], min(values["pre_j"], values["modeled_j"])):
                errors.append("energy_clipping_mismatch:" + uid)
            if not close(values["modeled_j"], math.fsum(values[k] for k in ("compute_j", "tx_j", "mobility_j"))):
                errors.append("energy_component_mismatch:" + uid)
    evaluations = record.get("evaluations")
    if not isinstance(evaluations, dict):
        errors.append("missing_evaluation_object")
    else:
        if record.get("evaluation_required") and not all(k in evaluations for k in ("test_accuracy", "test_loss")):
            errors.append("missing_required_evaluation")
        for name, value in evaluations.items():
            if not _finite(value):
                errors.append("nonfinite_evaluation:" + name)
            elif name.endswith("accuracy") and not 0 <= value <= 1:
                errors.append("evaluation_accuracy_out_of_range:" + name)
    for name in ("counters", "timers"):
        values = record.get(name)
        if not isinstance(values, dict) or not all(_nonnegative(value) for value in values.values()):
            errors.append("invalid_" + name)
    try:
        canonical_bytes(record)
    except (ValueError, TypeError):
        errors.append("nonfinite_or_non_json_record")
    if metadata is not None:
        if record.get("client_ids") != metadata.get("client_ids"):
            errors.append("round_client_identity_mismatch")
        if metadata.get("record_contract") == PRODUCTION_CONTRACT:
            errors.extend(_production_round_errors(record, metadata, sets))
    return sorted(set(errors))


def _production_round_errors(record, metadata, sets):
    errors = []
    resources = record.get("resources")
    required_resources = ("requested_frequency_hz", "requested_power_w", "realized_frequency_hz", "realized_power_w")
    if not isinstance(resources, dict) or set(resources) != {str(uid) for uid in sets["participating"]}:
        errors.append("production_resource_coverage")
    else:
        for uid, resource in resources.items():
            if not isinstance(resource, dict) or not all(_nonnegative(resource.get(key)) for key in required_resources):
                errors.append("production_resource_values:" + uid)
            elif not isinstance(resource.get("request_information_scope"), str) or not resource["request_information_scope"]:
                errors.append("production_resource_information_scope:" + uid)
    adaptive = record.get("adaptive")
    if not isinstance(adaptive, dict) or set(adaptive) != {str(uid) for uid in metadata["malicious_ids"]}:
        errors.append("production_preinvitation_adaptive_coverage")
    else:
        for uid, decision in adaptive.items():
            if not isinstance(decision, dict) or any(type(decision.get(key)) is not bool for key in ("prior_striking", "striking", "probe_active")):
                errors.append("production_adaptive_decision:" + uid)
            elif not _finite(decision.get("own_trust_observed")) or not isinstance(decision.get("available_information"), list) or not decision["available_information"]:
                errors.append("production_adaptive_information:" + uid)
    counters = record.get("counters", {})
    required = [purpose + "_" + suffix for purpose in QUERY_PURPOSES for suffix in ("queries", "examples", "batches")]
    required += ["total_query_" + suffix for suffix in ("queries", "examples", "batches")]
    required += ["client_sgd_steps", "client_examples", "client_batches", "root_sgd_steps", "root_training_examples", "root_training_batches"]
    if not isinstance(counters, dict) or any(type(counters.get(name)) is not int or counters[name] < 0 for name in required):
        errors.append("production_counter_coverage_or_type")
    else:
        for suffix in ("queries", "examples", "batches"):
            if counters["total_query_" + suffix] != sum(counters[purpose + "_" + suffix] for purpose in QUERY_PURPOSES):
                errors.append("production_query_total_mismatch:" + suffix)
    timers = record.get("timers", {})
    required_timers = REQUIRED_TIMERS + tuple("query_" + purpose for purpose in QUERY_PURPOSES)
    if not isinstance(timers, dict) or any(not _nonnegative(timers.get(name)) for name in required_timers):
        errors.append("production_timer_coverage")
    if not isinstance(record.get("timing_scope"), str) or not record["timing_scope"]:
        errors.append("production_timing_scope_missing")
    for name in ("method", "policy"):
        if record.get(name) != metadata.get(name):
            errors.append("production_" + name + "_changed")
    if not _sha(record.get("priority_sha256")):
        errors.append("production_priority_identity_missing")
    return errors


def _failure_audit_value(value):
    """Retain nonfinite nature explicitly in evidence, never coerce it to zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_value": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {str(k): _failure_audit_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_failure_audit_value(v) for v in value]
    return value


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_inventory(path):
    """Seal every durable auxiliary payload, including permutation/timing files."""
    path, inventory = Path(path), {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink() or (hasattr(file, "is_junction") and file.is_junction()):
            raise ValueError("symlink/junction inside immutable attempt")
        if file.is_file():
            name = file.relative_to(path).as_posix()
            if name in ("completion.json", "failure.json"):
                continue
            if file.name.startswith(".scientific-"):
                raise ValueError("uncommitted temporary payload at finalization")
            inventory[name] = {"sha256": _file_sha(file), "bytes": file.stat().st_size}
    return inventory


def _timing_sidecar_errors(path, round_ids, metadata):
    if metadata["record_contract"] != PRODUCTION_CONTRACT:
        return []
    errors = []
    for t in round_ids:
        name = "round-timing-%05d.json" % t
        try:
            value = strict_load(Path(path) / name)
            if value.get("round") != t or not all(_nonnegative(value.get(key)) for key in ("publication_s", "inclusive_wall_s")):
                errors.append("invalid_publication_timing:" + name)
            elif value["inclusive_wall_s"] < value["publication_s"]:
                errors.append("publication_exceeds_inclusive_timing:" + name)
        except (OSError, ValueError, TypeError, AttributeError):
            errors.append("missing_publication_timing:" + name)
    return errors


class AttemptWriter:
    """A new attempt only; there is no resume, overwrite or implicit retry API."""
    def __init__(self, path, metadata):
        self.path = _safe_path(path)
        errors = validate_metadata(metadata)
        if errors:
            raise ValueError("invalid attempt metadata: " + ",".join(errors))
        encoded = canonical_bytes(metadata)  # Validate before reserving a directory.
        self.path.mkdir(exist_ok=False)
        self.metadata = json.loads(encoded)
        self.metadata_sha256 = write_json_once(self.path / "attempt.json", self.metadata)
        self.source_sha256 = write_json_once(self.path / "source_identity.json", self.metadata["source_identity"])
        self.records = []
        self.round_hashes = {}
        self.terminal = False

    def write_round(self, record):
        if self.metadata.get("purpose") == "v5" and len(canonical_bytes(record)) > 1024**2:
            raise ValueError("v5 round telemetry exceeds one MiB; no truncation")
        if self.terminal:
            raise ValueError("terminal attempt is immutable")
        if record.get("round") != len(self.records) or len(self.records) >= self.metadata["planned_rounds"]:
            raise ValueError("duplicate, out-of-order or out-of-range round")
        errors = validate_round(record, self.metadata)
        if not errors and self.records:
            previous = self.records[-1]
            if record["client_ids"] != previous["client_ids"]:
                errors.append("client_identity_changed")
            for uid, values in record["energy"].items():
                if uid in previous["energy"] and values["pre_j"] != previous["energy"][uid]["post_j"]:
                    errors.append("battery_boundary_discontinuity:" + uid)
        if not errors and not self.records and {uid: values["pre_j"] for uid, values in record["energy"].items()} != self.metadata["initial_batteries_j"]:
            errors.append("initial_battery_boundary_mismatch")
        if errors:
            raise ValueError("invalid scientific round: " + ",".join(sorted(set(errors))))
        name = "round-%05d.json" % record["round"]
        digest = write_json_once(self.path / name, record)
        self.records.append(json.loads(canonical_bytes(record)))
        self.round_hashes[name] = digest
        return digest

    def record_failure_round(self, record):
        if self.terminal:
            raise ValueError("terminal attempt is immutable")
        audit = {"schema": "scientific-failed-round-audit-v2", "eligible_round": False,
                 "validation_errors": validate_round(record, self.metadata), "record": _failure_audit_value(record)}
        return write_json_once(self.path / ("failure-round-%05d.json" % len(self.records)), audit)

    def finalize(self, final_evaluation, final_model_finite=True):
        if self.terminal:
            raise ValueError("terminal attempt is immutable")
        if len(self.records) != self.metadata["planned_rounds"]:
            raise ValueError("incomplete expected round sequence")
        if any(self.path.glob("failure-round-*.json")) or (self.path / "failure.json").exists():
            raise ValueError("failure evidence precludes a clean-complete certificate")
        if final_model_finite is not True:
            raise ValueError("final model must be verified finite")
        if not isinstance(final_evaluation, dict) or not all(_finite(final_evaluation.get(k)) for k in ("test_accuracy", "test_loss")) or not 0 <= final_evaluation["test_accuracy"] <= 1:
            raise ValueError("required finite final evaluation missing or invalid")
        for name, expected in self.round_hashes.items():
            if hashlib.sha256((self.path / name).read_bytes()).hexdigest() != expected:
                raise ValueError("round evidence changed before completion")
        if hashlib.sha256((self.path / "attempt.json").read_bytes()).hexdigest() != self.metadata_sha256:
            raise ValueError("attempt provenance changed before completion")
        if _file_sha(self.path / "source_identity.json") != self.source_sha256:
            raise ValueError("source identity changed before completion")
        timing_errors = _timing_sidecar_errors(self.path, range(len(self.records)), self.metadata)
        if timing_errors:
            raise ValueError(";".join(timing_errors))
        evaluation_hash = write_json_once(self.path / "final-evaluation.json", final_evaluation)
        certificate = {"schema": "scientific-completion-v2", "status": "complete",
                       "complete": True, "failure_classification": "complete",
                       "planned_rounds": self.metadata["planned_rounds"],
                       "round_ids": [r["round"] for r in self.records],
                       "round_sha256": self.round_hashes, "attempt_sha256": self.metadata_sha256,
                       "source_identity_sha256": self.source_sha256,
                       "client_ids": self.metadata["client_ids"], "record_contract": self.metadata["record_contract"],
                       "protocol_id": self.metadata["protocol_id"], "scenario_id": self.metadata["scenario_id"],
                       "final_evaluation_sha256": evaluation_hash, "final_model_finite": True,
                       "initial_batteries_j": {uid: e["pre_j"] for uid, e in self.records[0]["energy"].items()},
                       "final_batteries_j": {uid: e["post_j"] for uid, e in self.records[-1]["energy"].items()},
                       "claim_eligible": self.metadata["claim_eligible"],
                       "metric_eligibility_requires_separate_endpoint_validation": True}
        certificate["payload_inventory"] = payload_inventory(self.path)
        write_json_once(self.path / "completion.json", certificate)
        self.terminal = True
        return certificate

    def fail(self, kind, evidence, message):
        from .analysis import failure_adjusted_score
        if self.terminal:
            raise ValueError("terminal attempt is immutable")
        if kind == "complete":
            raise ValueError("use finalize for completion")
        classification = failure_adjusted_score(kind, evidence=evidence)
        certificate = {"schema": "scientific-failure-v2", "status": "failed", "complete": False,
                       "claim_eligible": False, "round_ids": [r["round"] for r in self.records],
                       "round_sha256": self.round_hashes, "planned_rounds": self.metadata["planned_rounds"],
                       "attempt_sha256": self.metadata_sha256, "source_identity_sha256": self.source_sha256,
                       "client_ids": self.metadata["client_ids"], "record_contract": self.metadata["record_contract"],
                       "protocol_id": self.metadata["protocol_id"], "scenario_id": self.metadata["scenario_id"],
                       "initial_batteries_j": self.metadata["initial_batteries_j"],
                       "final_batteries_j": {uid: e["post_j"] for uid, e in self.records[-1]["energy"].items()} if self.records else None,
                       "evidence": _failure_audit_value(evidence),
                       "message": str(message), **classification}
        certificate["payload_inventory"] = payload_inventory(self.path)
        write_json_once(self.path / "failure.json", certificate)
        self.terminal = True
        return certificate


def reserve_attempt(path, metadata):
    return AttemptWriter(path, metadata)


def validate_attempt(path, expected_source_identity=None):
    """Verify complete OR failed terminal evidence without trusting caller flags.

    Byte/hash consistency is not a signature or remote attestation. Optionally
    bind the recorded source identity to an independently preserved manifest.
    """
    path, errors, records = Path(path), [], []
    metadata = certificate = None
    final = None
    try:
        metadata = strict_load(path / "attempt.json")
        errors.extend(validate_metadata(metadata))
        if errors:
            raise ValueError("invalid attempt metadata")
        has_complete, has_failure = (path / "completion.json").exists(), (path / "failure.json").exists()
        if has_complete == has_failure:
            raise ValueError("exactly one terminal certificate required")
        certificate = strict_load(path / ("completion.json" if has_complete else "failure.json"))
        status = "complete" if has_complete else "failed"
        if certificate.get("status") != status or certificate.get("complete") is not has_complete:
            errors.append("terminal_status_mismatch")
        if has_complete and any(path.glob("failure-round-*.json")):
            errors.append("failure_evidence_present")
        if has_complete and certificate.get("final_model_finite") is not True:
            errors.append("final_model_not_verified_finite")
        for name in ("planned_rounds", "client_ids", "record_contract", "protocol_id", "scenario_id", "initial_batteries_j"):
            if certificate.get(name) != metadata.get(name):
                errors.append("certificate_metadata_mismatch:" + name)
        expected_claim = metadata["claim_eligible"] if has_complete else False
        if certificate.get("claim_eligible") is not expected_claim:
            errors.append("certificate_claim_eligibility_mismatch")
        round_ids = certificate.get("round_ids")
        if not isinstance(round_ids, list) or any(type(t) is not int for t in round_ids) or round_ids != list(range(len(round_ids))) or len(round_ids) > metadata["planned_rounds"]:
            raise ValueError("invalid certificate round prefix")
        if has_complete and round_ids != list(range(metadata["planned_rounds"])):
            errors.append("incomplete_expected_round_sequence")
        expected_names = ["round-%05d.json" % t for t in round_ids]
        actual_names = sorted(p.name for p in path.glob("round-*.json") if re.fullmatch(r"round-[0-9]+\.json", p.name))
        if actual_names != expected_names:
            errors.append("round_file_sequence_mismatch")
        if set(certificate.get("round_sha256", {})) != set(expected_names):
            errors.append("certificate_round_hash_coverage_mismatch")
        previous = None
        for t, name in enumerate(expected_names):
            record = strict_load(path / name)
            errors.extend("round_%d:%s" % (t, e) for e in validate_round(record, metadata))
            if record["round"] != t:
                errors.append("round_identifier_mismatch")
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != certificate["round_sha256"].get(name):
                errors.append("round_hash_mismatch:" + name)
            if previous and any(record["energy"][uid]["pre_j"] != previous["energy"][uid]["post_j"] for uid in record["energy"]):
                errors.append("battery_boundary_discontinuity")
            if not previous and {uid: e["pre_j"] for uid, e in record["energy"].items()} != metadata["initial_batteries_j"]:
                errors.append("initial_battery_boundary_mismatch")
            previous = record
            records.append(record)
        final_batteries = {uid: e["post_j"] for uid, e in records[-1]["energy"].items()} if records else None
        if certificate.get("final_batteries_j") != final_batteries:
            errors.append("final_battery_certificate_mismatch")
        for name, field in (("attempt.json", "attempt_sha256"), ("source_identity.json", "source_identity_sha256")):
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != certificate.get(field):
                errors.append("certificate_hash_mismatch:" + name)
        source = strict_load(path / "source_identity.json")
        if source != metadata["source_identity"] or expected_source_identity is not None and source != expected_source_identity:
            errors.append("source_identity_mismatch")
        if payload_inventory(path) != certificate.get("payload_inventory"):
            errors.append("payload_inventory_mismatch")
        errors.extend(_timing_sidecar_errors(path, round_ids, metadata))
        if has_complete:
            final = strict_load(path / "final-evaluation.json")
            if not all(_finite(final.get(k)) for k in ("test_accuracy", "test_loss")) or not 0 <= final["test_accuracy"] <= 1:
                errors.append("invalid_final_evaluation")
            if _file_sha(path / "final-evaluation.json") != certificate.get("final_evaluation_sha256"):
                errors.append("final_evaluation_hash_mismatch")
            if certificate.get("failure_classification") != "complete":
                errors.append("complete_failure_classification_mismatch")
        else:
            from .analysis import failure_adjusted_score
            classification = failure_adjusted_score(certificate["failure_classification"], evidence=certificate["evidence"])
            if certificate["failure_classification"] == "complete" or any(certificate.get(name) != value for name, value in classification.items()):
                errors.append("failure_classification_mismatch")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append("missing_or_corrupt_completion_evidence:" + type(exc).__name__)
    verified = not errors
    complete = verified and certificate is not None and certificate.get("complete") is True
    return {"complete": complete, "verified": verified, "errors": sorted(set(errors)),
            "metadata": metadata, "certificate": certificate, "records": records,
            "final_evaluation": final, "verification": "saved_hash_bound_evidence_not_remote_attestation"}
