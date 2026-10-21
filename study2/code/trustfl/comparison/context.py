"""Common policy hooks; all scientific state remains in native objects."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .policy import executable_invitations, realize_resources, resource_request, validate_schedule
from .randomness import EventRandomness, canonical_bytes
from .records import append_round, fresh_round, validate_round


class CommonContext:
    def __init__(self, bundle, cfg, run_dir=None):
        if cfg.method.name not in ("trustfl", "fedavg"):
            raise ValueError("initial common policy allows TrustFL and FedAvg only")
        self.bundle, self.cfg = bundle, cfg
        self.rng = EventRandomness(bundle["rng"]["master_seed"], bundle["rng"]["namespace"])
        self.schedule = validate_schedule(bundle["schedule"], cfg.train.rounds,
                                          cfg.uav.num_uavs, cfg.trust.k_max)
        self.traces = bundle.get("traces", {})
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.device = getattr(cfg.train, "device", "cpu")
        self.round_idx = -1
        self.local_epoch = self.local_batch = 0
        self.current_client = -1
        self.record = None
        self._event_start = 0

    def begin_round(self, t, network):
        if t != self.round_idx + 1:
            raise ValueError("common rounds must be sequential with no restart")
        self.round_idx = t
        self.local_epoch = self.local_batch = 0
        self.record = fresh_round(t, network)
        self.record["public_resource_request"] = resource_request(self.cfg.uav)
        self.record["evaluation_required"] = (t % self.cfg.train.eval_every == 0 or
                                               t == self.cfg.train.rounds - 1)
        self._event_start = len(self.rng.ledger)

    def event_value(self, event, uid, value, **counters):
        digest = self.rng.digest(event, self.round_idx, uid, **counters)
        values = np.asarray(value)
        # dtype/shape and canonical payload are present, not a seed-only assertion.
        record = {"event": event, "uid": uid, "key_digest": digest,
                  "shape": list(values.shape), "dtype": str(values.dtype),
                  "value": values.tolist(),
                  "payload_sha256": hashlib.sha256(canonical_bytes(values.tolist())).hexdigest()}
        if self.record is not None:
            self.record["events"].append(record)
        return value

    def _potential(self, table, event, uid, distribution):
        generator = self.rng.numpy(event, self.round_idx, uid)
        generated = distribution(generator)
        saved = self.traces.get(table, {}).get(str(self.round_idx), {}).get(str(uid))
        # A production bundle is validated by scenario.py; direct hook fixtures
        # may omit frozen traces and exercise the same deterministic generator.
        value = generated if saved is None else saved
        return self.event_value(event, uid, value)

    def mobility_noise(self, uid, scale):
        value = self._potential("mobility_normal", "mobility_innovation", uid,
                                lambda g: g.normal(0, 1, size=3))
        return np.asarray(value, dtype=float) * scale

    def channel_fading(self, uid):
        return float(self._potential("channel_exponential", "channel_unit_exponential_fading", uid,
                                    lambda g: g.exponential(1.0)))

    def permutation(self, uid, epoch, n):
        event = "minibatch_permutation"
        generated = self.rng.numpy(event, self.round_idx, uid, epoch=epoch).permutation(n).tolist()
        saved = self.bundle.get("permutations", {}).get(str(self.round_idx), {}).get(str(uid), {}).get(str(epoch))
        value = generated if saved is None else saved
        if sorted(value) != list(range(n)):
            raise ValueError("invalid saved minibatch permutation")
        self.event_value(event, uid, value, epoch=epoch)
        return value

    def record_payload(self, event, uid, value, **counters):
        """Hash used high-dimensional bytes without bloating the frozen bundle."""
        actual_device = str(value.device) if hasattr(value, "device") else "cpu"
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        digest = self.rng.digest(event, self.round_idx, uid, **counters)
        record = {"event": event, "uid": uid, "key_digest": digest,
                  "shape": list(array.shape), "dtype": str(array.dtype),
                  "payload_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                  "device": actual_device}
        if self.record is not None:
            self.record["events"].append(record)

    def attack_uniform(self, event, uid, *, size=None, **counters):
        value = self.rng.numpy(event, self.round_idx, uid, **counters).random(size)
        if event in ("attack_participation", "attack_delay_branch") and size is None:
            saved = self.traces.get("attack_coins", {}).get(str(self.round_idx), {}).get(str(uid), {}).get(event)
            if saved is not None:
                value = float(saved)
        return self.event_value(event, uid, value, **counters)

    def invite(self, network):
        scheduled = list(self.schedule[self.round_idx])
        executable, exclusions = executable_invitations(scheduled, network,
                                                         enabled=self.cfg.uav.enabled)
        self.record.update(scheduled=scheduled, executable=executable, exclusions=exclusions)
        return executable

    def resources(self, uav):
        values = realize_resources(resource_request(self.cfg.uav), uav, self.cfg.uav)
        self.record["clients"].setdefault(str(uav.uid), {}).update(resources=values)
        self.record["clients"][str(uav.uid)]["resources_status"] = "executed"
        return values["realized_frequency_hz"], values["realized_power_w"]

    def energy_components(self, uid, compute, tx, mobility, *, idle=False):
        self.record["energy"][str(uid)]["components"].append(
            {"compute_j": float(compute), "tx_j": float(tx),
             "mobility_j": float(mobility), "idle": bool(idle)})

    def drain(self, uid, before, after, modeled):
        energy = self.record["energy"][str(uid)]
        energy["modeled_j"] += float(modeled)
        energy["depletion_j"] += float(before - after)

    def coefficients(self, info, precursor):
        coefficients = info.get("coefficients")
        self.record.update(coefficient_round=self.round_idx, coefficients=coefficients,
                           precursor_coefficients={str(k): float(v) for k, v in precursor.items()},
                           coefficient_dtype=info.get("coefficient_dtype"),
                           coefficient_mass=info.get("coefficient_mass"),
                           coefficient_denominator=info.get("coefficient_denominator"),
                           coefficient_clamp_min=info.get("coefficient_clamp_min"))
        if coefficients is None:
            self.record["coefficient_status"] = "missing"
        elif not all(np.isfinite(v) and v >= 0 for v in coefficients.values()):
            self.record["coefficient_status"] = "nonfinite_coefficients"
        elif any(v > 0 for v in coefficients.values()):
            self.record["coefficient_status"] = "valid_positive_coefficients"
        else:
            self.record["coefficient_status"] = "valid_zero_coefficients"
        self.record["positive_coefficient_candidate_ids"] = sorted(
            int(k) for k, v in (coefficients or {}).items() if np.isfinite(v) and v > 0)

    def finish_round(self, metrics, network):
        self.record.update({key: value for key, value in metrics.items()
                            if key in ("test_acc", "test_loss", "val_acc", "val_loss", "agg_time_s")})
        for u in network.uavs:
            self.record["energy"][str(u.uid)].update(post_j=float(u.energy_j), alive_post=bool(u.alive))
        self.record["used_key_registry"] = self.rng.ledger[self._event_start:]
        errors = validate_round(self.record)
        self.record["errors"] = errors
        if self.run_dir is not None:
            append_round(self.run_dir / "common_rounds.jsonl", self.record)
        if errors:
            raise RuntimeError("invalid common round: " + "; ".join(errors))
        return self.record
