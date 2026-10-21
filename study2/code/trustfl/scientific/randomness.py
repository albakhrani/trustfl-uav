"""Compact, versioned event PRF registry for prospective scientific scenarios.

No potential per-sample JSON inventories. Existing deterministic transforms
consume no PRF events. Each round retains only bounded seed-collision state;
public summaries contain domain counts and order-independent digest sums.
"""
from __future__ import annotations
import hashlib
import hmac
from ..comparison.randomness import EventRandomness as EngineeringRandomness, canonical_bytes
from .protocol import RNG_VERSION

DOMAINS = (
    "initialization", "root_and_client_partition", "memberships",
    "physical_initialization", "invitation_schedule", "mobility_innovation",
    "channel_unit_exponential_fading", "attack_participation", "attack_delay_branch",
    "minibatch_permutation", "model_forward_stochastic_operations",
    "server_validation_and_evaluation", "per_sample_transform", "attack_label_mask",
    "attack_random_delta", "uav_initial_state_and_memberships",
)
MAX_DISTINCT_ROUND_EVENTS = 100000


class CompactEventRandomness(EngineeringRandomness):
    def __init__(self, master_seed, namespace):
        super().__init__(master_seed, namespace)
        self._summary = {}
        self._round = -1

    def key(self, event, round_idx=-1, uid=-1, **counters):
        if event not in DOMAINS:
            raise ValueError("Unregistered scientific RNG domain: " + str(event))
        value = super().key(event, round_idx, uid, **counters)
        value["version"] = RNG_VERSION
        return value

    def begin_round(self, round_idx):
        if type(round_idx) is not int or round_idx < -1:
            raise ValueError("Invalid registry round")
        self._round = round_idx
        self._summary = {}
        self._keys.clear()
        self._seeds.clear()
        self.ledger.clear()

    def digest(self, event, round_idx=-1, uid=-1, **counters):
        encoded = canonical_bytes(self.key(event, round_idx, uid, **counters))
        raw = hmac.new(self._master, encoded, hashlib.sha256).digest()
        digest = raw.hex()
        seed = int.from_bytes(raw[:8], "big") & ((1 << 63) - 1)
        previous = self._seeds.get(seed)
        if previous is not None and previous != digest:
            raise RuntimeError("scientific_rng_v1 reduced seed collision within registry round")
        if digest not in self._keys:
            if len(self._keys) >= MAX_DISTINCT_ROUND_EVENTS:
                raise RuntimeError("Scientific round RNG registry bound exceeded; no unbounded logging")
            self._keys[digest] = encoded
            self._seeds[seed] = digest
        elif self._keys[digest] != encoded:
            raise RuntimeError("scientific_rng_v1 full digest collision")
        summary = self._summary.setdefault(event, {"requests":0, "digest_sum_mod_2_256":0})
        summary["requests"] += 1
        summary["digest_sum_mod_2_256"] = (summary["digest_sum_mod_2_256"] + int.from_bytes(raw,"big")) % (1 << 256)
        return digest

    def round_summary(self):
        return {"version":RNG_VERSION, "round":self._round,
                "collision_check_scope":"distinct requested keys in this registry round",
                "distinct_keys":len(self._keys),
                "domains":{k:{"requests":v["requests"],
                              "digest_sum_mod_2_256":f'{v["digest_sum_mod_2_256"]:064x}'}
                           for k,v in sorted(self._summary.items())}}


EventRandomness = CompactEventRandomness


def domain_registry():
    return {"version":RNG_VERSION, "domains":list(DOMAINS),
            "derivation":"HMAC-SHA256; inherited master prefix TrustFLCommonPolicy.master.v1:",
            "canonical_encoding":"UTF-8 sorted-key compact finite JSON",
            "torch_seed":"first8bytes big-endian masked63bits",
            "numpy_seed":"PCG64(SeedSequence(eight big-endian uint32 digest words))",
            "deterministic_transforms":"No per-sample RNG draw; content/index/transform identities saved",
            "round_registry_limit":MAX_DISTINCT_ROUND_EVENTS,
            "seed_collision_scope":"per round; not a global proof across all potential keys",
            "unused_keys":"derivable; not materialized"}
