"""Versioned event-indexed RNGs; no process-dependent hash or shared cursor."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import json
import random
import struct

import numpy as np

VERSION = "cp_rng_v1"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class EventRandomness:
    """Each key creates a fresh generator, so a skipped key shifts no other key.

    The ledger records keys actually requested, not hypothetical participation.
    Full digests and reduced Torch-seed collisions are checked per instance.
    """
    def __init__(self, master_seed: int, namespace: str):
        if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("scenario namespace is required")
        self.master_seed, self.namespace = master_seed, namespace
        self._master = hashlib.sha256(
            ("TrustFLCommonPolicy.master.v1:" + str(master_seed)).encode("utf-8")).digest()
        self._seeds = {}
        self._keys = {}
        self.ledger = []

    def key(self, event, round_idx=-1, uid=-1, **counters):
        if not isinstance(event, str) or not event:
            raise ValueError("event type is required")
        key = {"version": VERSION, "scenario_namespace_id": self.namespace,
               "round": round_idx, "client": uid, "event_type": event}
        allowed = {"epoch", "batch", "sample", "occurrence", "operation", "block"}
        if set(counters) - allowed:
            raise ValueError("unsupported event subkey")
        for name, value in {"round": round_idx, "client": uid, **counters}.items():
            if name == "operation":
                if not isinstance(value, str):
                    raise TypeError("operation must be a string")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a typed integer")
        key.update(counters)
        return key

    def digest(self, event, round_idx=-1, uid=-1, **counters):
        key = self.key(event, round_idx, uid, **counters)
        encoded = canonical_bytes(key)
        digest = hmac.new(self._master, encoded, hashlib.sha256).digest()
        hex_digest = digest.hex()
        reduced = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        if reduced in self._seeds and self._seeds[reduced] != hex_digest:
            raise RuntimeError("cp_rng_v1 reduced seed collision")
        if hex_digest in self._keys and self._keys[hex_digest] != encoded:
            raise RuntimeError("cp_rng_v1 full digest collision")
        self._seeds[reduced], self._keys[hex_digest] = hex_digest, encoded
        self.ledger.append({"key": key, "digest": hex_digest, "torch_seed": reduced})
        return hex_digest

    def numpy(self, event, round_idx=-1, uid=-1, **counters):
        digest = bytes.fromhex(self.digest(event, round_idx, uid, **counters))
        words = struct.unpack(">8I", digest)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))

    np_generator = numpy

    def torch_seed(self, event, round_idx=-1, uid=-1, **counters):
        digest = bytes.fromhex(self.digest(event, round_idx, uid, **counters))
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def torch_generator(self, event, round_idx=-1, uid=-1, **counters):
        import torch
        return torch.Generator().manual_seed(self.torch_seed(event, round_idx, uid, **counters))

    @contextmanager
    def scope(self, event, round_idx=-1, uid=-1, *, device="cpu", **counters):
        """Restore ambient state even on failure; only sequential workers=0.

        CPU and the selected CUDA generator are seeded directly. manual_seed()
        is deliberately not used because it can touch unselected CUDA devices.
        """
        import torch
        raw_digest = bytes.fromhex(self.digest(event, round_idx, uid, **counters))
        seed = int.from_bytes(raw_digest[:8], "big") & ((1 << 63) - 1)
        dev = torch.device(device)
        if dev.type not in ("cpu", "cuda"):
            raise ValueError("common policy supports CPU or CUDA only")
        devices = [] if dev.type == "cpu" else [dev.index if dev.index is not None else torch.cuda.current_device()]
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                for idx in devices:
                    torch.cuda.default_generators[idx].manual_seed(seed)
                random.seed(seed)
                # Scoped legacy np.random uses MT19937's array-seed API with
                # the complete digest, not an unchecked 32-bit truncation.
                # numpy() independently uses explicit PCG64/SeedSequence.
                np.random.seed(struct.unpack(">8I", raw_digest))
                yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
