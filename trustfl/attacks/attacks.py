"""Attack models — the three families in the novelty thesis.

GRADIENT-SPACE (what existing defenses were built for):
  label_flip      y -> (C-1) - y with prob p during local training
  sign_flip       delta -> -delta
  grad_scale      delta -> s * delta
  random_update   delta ~ N(0, sigma^2 I)

SYSTEM-DIMENSION (invisible to gradient-only defenses — flagship claim #2):
  resource_misreport      claim f_reported = factor * f_actual, so committed
                          compute time is factor-x optimistic; actual training
                          runs slower than promised, wasting round budget
  strategic_delay         deliver the update as late as possible (just inside
                          or beyond the deadline), degrading round timing
  selective_participation show up only with probability p when invited

ADAPTIVE (flagship claim #4): the adversary observes its own trust score,
behaves honestly until trust >= target, then attacks in gradient space while
keeping system behavior clean; if trust drops below target it goes quiet and
rebuilds. This is the trust-aware "on-off" adversary.

All attacks are implemented server-blind: the runner applies them, the
defense never sees ground-truth malice.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np
import torch

from ..seeding import spawn_generator

GRADIENT_ATTACKS = {"label_flip", "sign_flip", "grad_scale", "random_update"}
SYSTEM_ATTACKS = {"resource_misreport", "strategic_delay", "selective_participation"}
ADAPTIVE_ATTACKS = {"adaptive_trust", "adaptive_evasive", "adaptive_system"}


class AttackController:
    def __init__(self, cfg, num_uavs: int, seed: int):
        self.cfg = cfg
        self.g = spawn_generator(seed, 6666)
        n_mal = int(round(cfg.malicious_fraction * num_uavs))
        # malicious ids fixed per run (reproducible), chosen uniformly
        self.malicious: Set[int] = set(
            self.g.choice(num_uavs, size=n_mal, replace=False).tolist()
        ) if n_mal > 0 else set()
        self._adaptive_attacking: Dict[int, bool] = {j: False for j in self.malicious}
        self._shared_dir = None  # coalition poison direction (adaptive_evasive)
        # adaptive_system: which system attack each striking attacker deploys,
        # and a per-attacker rotation counter so modes cycle over strikes
        self._sys_mode: Dict[int, Optional[str]] = {j: None for j in self.malicious}
        self._sys_rotation: Dict[int, int] = {j: 0 for j in self.malicious}

    # ------------- setup-time effects -------------
    def apply_static_effects(self, network) -> None:
        """resource_misreport is a standing lie about capability."""
        if self.cfg.attack_type == "resource_misreport":
            for u in network.uavs:
                if u.uid in self.malicious:
                    u.f_reported_hz = u.f_actual_hz * self.cfg.misreport_factor

    # ------------- per-round decisions -------------
    def will_participate(self, uid: int, invited: bool) -> bool:
        if not invited:
            return False
        if uid not in self.malicious:
            return True
        atk = self.cfg.attack_type
        if atk == "selective_participation":
            return bool(self.g.random() < self.cfg.selective_prob)
        # adaptive system adversary: skip participation only while striking
        if atk == "adaptive_system" and self._adaptive_attacking.get(uid, False):
            if self._sys_mode.get(uid) == "selective":
                return bool(self.g.random() < self.cfg.selective_prob)
        return True

    def comm_delay(self, uid: int, deadline_s: float, base_t_comm: float) -> float:
        """Extra delay injected by strategic_delay. Returns the ADDITIONAL
        seconds needed so total arrival lands at the attacker's target:
        just inside the deadline (stealth) half the time, just beyond
        (wasted slot) the other half."""
        if uid not in self.malicious:
            return 0.0
        atk = self.cfg.attack_type
        striking = self._adaptive_attacking.get(uid, False)
        delay_active = (atk == "strategic_delay") or (
            atk == "adaptive_system" and striking and self._sys_mode.get(uid) == "delay")
        if delay_active:
            slack = self.cfg.delay_slack_s
            target = deadline_s - slack if self.g.random() < 0.5 else deadline_s + slack
            return float(max(0.0, target - base_t_comm))
        return 0.0

    def update_adaptive(self, uid: int, trust_score: float, round_idx: int,
                        network=None) -> None:
        """Adaptive on-off logic (called before local training each round).

        For adaptive_system, when the adversary transitions to 'striking' it
        also SELECTS which system-dimension attack to deploy, cycling through
        modes so no single behavioral signal dominates -- this is the adaptive
        adversary that evolves against the trust score in the SYSTEM dimension.
        """
        atk = self.cfg.attack_type
        if uid not in self.malicious or atk not in ADAPTIVE_ATTACKS:
            return
        was_striking = self._adaptive_attacking.get(uid, False)
        if round_idx < self.cfg.adaptive_probe_rounds:
            striking = False
        elif trust_score >= self.cfg.adaptive_trust_target:
            striking = True
        elif trust_score < self.cfg.adaptive_trust_target - 0.15:
            striking = False  # trust dropped: back off and rebuild
        else:
            striking = was_striking  # hysteresis band
        self._adaptive_attacking[uid] = striking

        if atk == "adaptive_system":
            if striking:
                # choose/rotate the system-dimension mode on each fresh strike
                if not was_striking or self._sys_mode.get(uid) is None:
                    modes = ["delay", "selective", "misreport"]
                    self._sys_mode[uid] = modes[self._sys_rotation[uid] % len(modes)]
                    self._sys_rotation[uid] += 1
            else:
                self._sys_mode[uid] = None
            # apply/clear the misreport lie dynamically (it's a standing state)
            if network is not None:
                for u in network.uavs:
                    if u.uid == uid:
                        if striking and self._sys_mode.get(uid) == "misreport":
                            u.f_reported_hz = u.f_actual_hz * self.cfg.misreport_factor
                        else:
                            u.f_reported_hz = u.f_actual_hz  # honest report
                        break

    def label_hook(self, uid: int, num_classes: int):
        """Returns malicious_hook for Client.local_update, or None."""
        atk = self.cfg.attack_type
        if uid not in self.malicious or atk != "label_flip":
            return None
        p = self.cfg.label_flip_prob
        gen = self.g

        def hook(kind, y):
            if kind != "label":
                return y
            mask = torch.tensor(gen.random(len(y)) < p, device=y.device)
            return torch.where(mask, (num_classes - 1) - y, y)
        return hook

    def corrupt_delta(self, uid: int, delta: torch.Tensor) -> torch.Tensor:
        """Delta-space corruption after local training."""
        if uid not in self.malicious:
            return delta
        atk = self.cfg.attack_type
        if atk == "adaptive_trust":
            # payload while active: scaled sign flip; clean behavior otherwise
            if self._adaptive_attacking.get(uid, False):
                return -delta * self.cfg.grad_scale_factor
            return delta
        if atk == "adaptive_evasive":
            # Trust-farming + COORDINATED gradient-evasive payload (corrected).
            #
            # All attackers add the SAME fixed malicious direction m (seeded,
            # identical across colluders, INDEPENDENT of each attacker's benign
            # update so it is neither parallel to benign nor self-cancelling).
            # The emitted update is  benign + kappa * ||benign|| * m_hat, i.e.
            # the honest update plus a shared poison of controlled relative
            # magnitude kappa. Because m is identical across all f attackers,
            # the poison accumulates as f*kappa in the sum (PoisonedFL-style
            # coordination) rather than averaging out. Trust-farming gates it:
            # honest until trusted, then strike.
            if not self._adaptive_attacking.get(uid, False):
                return delta  # farm trust
            g = delta
            gnorm = g.norm().clamp_min(1e-12)
            if self._shared_dir is None or self._shared_dir.numel() != g.numel():
                # one fixed unit direction for the whole attacker coalition
                gen = torch.Generator().manual_seed(1234567)
                m = torch.randn(g.numel(), generator=gen)
                self._shared_dir = (m / m.norm()).to(g.dtype)
            kappa = float(self.cfg.evasive_poison_scale)
            out = g + kappa * gnorm * self._shared_dir
            # optional cosine cap: if evasive_align>0, shrink poison so
            # cosine(out, g) >= evasive_align (evade cosine filters). align<=0
            # disables the cap (maximally damaging, geometric-only evasion).
            a = float(self.cfg.evasive_align)
            if a > 0.0:
                # binary-search the largest kappa' <= kappa with cosine >= a
                lo, hi = 0.0, kappa
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    cand = g + mid * gnorm * self._shared_dir
                    cos = float((cand @ g) / (cand.norm() * gnorm).clamp_min(1e-12))
                    if cos >= a:
                        lo = mid
                    else:
                        hi = mid
                out = g + lo * gnorm * self._shared_dir
            return out
        if atk == "sign_flip":
            return -delta
        if atk == "grad_scale":
            return delta * self.cfg.grad_scale_factor
        if atk == "random_update":
            return torch.randn_like(delta) * self.cfg.random_update_std
        return delta
