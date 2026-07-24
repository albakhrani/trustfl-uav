"""Hierarchical trust model — draft Section III-E, Algorithm 1.

Three dimensions, each an EWMA of an instantaneous score:
  behavioral    T^beh,  Eqs. (17)-(20): participation, commitment adherence,
                gradient consistency
  computational T^comp, Eqs. (21)-(22): |C_actual - C_expected| / C_expected
  communication T^comm, Eqs. (23)-(24): exp(-T_comm / T_max) * delivered

Overall trust, Eq. (25): T_j = w_beh T^beh + w_comp T^comp + w_comm T^comm,
with w on the simplex.

MODERNIZATION #1 (learnable weights): the combination weights are stored as
logits theta with w = softmax(theta). In `weight_mode: learnable`, after each
round the server updates theta by exponentiated gradient ascent on a reward
signal: dimensions whose per-UAV trust scores better separate
delivered-useful updates (validation-loss-improving) from harmful ones get
more weight. This is the online mechanism; the joint-optimization variant
(alternating minimization, cf. FedLAW arXiv:2511.03529) plugs into the same
interface for the theory work in task #3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def _ewma(prev: float, phi: float, alpha: float, asym: float) -> float:
    """Asymmetric EWMA: trust decreases (phi < prev) are incorporated `asym`
    times faster than increases. asym=1 recovers the draft's Eqs. (17)(21)(23).
    Counters on-off adaptive adversaries that farm trust between attacks."""
    gain = (1 - alpha) * (asym if phi < prev else 1.0)
    gain = min(gain, 1.0)
    return (1 - gain) * prev + gain * phi


@dataclass
class TrustRecord:
    t_beh: float
    t_comp: float
    t_comm: float
    rounds_invited: int = 0
    rounds_participated: int = 0
    rounds_delivered: int = 0
    staleness: int = 0          # rounds since last participation, Eq. (31)
    history: list = field(default_factory=list)


class TrustManager:
    def __init__(self, cfg, num_uavs: int):
        self.cfg = cfg
        self.records: Dict[int, TrustRecord] = {
            j: TrustRecord(cfg.init_trust, cfg.init_trust, cfg.init_trust)
            for j in range(num_uavs)
        }
        # combination weights as logits -> softmax keeps them on the simplex
        w0 = np.array([cfg.w_beh, cfg.w_comp, cfg.w_comm], dtype=float)
        w0 = np.clip(w0, 1e-6, None)
        self.theta = np.log(w0 / w0.sum())
        self.weight_history: list = []

    # ---------------- weights ----------------
    @property
    def weights(self) -> np.ndarray:
        e = np.exp(self.theta - self.theta.max())
        return e / e.sum()

    def update_weights(self, reward_per_dim: np.ndarray) -> None:
        """Exponentiated-gradient step (learnable mode only).
        reward_per_dim[k] > 0 means dimension k currently discriminates
        well between useful and harmful participants."""
        if self.cfg.weight_mode != "learnable":
            return
        r = np.nan_to_num(reward_per_dim, nan=0.0)
        if np.abs(r).max() > 0:
            r = r / (np.abs(r).max())        # scale-free
        # mild decay toward uniform prevents premature collapse on noisy reward
        self.theta = 0.98 * self.theta + self.cfg.learnable_lr * r
        self.theta -= self.theta.max()       # numeric stability
        self.weight_history.append(self.weights.copy())

    # ---------------- Algorithm 1 ----------------
    def overall(self, j: int) -> float:
        r = self.records[j]
        w = self.weights
        t = w[0] * r.t_beh + w[1] * r.t_comp + w[2] * r.t_comm  # Eq. (25)
        return float(np.clip(t, 0.0, 1.0))                       # Alg.1 line 16

    def update(self, j: int, *, invited: bool, participated: bool,
               committed_time: float, actual_time: float,
               grad_consistency: Optional[float],
               comm_time: float, comm_max: float, delivered: bool,
               c_expected: float, c_actual: float) -> None:
        """One UAV's per-round trust update (Algorithm 1 lines 2-16).

        grad_consistency: score in [0,1] (cosine-to-robust-reference with
        norm-ratio penalty, computed by the server); None if UAV did not
        deliver (behavioral component then uses participation+commitment).
        """
        cfg, r = self.cfg, self.records[j]
        if invited:
            r.rounds_invited += 1
            if participated:
                r.rounds_participated += 1
            if delivered:
                r.rounds_delivered += 1

        # Behavioral, Alg.1 lines 3-7
        if invited:
            # Effective participation = DELIVERED rounds (documented deviation
            # from the draft: training without delivering contributes nothing,
            # and counting it lets never-delivering attackers keep p_rate=1).
            p_rate = r.rounds_delivered / max(r.rounds_invited, 1)
            if participated and committed_time > 0:
                # adherence: 1 when on time, decays when slower than committed
                c_adh = float(np.clip(committed_time / max(actual_time, 1e-9), 0.0, 1.0))
            else:
                c_adh = 0.0
            g_cons = float(np.clip(grad_consistency, 0.0, 1.0)) if grad_consistency is not None else 0.0
            phi_beh = (cfg.w_participation * p_rate
                       + cfg.w_commitment * c_adh
                       + cfg.w_gradient * g_cons)
            r.t_beh = _ewma(r.t_beh, phi_beh, cfg.alpha_beh, cfg.asymmetry)  # Eq. (17), asymmetric

        # Computational, Alg.1 lines 8-10 (only measurable when it computed)
        if participated and c_expected > 0:
            phi_comp = float(np.exp(-abs(c_actual - c_expected) / c_expected))  # Eq. (22)
            r.t_comp = _ewma(r.t_comp, phi_comp, cfg.alpha_comp, cfg.asymmetry)

        # Communication, Alg.1 lines 11-13
        if participated:
            phi_comm = float(np.exp(-comm_time / max(comm_max, 1e-9))) * (1.0 if delivered else 0.0)  # Eq. (24)
            r.t_comm = _ewma(r.t_comm, phi_comm, cfg.alpha_comm, cfg.asymmetry)

        # staleness bookkeeping for Q_j, Eq. (31)
        r.staleness = 0 if participated else r.staleness + 1
        r.history.append((r.t_beh, r.t_comp, r.t_comm, self.overall(j)))

    def snapshot(self) -> Dict[int, dict]:
        return {j: {"T": self.overall(j), "beh": r.t_beh, "comp": r.t_comp,
                    "comm": r.t_comm, "staleness": r.staleness}
                for j, r in self.records.items()}
