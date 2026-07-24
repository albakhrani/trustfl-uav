"""Algorithm 2: trust-constrained participant selection (Eqs. 26-31) and
aggregation-weight computation (Eqs. 32-33).

Greedy selection is exactly optimal for this problem because the objective
(26) is additive with a cardinality constraint (27) — sorting by utility is
the O(M log M) step behind flagship claim #3 (cost vs Krum's O(M^2)).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def quality_factor(staleness: int, sigma2: float, sigma2_max: float, beta: float) -> float:
    """Q_j, Eq. (31)."""
    q = np.exp(-beta * staleness) * (1.0 - min(sigma2 / max(sigma2_max, 1e-9), 1.0))
    return float(max(q, 1e-4))


def select_participants(trust: Dict[int, float], energy_ok: Dict[int, bool],
                        n_samples: Dict[int, int], staleness: Dict[int, int],
                        sigma2: Dict[int, float], cfg, available: List[int],
                        ) -> Tuple[List[int], Dict[int, float]]:
    """Returns (selected UAV ids, utility per eligible UAV).

    Phase 1 (Alg.2 lines 1-7): filter by T_min (Eq. 28) and E_min (Eq. 29).
    Phase 2 (lines 8-18): pick top-Kmax by utility T_j * (n_j/n) * Q_j (Eq. 26).
    """
    sigma2_max = max(max(sigma2.values(), default=1.0), 1e-9)
    n_total = max(sum(n_samples.values()), 1)
    eligible = [j for j in available
                if trust[j] >= cfg.t_min and energy_ok.get(j, False)]
    utility = {}
    for j in eligible:
        q = quality_factor(staleness[j], sigma2[j], sigma2_max, cfg.staleness_beta)
        utility[j] = trust[j] * (n_samples[j] / n_total) * q
    ranked = sorted(utility, key=utility.get, reverse=True)  # O(M log M)
    n_exp = min(getattr(cfg, "explore_slots", 0), cfg.k_max)
    top = ranked[: cfg.k_max - n_exp]
    if n_exp > 0:
        benched = [j for j in eligible if j not in top]
        benched.sort(key=lambda j: staleness[j], reverse=True)  # longest wait first
        top = top + benched[:n_exp]
    return top, utility


def aggregation_weights(selected: List[int], trust: Dict[int, float],
                        n_samples: Dict[int, int], t_min: float = 0.0,
                        gate: bool = False) -> Dict[int, float]:
    """Aggregation weights, Alg.2 phase 3.

    gate=False: draft's linear rule, omega_j = T_j n_j / Z (Eq. 33).
    gate=True (default in config): trust-gated rule
        omega_j = ReLU(T_j - T_min) n_j / Z.
    Linear weighting attenuates a below-average-trust attacker only
    proportionally; gating removes its influence entirely once trust falls
    to the threshold — this is the version we analyze in the paper, with
    Eq. (33) kept as an ablation."""
    def w(j):
        return (max(trust[j] - t_min, 0.0) if gate else trust[j]) * n_samples[j]
    z = sum(w(j) for j in selected)
    if z <= 0:
        return {j: 1.0 / max(len(selected), 1) for j in selected}
    return {j: w(j) / z for j in selected}


# ---------------------------------------------------------------------------
# Algorithm 3: resource allocation by dual decomposition (Eqs. 34-39).
# Per-UAV, the frequency subproblem has the closed form of Alg.3 line 9;
# the power subproblem is 1-D and solved by golden-section search (line 13).
# ---------------------------------------------------------------------------

def _golden_section(f, lo: float, hi: float, tol: float = 1e-4, iters: int = 60) -> float:
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if abs(b - a) < tol * (abs(lo) + abs(hi)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2


def allocate_resources(network, uav, n_samples: int, epochs: int,
                       num_params: int, res_cfg, uav_cfg) -> Tuple[float, float]:
    """Returns (f_j*, P_tx_j*) minimizing Eq. (34) for one UAV via dual
    decomposition (Algorithm 3). Falls back to midpoint values if disabled."""
    if not res_cfg.enabled:
        return uav.f_actual_hz, (uav_cfg.p_tx_min_w + uav_cfg.p_tx_max_w) / 2

    cycles = uav_cfg.cycles_per_sample * n_samples * epochs
    kappa = uav_cfg.kappa
    l1, l2, l3 = res_cfg.lambda1, res_cfg.lambda2, res_cfg.lambda3
    lam = mu = 0.0  # dual variables for (35),(36)
    f_lo, f_hi = uav_cfg.f_min_hz, min(uav_cfg.f_max_hz, uav.f_actual_hz)
    f_j = (f_lo + f_hi) / 2
    p_j = (uav_cfg.p_tx_min_w + uav_cfg.p_tx_max_w) / 2

    # expected rate at current position (fading averaged out for planning:
    # E[|g|^2]=1), used inside the optimizer; realized comm time is sampled
    # by the network with actual fading.
    d = float(np.linalg.norm(uav.pos - network.server_pos)) + 1.0
    beta0 = 10 ** (uav_cfg.beta0_db / 10)
    gain = beta0 / (d ** uav_cfg.pathloss_alpha)
    n0b = 10 ** ((uav_cfg.noise_dbm_per_hz - 30) / 10) * uav_cfg.bandwidth_hz
    bits = num_params * uav_cfg.bits_per_param

    def t_comm(p):
        rate = uav_cfg.bandwidth_hz * np.log2(1 + p * gain / n0b)
        return bits / max(rate, 1e3)

    prev_obj = np.inf
    for _ in range(res_cfg.max_iters):
        # frequency closed form (Alg.3 line 9): d/df [ (l1+lam) C_comp + l3 kappa f^3 C_comp ]
        # with C_comp = cycles/f  =>  f* = cbrt((l1+lam) / (2 l3 kappa))
        f_star = ((l1 + lam) / max(2 * l3 * kappa, 1e-30)) ** (1.0 / 3.0)
        f_j = float(np.clip(f_star, f_lo, f_hi))
        t_comp = cycles / f_j

        # power subproblem (Alg.3 lines 10-13): golden-section on
        # (l2+mu) T_comm(p) + l3 p T_comm(p)
        p_j = _golden_section(lambda p: (l2 + mu) * t_comm(p) + l3 * p * t_comm(p),
                              uav_cfg.p_tx_min_w, uav_cfg.p_tx_max_w)

        # dual updates (Alg.3 lines 14-17), projected to >= 0
        lam = max(0.0, lam + res_cfg.dual_step * (t_comp - res_cfg.tau_max_s))
        mu = max(0.0, mu + res_cfg.dual_step * (t_comm(p_j) - res_cfg.t_max_s))

        obj = l1 * t_comp + l2 * t_comm(p_j) + l3 * (kappa * f_j**3 * t_comp + p_j * t_comm(p_j))
        if abs(prev_obj - obj) < res_cfg.tol:
            break
        prev_obj = obj
    return f_j, p_j
