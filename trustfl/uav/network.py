"""UAV network simulation — draft Section III.

State vector S_j(t) = [p_j, v_j, E_j, f_j, M_j], Eq. (3).
Mobility: Gauss-Markov with speed/acceleration clipping, Eqs. (4)-(6).
Energy:   E_dot = -(P_comp + P_comm + P_mob), Eqs. (7)-(10).
Channel:  Rayleigh air-to-ground, gain Eq. (11), rate Eq. (12), comm time Eq. (13).

The simulator advances once per FL round by `round_duration_s` seconds and
produces, for each UAV, the physical quantities the trust module observes:
actual compute time, comm time, delivery success, and energy drain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..seeding import spawn_generator


@dataclass
class UAVState:
    """One UAV's physical state (Eq. 3) plus reported-vs-actual compute
    capability, which system-dimension attacks exploit."""
    uid: int
    pos: np.ndarray
    vel: np.ndarray
    energy_j: float
    f_actual_hz: float          # true CPU frequency
    f_reported_hz: float        # what the UAV *claims* (== actual if honest)
    high_mobility: bool = False
    alive: bool = True
    # last-round physical observations (inputs to trust, Algorithm 1)
    last_compute_time_s: float = 0.0
    last_committed_time_s: float = 0.0
    last_comm_time_s: float = 0.0
    last_delivered: bool = True
    energy_history: list = field(default_factory=list)


class UAVNetwork:
    def __init__(self, cfg, seed: int):
        self.cfg = cfg
        self.g = spawn_generator(seed, 9001)
        self.server_pos = np.array(cfg.server_pos, dtype=float)
        self.uavs: List[UAVState] = []
        n_high = int(round(cfg.high_mobility_fraction * cfg.num_uavs))
        # High-mobility ids drawn at random per seed (NOT fixed to 0..n_high-1)
        # so group effects cannot alias with uid-linked partition luck.
        hm_ids = set(self.g.choice(cfg.num_uavs, size=n_high,
                                   replace=False).tolist()) if n_high else set()
        for j in range(cfg.num_uavs):
            pos = np.array([self.g.uniform(0, cfg.area_m),
                            self.g.uniform(0, cfg.area_m),
                            self.g.uniform(cfg.alt_min_m, cfg.alt_max_m)])
            f_act = self.g.uniform(cfg.f_min_hz, cfg.f_max_hz)
            self.uavs.append(UAVState(
                uid=j, pos=pos, vel=np.zeros(3),
                energy_j=cfg.battery_j * self.g.uniform(0.8, 1.2),
                f_actual_hz=f_act, f_reported_hz=f_act,
                high_mobility=(j in hm_ids),
            ))
        # Gauss-Markov per-UAV mean velocity direction
        self._gm_mean = {u.uid: self._rand_vel(u) for u in self.uavs}

    # ---------------- mobility, Eqs. (4)-(6) ----------------
    def _rand_vel(self, u: UAVState) -> np.ndarray:
        vmax = self.cfg.v_max * (self.cfg.high_mobility_speed_factor if u.high_mobility else 1.0)
        v = self.g.normal(0, vmax / 2, size=3)
        v[2] *= 0.2  # mostly horizontal motion
        return v

    def step_mobility(self, dt: float) -> None:
        a = self.cfg.gm_alpha
        for u in self.uavs:
            if not u.alive:
                continue
            vmax = self.cfg.v_max * (self.cfg.high_mobility_speed_factor if u.high_mobility else 1.0)
            noise = self.g.normal(0, vmax * np.sqrt(1 - a**2) / 2, size=3)
            new_vel = a * u.vel + (1 - a) * self._gm_mean[u.uid] + noise
            # acceleration clip, Eq. (6)
            dv = new_vel - u.vel
            dv_max = self.cfg.a_max * dt
            if np.linalg.norm(dv) > dv_max:
                dv *= dv_max / (np.linalg.norm(dv) + 1e-12)
            u.vel = u.vel + dv
            # speed clip, Eq. (5)
            sp = np.linalg.norm(u.vel)
            if sp > vmax:
                u.vel *= vmax / sp
            u.pos = u.pos + u.vel * dt  # Eq. (4)
            # reflect at area boundaries / altitude band
            for k, (lo, hi) in enumerate([(0, self.cfg.area_m), (0, self.cfg.area_m),
                                          (self.cfg.alt_min_m, self.cfg.alt_max_m)]):
                if u.pos[k] < lo:
                    u.pos[k], u.vel[k] = 2 * lo - u.pos[k], -u.vel[k]
                elif u.pos[k] > hi:
                    u.pos[k], u.vel[k] = 2 * hi - u.pos[k], -u.vel[k]

    # ---------------- channel, Eqs. (11)-(13) ----------------
    def channel_gain(self, u: UAVState) -> float:
        d = float(np.linalg.norm(u.pos - self.server_pos)) + 1.0
        beta0 = 10 ** (self.cfg.beta0_db / 10)
        rayleigh = self.g.exponential(1.0)  # |g|^2 for Rayleigh fading, Eq. (11)
        return beta0 / (d ** self.cfg.pathloss_alpha) * rayleigh

    def uplink_rate(self, u: UAVState, p_tx_w: float) -> float:
        """Shannon rate, Eq. (12). Interference I_j omitted at this scale
        (orthogonal uplink assumption); hook kept for later."""
        n0 = 10 ** ((self.cfg.noise_dbm_per_hz - 30) / 10)
        snr = p_tx_w * self.channel_gain(u) / (n0 * self.cfg.bandwidth_hz)
        return self.cfg.bandwidth_hz * np.log2(1 + snr)

    def comm_time(self, u: UAVState, num_params: int, p_tx_w: float) -> float:
        bits = num_params * self.cfg.bits_per_param  # Eq. (13)
        rate = max(self.uplink_rate(u, p_tx_w), 1e3)
        return bits / rate

    def nominal_comm_time(self, num_params: int) -> float:
        """Transmission time at a median channel (SNR from mean distance/power),
        used to scale the comm deadline to model size. Independent of any single
        UAV's instantaneous channel so the deadline is a fixed per-round budget."""
        bits = num_params * self.cfg.bits_per_param
        # median SNR: use a mid-range path loss and nominal tx power
        snr = getattr(self.cfg, "nominal_snr", 10.0)  # linear ~10 dB default
        rate = max(self.cfg.bandwidth_hz * np.log2(1 + snr), 1e3)
        return bits / rate

    # ---------------- compute & energy, Eqs. (7)-(10) ----------------
    def compute_time(self, u: UAVState, n_samples: int, epochs: int, f_hz: float) -> float:
        cycles = self.cfg.cycles_per_sample * n_samples * epochs
        return cycles / max(f_hz, 1e6)

    def round_energy(self, u: UAVState, t_comp: float, f_hz: float,
                     t_comm: float, p_tx_w: float, dt: float) -> float:
        p_comp = self.cfg.kappa * f_hz**3                       # Eq. (8)
        p_mob = self.cfg.p0_hover_w + self.cfg.p1_speed_w * np.linalg.norm(u.vel) ** 2  # Eq. (10)
        e = p_comp * t_comp + p_tx_w * t_comm + p_mob * dt      # Eqs. (7),(9)
        return float(e)

    def drain(self, u: UAVState, e_j: float) -> None:
        u.energy_j = max(0.0, u.energy_j - e_j)
        u.energy_history.append(u.energy_j)
        if u.energy_j <= 0.0:
            u.alive = False

    def eligible_energy(self, u: UAVState) -> bool:
        return u.alive and u.energy_j >= self.cfg.e_min_j       # Eq. (29)
