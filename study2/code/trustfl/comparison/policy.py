"""Public requests, saved opportunities and simulator-only realization."""
from __future__ import annotations

import math


def validate_schedule(schedule, rounds, num_clients, k):
    if len(schedule) != rounds:
        raise ValueError("schedule length differs from planned rounds")
    for row in schedule:
        if (not isinstance(row, list) or len(row) != k or
                any(type(j) is not int or not 0 <= j < num_clients for j in row) or
                row != sorted(set(row))):
            raise ValueError("schedule must contain sorted, distinct, in-range K-subsets")
    return schedule


def generate_schedule(rng, rounds, num_clients, k):
    if not 0 <= k <= num_clients:
        raise ValueError("invalid K")
    return [sorted(rng.numpy("invitation_schedule", t).choice(
        num_clients, k, replace=False).tolist()) for t in range(rounds)]


def executable_invitations(scheduled, network, *, enabled=True):
    states = {u.uid: u for u in network.uavs}
    executable, excluded = [], {}
    for uid in scheduled:
        if uid not in states:
            raise ValueError(f"unknown scheduled client {uid}")
        u = states[uid]
        reason = None
        if enabled:
            if not u.alive:
                reason = "dead"
            elif not network.eligible_energy(u):
                reason = "below_energy_floor"
        if reason:
            excluded[str(uid)] = reason
        else:
            executable.append(uid)
    return executable, excluded


def resource_request(uav_cfg):
    """The defender request reads public configuration only, never a UAV."""
    return {"frequency_hz": float(uav_cfg.f_max_hz),
            "power_w": float((uav_cfg.p_tx_min_w + uav_cfg.p_tx_max_w) / 2)}


def realize_resources(request, uav, uav_cfg):
    """Simulator oracle boundary: enforce true CPU cap and global TX bounds."""
    f, p = request["frequency_hz"], request["power_w"]
    if not all(math.isfinite(v) for v in (f, p, uav.f_actual_hz)) or min(f, uav.f_actual_hz) <= 0:
        raise ValueError("invalid resource request or physical cap")
    realized_f = min(f, uav.f_actual_hz)
    realized_p = min(max(p, uav_cfg.p_tx_min_w), uav_cfg.p_tx_max_w)
    reasons = []
    if realized_f != f:
        reasons.append("true_frequency_cap")
    if realized_p != p:
        reasons.append("configured_power_bounds")
    return {"requested_frequency_hz": f, "requested_power_w": p,
            "realized_frequency_hz": float(realized_f), "realized_power_w": float(realized_p),
            "true_frequency_cap_hz": float(uav.f_actual_hz),
            "reported_frequency_hz": float(uav.f_reported_hz), "cap_reasons": reasons,
            "information_scope": "oracle_simulation"}
