"""Paired invitation inputs and explicitly scoped resource information."""
from __future__ import annotations
from ..comparison.policy import executable_invitations, realize_resources, resource_request
from .protocol import METHODS, POLICIES


def validate_priorities(priorities, rounds, num_clients):
    import numpy as np
    values = np.asarray(priorities)
    if values.dtype.kind not in "iu" or values.shape != (rounds, num_clients):
        raise ValueError("Priority array must be rounds x enrolled clients, integer")
    expected = list(range(num_clients))
    if any(sorted(row.tolist()) != expected for row in values):
        raise ValueError("Every priority row must permute every enrolled client exactly once")
    return values


def generate_priorities(rng, rounds, num_clients):
    import numpy as np
    if type(rounds) is not int or type(num_clients) is not int or min(rounds,num_clients) < 1:
        raise ValueError("Positive typed rounds/client counts required")
    rows = [rng.numpy("invitation_schedule",t).permutation(num_clients) for t in range(rounds)]
    return validate_priorities(np.asarray(rows,dtype=np.int64),rounds,num_clients)


def select_invitations(priority, network, k, policy, method, *, native_trustfl=None, enabled=True):
    """Return scheduled, executable, exclusions. No common-policy replacement.

    Native baseline first-K-eligible preserves random-uniform subset selection
    with changed, explicitly prospective event-indexed consumption. TrustFL's
    existing native selection callback owns its score/exploration mathematics.
    """
    if policy not in POLICIES or method not in METHODS:
        raise ValueError("Unsupported scientific policy or method")
    ids = [int(v) for v in priority]
    actual = sorted(u.uid for u in network.uavs)
    if sorted(ids) != actual or type(k) is not int or not 0 <= k <= len(ids):
        raise ValueError("Full enrolled priority permutation and bounded typed K required")
    if policy == "common":
        scheduled = ids[:k]
    elif method == "trustfl":
        if native_trustfl is None:
            raise ValueError("Native TrustFL selection requires unchanged selector callback")
        scheduled = list(native_trustfl())
        if len(scheduled) > k or len(scheduled) != len(set(scheduled)) or any(x not in actual for x in scheduled):
            raise ValueError("Native TrustFL callback returned invalid IDs")
    else:
        eligible, _ = executable_invitations(ids,network,enabled=enabled)
        scheduled = eligible[:k]
    executable, excluded = executable_invitations(scheduled,network,enabled=enabled)
    return scheduled, executable, excluded


def requested_resources(policy, method, uav_cfg, uav, *, native_allocation=None):
    """Record defender requests; physics clipping is a distinct operation."""
    if policy not in POLICIES or method not in METHODS:
        raise ValueError("Unsupported resource policy")
    if policy == "prospective_native_policy" and method == "trustfl":
        if native_allocation is None:
            raise ValueError("Existing native allocator callback required")
        frequency,power = native_allocation()
        request = {"frequency_hz":float(frequency),"power_w":float(power)}
        scope = "oracle_simulation_native_allocator_reads_true_caps_and_positions"
    elif policy == "prospective_native_policy":
        request = {"frequency_hz":float(uav.f_actual_hz),
                   "power_w":float((uav_cfg.p_tx_min_w+uav_cfg.p_tx_max_w)/2)}
        scope = "oracle_simulation_native_baseline_requests_true_cpu_cap"
    else:
        request = resource_request(uav_cfg)
        scope = "public_fixed_request;true_caps_used_only_by_simulator_realization"
    result = realize_resources(request,uav,uav_cfg)
    result["request_information_scope"] = scope
    return result


def execution_order(policy, method, executable):
    """Preserve native baseline UID order and native TrustFL selector order."""
    if policy not in POLICIES or method not in METHODS:
        raise ValueError("Unsupported execution policy/method")
    return sorted(executable) if policy=="prospective_native_policy" and method!="trustfl" else list(executable)
