"""Regression test for the NaN-propagation defect seen in
eur_adsys_trustfl/seed2 (EuroSAT grid, rounds 43-52): a delivered client
update containing NaN/Inf entered the median reference and the
gradient-consistency scores, drove the trust EWMAs to NaN, the aggregation
gate then rejected every update (empty influence), selection collapsed to
1 UAV/round, and the global model went NaN (final acc 0.111 vs 0.819 at
round 45).

Guards under test:
  1. Server.run_round — a delivered update with non-finite entries is
     scored as a failed delivery and never reaches the consistency
     reference, trust updates, or aggregation.
  2. trust._ewma — skips non-finite instantaneous scores, so trust can
     never become NaN regardless of upstream observations.

Runs standalone (no pytest dependency):
    .venv/Scripts/python.exe tests/test_nan_guard.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def test_ewma_skips_nonfinite():
    from trustfl.trust.trust import _ewma

    for bad in (float("nan"), float("inf"), float("-inf")):
        assert _ewma(0.6, bad, 0.7, 3.0) == 0.6

    # finite semantics unchanged: increase uses gain (1-alpha),
    # decrease uses asym*(1-alpha) capped at 1
    assert abs(_ewma(0.5, 1.0, 0.7, 3.0) - (0.7 * 0.5 + 0.3 * 1.0)) < 1e-12
    assert abs(_ewma(0.5, 0.0, 0.7, 3.0) - (0.1 * 0.5 + 0.9 * 0.0)) < 1e-12


def test_trust_update_nonfinite_inputs():
    from trustfl.config import TrustConfig
    from trustfl.trust.trust import TrustManager

    tm = TrustManager(TrustConfig(), 3)
    nan, inf = float("nan"), float("inf")
    # every non-finite observation combination the server could conceivably pass
    tm.update(0, invited=True, participated=True, committed_time=5.0,
              actual_time=nan, grad_consistency=nan, comm_time=inf,
              comm_max=5.0, delivered=True, c_expected=5.0, c_actual=nan)
    tm.update(1, invited=True, participated=True, committed_time=inf,
              actual_time=3.0, grad_consistency=inf, comm_time=nan,
              comm_max=5.0, delivered=True, c_expected=inf, c_actual=3.0)
    tm.update(2, invited=True, participated=True, committed_time=5.0,
              actual_time=4.0, grad_consistency=0.8, comm_time=2.0,
              comm_max=5.0, delivered=True, c_expected=5.0, c_actual=4.0)

    for j, s in tm.snapshot().items():
        vals = [s["T"], s["beh"], s["comp"], s["comm"]]
        assert np.all(np.isfinite(vals)), f"trust for uav {j} non-finite: {s}"
        assert 0.0 <= s["T"] <= 1.0


def test_nan_update_short_synthetic_run():
    """End-to-end: inject a NaN/Inf delta into a delivered update mid-run
    and require that the model stays finite, trust stays finite, the
    poisoned update is excluded from aggregation, and selection does not
    collapse afterwards."""
    from trustfl.attacks.attacks import AttackController
    from trustfl.config import ExperimentConfig
    from trustfl.data import iid_partition, label_variance, load_dataset, split_server_val
    from trustfl.fl.client import Client, flatten_state
    from trustfl.fl.server import Server
    from trustfl.models import build_model
    from trustfl.seeding import set_all_seeds
    from trustfl.trust.trust import TrustManager
    from trustfl.uav.network import UAVNetwork

    cfg = ExperimentConfig.from_dict({
        "name": "nan_guard_regression",
        "seed": 0,
        "data": {"dataset": "synthetic", "partition": "iid"},
        "train": {"rounds": 6, "local_epochs": 1, "batch_size": 32,
                  "lr": 0.05, "eval_every": 10, "device": "cpu"},
        "uav": {"enabled": False, "num_uavs": 8},
        "trust": {"k_max": 8},
        "method": {"name": "trustfl"},
        "attack": {"attack_type": "none", "malicious_fraction": 0.0},
    })
    n = cfg.uav.num_uavs
    set_all_seeds(cfg.seed)
    train, test = load_dataset(cfg.data, cfg.seed)
    train, server_val = split_server_val(train, cfg.data.server_val_fraction, cfg.seed)
    parts = iid_partition(train, n, cfg.seed)
    sigma2 = {j: label_variance(train, parts[j], 5) for j in range(n)}
    base = train.dataset if hasattr(train, "dataset") else train
    clients = [Client(j, base, parts[j], cfg.train, "cpu", cfg.seed) for j in range(n)]

    with tempfile.TemporaryDirectory() as out_dir:
        server = Server(cfg, build_model(cfg.model, "synthetic"), clients,
                        UAVNetwork(cfg.uav, cfg.seed), TrustManager(cfg.trust, n),
                        AttackController(cfg.attack, n, cfg.seed),
                        server_val, test, sigma2, "cpu", out_dir)

        # poison uav 0's delta on round 2 with mixed NaN/Inf entries, exactly
        # the payload class that killed eur_adsys_trustfl/seed2 at round 47
        poison_round, poison_uid = 2, 0
        current = {"t": -1}
        orig_corrupt = server.attack.corrupt_delta

        def poisoned_corrupt(uid, delta):
            d = orig_corrupt(uid, delta)
            if current["t"] == poison_round and uid == poison_uid:
                d = d.clone()
                d.view(-1)[::5] = float("nan")
                d.view(-1)[1] = float("inf")
            return d

        server.attack.corrupt_delta = poisoned_corrupt

        recs = []
        for t in range(cfg.train.rounds):
            current["t"] = t
            recs.append(server.run_round(t))
            flat = flatten_state(server.model.state_dict())
            assert bool(torch.isfinite(flat).all()), f"model non-finite after round {t}"
            for j, s in server.trust.snapshot().items():
                vals = [s["T"], s["beh"], s["comp"], s["comm"]]
                assert np.all(np.isfinite(vals)), \
                    f"trust for uav {j} non-finite after round {t}: {s}"

    pr = recs[poison_round]
    assert poison_uid in pr["invited"], "test setup broken: poisoned uav not invited"
    assert poison_uid not in pr["delivered"], "non-finite update reached aggregation"
    assert pr.get("nonfinite_dropped") == [poison_uid]

    # the seed2 failure mode: selection collapsed to 1 invited uav/round
    for rec in recs[poison_round + 1:]:
        assert len(rec["invited"]) >= n - 1, \
            f"selection collapsed after poison round: {rec['invited']}"
    assert np.isfinite(recs[-1]["mean_trust_honest"])
    assert np.isfinite(recs[-1]["test_acc"])


def test_quarantine_zeroes_all_dimensions():
    """Spec semantics: a quarantined round scores 0 on ALL THREE dimensions
    and each EWMA processes the 0 normally (asymmetric down-update) —
    plausible-looking compute/comm times must earn nothing that round."""
    from trustfl.config import TrustConfig
    from trustfl.trust.trust import TrustManager, _ewma

    cfg = TrustConfig()
    tm = TrustManager(cfg, 2)
    # healthy-looking observations, but the payload was quarantined
    tm.update(0, invited=True, participated=True, committed_time=5.0,
              actual_time=4.0, grad_consistency=0.9, comm_time=1.0,
              comm_max=5.0, delivered=False, c_expected=5.0, c_actual=4.0,
              quarantined=True)
    expect = _ewma(cfg.init_trust, 0.0, cfg.alpha_beh, cfg.asymmetry)
    r = tm.records[0]
    for name, got in (("beh", r.t_beh), ("comp", r.t_comp), ("comm", r.t_comm)):
        assert abs(got - expect) < 1e-12, f"t_{name}={got}, expected {expect}"

    # identical observations WITHOUT quarantine keep the normal (higher) scores
    tm.update(1, invited=True, participated=True, committed_time=5.0,
              actual_time=4.0, grad_consistency=0.9, comm_time=1.0,
              comm_max=5.0, delivered=True, c_expected=5.0, c_actual=4.0)
    assert tm.records[1].t_beh > expect and tm.records[1].t_comp > expect


def test_zero_norm_quarantined_short_run():
    """A delivered all-zeros delta (zero norm, perfectly finite) must be
    quarantined exactly like a NaN payload: excluded from aggregation,
    logged, all three trust dimensions driven down."""
    from trustfl.attacks.attacks import AttackController
    from trustfl.config import ExperimentConfig
    from trustfl.data import iid_partition, label_variance, load_dataset, split_server_val
    from trustfl.fl.client import Client, flatten_state
    from trustfl.fl.server import Server
    from trustfl.models import build_model
    from trustfl.seeding import set_all_seeds
    from trustfl.trust.trust import TrustManager
    from trustfl.uav.network import UAVNetwork

    cfg = ExperimentConfig.from_dict({
        "name": "zero_norm_guard",
        "seed": 0,
        "data": {"dataset": "synthetic", "partition": "iid"},
        "train": {"rounds": 5, "local_epochs": 1, "batch_size": 32,
                  "lr": 0.05, "eval_every": 10, "device": "cpu"},
        "uav": {"enabled": False, "num_uavs": 8},
        "trust": {"k_max": 8},
        "method": {"name": "trustfl"},
        "attack": {"attack_type": "none", "malicious_fraction": 0.0},
    })
    n = cfg.uav.num_uavs
    set_all_seeds(cfg.seed)
    train, test = load_dataset(cfg.data, cfg.seed)
    train, server_val = split_server_val(train, cfg.data.server_val_fraction, cfg.seed)
    parts = iid_partition(train, n, cfg.seed)
    sigma2 = {j: label_variance(train, parts[j], 5) for j in range(n)}
    base = train.dataset if hasattr(train, "dataset") else train
    clients = [Client(j, base, parts[j], cfg.train, "cpu", cfg.seed) for j in range(n)]

    with tempfile.TemporaryDirectory() as out_dir:
        server = Server(cfg, build_model(cfg.model, "synthetic"), clients,
                        UAVNetwork(cfg.uav, cfg.seed), TrustManager(cfg.trust, n),
                        AttackController(cfg.attack, n, cfg.seed),
                        server_val, test, sigma2, "cpu", out_dir)

        poison_round, poison_uid = 2, 0
        current = {"t": -1}
        orig_corrupt = server.attack.corrupt_delta

        def zeroing_corrupt(uid, delta):
            d = orig_corrupt(uid, delta)
            if current["t"] == poison_round and uid == poison_uid:
                d = torch.zeros_like(d)
            return d

        server.attack.corrupt_delta = zeroing_corrupt

        recs = []
        trust_before = None
        for t in range(cfg.train.rounds):
            current["t"] = t
            if t == poison_round:
                trust_before = server.trust.overall(poison_uid)
            recs.append(server.run_round(t))
            flat = flatten_state(server.model.state_dict())
            assert bool(torch.isfinite(flat).all()), f"model non-finite after round {t}"

        pr = recs[poison_round]
        assert poison_uid in pr["invited"], "test setup broken: poisoned uav not invited"
        assert poison_uid not in pr["delivered"], "zero-norm update reached aggregation"
        assert pr.get("nonfinite_dropped") == [poison_uid]
        assert server.trust.overall(poison_uid) < trust_before, \
            "quarantined round did not reduce trust"
        r = server.trust.records[poison_uid]
        assert np.all(np.isfinite([r.t_beh, r.t_comp, r.t_comm]))


if __name__ == "__main__":
    test_ewma_skips_nonfinite()
    test_trust_update_nonfinite_inputs()
    test_nan_update_short_synthetic_run()
    test_quarantine_zeroes_all_dimensions()
    test_zero_norm_quarantined_short_run()
    print("all NaN-guard regression tests passed")
