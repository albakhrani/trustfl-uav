"""Server / round orchestrator — Algorithm 4 (main TrustFL-UAV loop),
generalized so every baseline runs through the same physical simulation.

Per round:
  1. advance UAV mobility (Sec III-B)
  2. decide invitations:
       - trustfl: Algorithm 2 (trust+energy filter, utility top-K)
       - baselines: random K among energy-eligible alive UAVs
  3. resource allocation for participants (Algorithm 3; trustfl only,
     baselines use nominal f and mid transmit power)
  4. local training (with gradient attacks), physical compute/comm times
     (with system attacks), deadline enforcement, energy drain
  5. aggregation (method-specific), global model update
  6. trust update (Algorithm 1) — always computed so we can log trust
     trajectories even for baselines
  7. learnable trust-weight update (modernization #1)
  8. evaluation + metrics logging (JSONL)

Timing measurement for flagship claim #3 (cost vs robustness): wall-clock of
the aggregation step is recorded separately per round.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..fl.aggregation import AGGREGATORS
from ..fl.client import Client, flatten_state, unflatten_like
from ..trust.selection import aggregation_weights, allocate_resources, select_participants
from ..trust.trust import TrustManager
from ..seeding import spawn_generator


class Server:
    def __init__(self, cfg, model, clients: List[Client], network, trust_mgr: TrustManager,
                 attack, server_val, test_set, sigma2: Dict[int, float], device, out_dir: str):
        self.cfg = cfg
        self.model = model.to(device)
        self.clients = {c.uid: c for c in clients}
        self.network = network
        self.trust = trust_mgr
        self.attack = attack
        self.device = device
        self.out_dir = out_dir
        self.sigma2 = sigma2
        self.g = spawn_generator(cfg.seed, 4242)
        self.num_params = sum(p.numel() for p in model.parameters())
        self.val_loader = DataLoader(server_val, batch_size=256, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        open(self.metrics_path, "w").close()  # fresh file per run
        self._ground_truth_malicious = set(getattr(attack, "malicious", set()))
        groups = {
            "malicious": sorted(self._ground_truth_malicious),
            "high_mobility": sorted(u.uid for u in network.uavs if u.high_mobility),
        }
        with open(os.path.join(out_dir, "groups.json"), "w") as f:
            json.dump(groups, f, indent=2)
        self._high_mobility = set(groups["high_mobility"])

    # ---------------- evaluation ----------------
    @torch.no_grad()
    def _eval(self, loader) -> Dict[str, float]:
        self.model.eval()
        correct = total = 0
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss_sum += float(F.cross_entropy(out, y, reduction="sum"))
            correct += int((out.argmax(1) == y).sum())
            total += len(y)
        return {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1)}

    def _val_loss_with_delta(self, delta: torch.Tensor) -> float:
        """Apply candidate delta, measure val loss, revert. Used by FedGreed
        and by the learnable-trust-weight reward."""
        sd = self.model.state_dict()
        backup = copy.deepcopy(sd)
        flat = flatten_state(sd).cpu() + delta
        self.model.load_state_dict({k: v.to(self.device) for k, v in unflatten_like(flat, backup).items()})
        m = self._eval(self.val_loader)
        self.model.load_state_dict(backup)
        return m["loss"]

    def _server_update(self) -> torch.Tensor:
        """FLTrust's root-dataset update delta_0 (one local pass on val set)."""
        local = copy.deepcopy(self.model).to(self.device)
        opt = torch.optim.SGD(local.parameters(), lr=self.cfg.train.lr, momentum=self.cfg.train.momentum)
        local.train()
        for _ in range(self.cfg.train.local_epochs):
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                F.cross_entropy(local(x), y).backward()
                opt.step()
        return flatten_state(local.state_dict()).cpu() - flatten_state(self.model.state_dict()).cpu()

    # ---------------- invitations ----------------
    def _invite(self) -> List[int]:
        method = self.cfg.method.name
        net_on = self.cfg.uav.enabled
        alive = [u.uid for u in self.network.uavs if u.alive] if net_on else list(self.clients)
        energy_ok = {u.uid: self.network.eligible_energy(u) for u in self.network.uavs} if net_on \
            else {j: True for j in self.clients}

        if method == "trustfl":
            trust_scores = {j: self.trust.overall(j) for j in alive}
            staleness = {j: self.trust.records[j].staleness for j in alive}
            n_samples = {j: self.clients[j].n for j in alive}
            sel, _util = select_participants(trust_scores, energy_ok, n_samples,
                                             staleness, {j: self.sigma2[j] for j in alive},
                                             self.cfg.trust, alive)
            return sel
        # baselines: uniform random K among energy-eligible
        pool = [j for j in alive if energy_ok.get(j, True)]
        k = min(self.cfg.trust.k_max, len(pool))
        return sorted(self.g.choice(pool, size=k, replace=False).tolist()) if k else []

    # ---------------- one round ----------------
    def run_round(self, t: int) -> Dict:
        cfg = self.cfg
        net_on = cfg.uav.enabled
        if net_on:
            self.network.step_mobility(cfg.uav.round_duration_s)

        # adaptive adversaries peek at their own trust before acting
        for j in self.attack.malicious:
            self.attack.update_adaptive(j, self.trust.overall(j), t,
                                        network=self.network if self.cfg.uav.enabled else None)

        invited = self._invite()
        deltas, n_map, arrivals = {}, {}, {}
        round_obs = {}  # uid -> physical observations for trust update

        for j in invited:
            participates = self.attack.will_participate(j, invited=True)
            u = next((x for x in self.network.uavs if x.uid == j), None) if net_on else None

            if not participates:
                round_obs[j] = dict(participated=False)
                continue

            # --- resource allocation (Algorithm 3) ---
            if net_on and cfg.method.name == "trustfl":
                f_hz, p_tx = allocate_resources(self.network, u, self.clients[j].n,
                                                cfg.train.local_epochs, self.num_params,
                                                cfg.resources, cfg.uav)
                f_hz = min(f_hz, u.f_actual_hz)  # physics: can't exceed real capability
            elif net_on:
                f_hz, p_tx = u.f_actual_hz, (cfg.uav.p_tx_min_w + cfg.uav.p_tx_max_w) / 2

            # --- local training (gradient attacks act here / on delta) ---
            hook = self.attack.label_hook(j, num_classes=self._num_classes())
            result = self.clients[j].local_update(self.model, malicious_hook=hook)
            delta = self.attack.corrupt_delta(j, result["delta"])

            # --- physical times, deadline, energy ---
            if net_on:
                committed = self.network.compute_time(u, self.clients[j].n,
                                                      cfg.train.local_epochs, u.f_reported_hz)
                actual = self.network.compute_time(u, self.clients[j].n,
                                                   cfg.train.local_epochs, f_hz)
                t_comm = self.network.comm_time(u, self.num_params, p_tx)
                t_comm += self.attack.comm_delay(j, cfg.uav.comm_deadline_s, t_comm)
                # comm deadline: slack-scaled to model size if comm_slack>0,
                # else the absolute fallback. Scales 0.02 Mbit -> 358 Mbit.
                if cfg.uav.comm_slack > 0:
                    comm_ddl = cfg.uav.comm_slack * self.network.nominal_comm_time(self.num_params)
                else:
                    comm_ddl = cfg.uav.comm_deadline_s
                comm_ok = (t_comm <= comm_ddl)
                # Compute-deadline: actual compute time must fit the round's
                # compute budget. A UAV that MISREPORTED its speed to get
                # selected now runs slower than promised and can miss this.
                # Fair rule applied to everyone -- honest-but-slow UAVs risk it
                # too, which is exactly the honest-unreliable-vs-malicious
                # distinction the trust mechanism is built to make.
                compute_ok = True
                if cfg.resources.compute_deadline:
                    # deadline scales with THIS UAV's workload: slack x the time
                    # an average-speed UAV would need for n_samples*epochs.
                    # Dataset-independent -- works for MNIST, FEMNIST, CIFAR.
                    f_med = 0.5 * (cfg.uav.f_min_hz + cfg.uav.f_max_hz)
                    cycles = cfg.uav.cycles_per_sample * self.clients[j].n * cfg.train.local_epochs
                    deadline = cfg.resources.compute_slack * (cycles / f_med)
                    compute_ok = (actual <= deadline)
                delivered = comm_ok and compute_ok
                e = self.network.round_energy(u, actual, f_hz, t_comm, p_tx,
                                              cfg.uav.round_duration_s)
                self.network.drain(u, e)
            else:
                committed = actual = t_comm = 0.0
                delivered = True

            round_obs[j] = dict(participated=True, committed=committed, actual=actual,
                                t_comm=t_comm, delivered=delivered)
            if delivered:
                deltas[j] = delta
                n_map[j] = result["n"]

        # idle UAVs still burn mobility/hover power (physics fix; config-gated)
        if net_on and cfg.uav.idle_energy_drain:
            active = {j for j, o in round_obs.items() if o.get("participated")}
            for u in self.network.uavs:
                if u.alive and u.uid not in active:
                    p_mob = (cfg.uav.p0_hover_w
                             + cfg.uav.p1_speed_w * float(np.linalg.norm(u.vel)) ** 2)
                    self.network.drain(u, p_mob * cfg.uav.round_duration_s)

        # --- gradient deviation for behavioral trust (Alg.1 line 5) ---
        # NOTE (documented deviation from the draft): the draft uses deviation
        # from the MEAN update; under strong attacks the mean is itself
        # corrupted, so honest nodes look deviant too and trust cannot
        # separate. We use the coordinate-wise MEDIAN as the reference
        # (robust for <50% outliers). Goes into the paper's Algorithm 1.
        # Consistency score in [0,1]: direction via cosine to the median
        # reference (scale-invariant; catches sign flips and random updates),
        # magnitude via a soft norm-ratio penalty (catches scaling attacks).
        grad_cons = {}
        if len(deltas) >= 2:
            ref = torch.stack(list(deltas.values())).median(0).values
            rnorm = ref.norm().clamp_min(1e-12)
            for j, d in deltas.items():
                cos = float(torch.nn.functional.cosine_similarity(d[None], ref[None]))
                ratio = float(d.norm() / rnorm)
                ratio = min(ratio, 1.0 / max(ratio, 1e-12))
                grad_cons[j] = max(0.0, (1.0 + cos) / 2.0) * ratio ** 0.5

        self._last_delta_norms = {j: float(d.norm()) for j, d in deltas.items()}

        # --- aggregation (timed for flagship claim #3) ---
        agg_delta, agg_wallclock = None, 0.0
        if deltas:
            t0 = time.perf_counter()
            agg_delta = self._aggregate(deltas, n_map, grad_cons)
            agg_wallclock = time.perf_counter() - t0
            sd = self.model.state_dict()
            new_flat = flatten_state(sd).cpu() + agg_delta
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in unflatten_like(new_flat, sd).items()})

        # --- trust updates (Algorithm 1) for every UAV, participant or not ---
        for j in list(self.clients):
            obs = round_obs.get(j)
            invited_j = j in invited
            if obs is None or not obs.get("participated"):
                self.trust.update(j, invited=invited_j, participated=False,
                                  committed_time=0, actual_time=0, grad_consistency=None,
                                  comm_time=0, comm_max=cfg.uav.comm_deadline_s,
                                  delivered=False, c_expected=0, c_actual=0)
            else:
                self.trust.update(j, invited=True, participated=True,
                                  committed_time=obs["committed"], actual_time=obs["actual"],
                                  grad_consistency=grad_cons.get(j),
                                  comm_time=obs["t_comm"], comm_max=cfg.uav.comm_deadline_s,
                                  delivered=obs["delivered"],
                                  c_expected=obs["committed"], c_actual=obs["actual"])

        # --- learnable trust weights (modernization #1) ---
        if cfg.trust.weight_mode == "learnable" and len(deltas) >= 2:
            self._update_trust_weights(deltas)

        return self._log_round(t, invited, deltas, agg_wallclock)

    # ---------------- helpers ----------------
    def _num_classes(self) -> int:
        from ..data import NUM_CLASSES
        return self.cfg.model.num_classes or NUM_CLASSES[self.cfg.data.dataset]

    def _aggregate(self, deltas, n_map, grad_cons=None) -> torch.Tensor:
        m = self.cfg.method
        self._last_influence = None
        info = {}
        if m.name == "trustfl":
            trust_scores = {j: self.trust.overall(j) for j in deltas}
            omega = aggregation_weights(list(deltas), trust_scores, n_map,
                                        t_min=self.cfg.trust.t_min,
                                        gate=self.cfg.trust.aggregation_gate)
            if self.cfg.trust.round_consistency_gate and grad_cons:
                # Historical trust decides WHO counts; the current round's
                # consistency decides HOW MUCH — closes the one-round EWMA
                # lag that on-off adaptive attackers exploit.
                omega = {j: w * max(grad_cons.get(j, 1.0), 0.0)
                         for j, w in omega.items()}
            z = sum(omega.values())
            if z > 1e-12:
                self._last_influence = {j: w / z for j, w in omega.items()}
            return AGGREGATORS["trust_weighted"](deltas, omega=omega)
        if m.name in ("krum", "multikrum"):
            multi_m = m.multikrum_m if m.name == "multikrum" else 1
            out = AGGREGATORS["krum"](deltas, f=m.krum_f, multi_m=multi_m, info=info)
            self._last_influence = info.get("influence")
            self._last_krum_nn = info.get("nearest_neighbor")
            return out
        if m.name == "trimmed_mean":
            return AGGREGATORS["trimmed_mean"](deltas, beta=m.trimmed_mean_beta)
        if m.name == "fltrust":
            out = AGGREGATORS["fltrust"](deltas, server_delta=self._server_update(), info=info)
            self._last_influence = info.get("influence")
            return out
        if m.name == "sophon":
            out = AGGREGATORS["sophon"](deltas, server_delta=self._server_update(), info=info)
            self._last_influence = info.get("influence")
            return out
        if m.name == "fedgreed":
            out = AGGREGATORS["fedgreed"](deltas, loss_eval=self._val_loss_with_delta,
                                          k=m.fedgreed_k, info=info)
            self._last_influence = info.get("influence")
            return out
        if m.name == "fedaa":
            if not hasattr(self, "_fedaa_state"):
                self._fedaa_state = {}
            out = AGGREGATORS["fedaa"](deltas, loss_eval=self._val_loss_with_delta,
                                       state=self._fedaa_state, info=info)
            self._last_influence = info.get("influence")
            return out
        if m.name == "median":
            return AGGREGATORS["median"](deltas)  # coordinate-wise: influence N/A
        # fedavg: influence = data-size weights
        tot = sum(n_map.values())
        self._last_influence = {j: n_map[j] / tot for j in deltas} if tot else None
        return AGGREGATORS["fedavg"](deltas, n=n_map)

    def _update_trust_weights(self, deltas) -> None:
        """Reward each trust dimension by how well it separates helpful from
        harmful updates this round: correlation between the dimension's score
        and the (negative) validation-loss change of each delta."""
        uids = list(deltas)
        # helpfulness: -delta_val_loss when applying each update alone
        base = self._eval(self.val_loader)["loss"]
        helpful = np.array([base - self._val_loss_with_delta(deltas[j]) for j in uids])
        dims = np.array([[self.trust.records[j].t_beh,
                          self.trust.records[j].t_comp,
                          self.trust.records[j].t_comm] for j in uids])
        reward = np.zeros(3)
        if helpful.std() > 1e-9:
            hz = (helpful - helpful.mean()) / helpful.std()
            for k in range(3):
                col = dims[:, k]
                if col.std() > 1e-9:
                    reward[k] = float(np.mean(hz * (col - col.mean()) / col.std()))
        self.trust.update_weights(reward)

    def _log_round(self, t, invited, deltas, agg_wallclock) -> Dict:
        cfg = self.cfg
        rec = {"round": t, "invited": invited, "delivered": sorted(deltas),
               "agg_time_s": agg_wallclock,
               "trust_weights": self.trust.weights.tolist()}
        norms = getattr(self, "_last_delta_norms", None)
        if norms:
            rec["delta_norms"] = {str(k): round(v, 4) for k, v in norms.items()}
        knn = getattr(self, "_last_krum_nn", None)
        if knn:
            rec["krum_nn"] = {str(k): v for k, v in knn.items()}
        infl = getattr(self, "_last_influence", None)
        if infl:
            rec["influence"] = {str(k): round(float(v), 5) for k, v in infl.items()}
            hm = self._high_mobility
            rec["highmob_influence_share"] = round(sum(v for k, v in infl.items() if k in hm), 5)
            rec["malicious_influence_share"] = round(
                sum(v for k, v in infl.items() if k in self._ground_truth_malicious), 5)
        # selection quality vs ground truth (server never uses this; logging only)
        mal = self._ground_truth_malicious
        if mal:
            sel = set(deltas)
            rec["malicious_selected"] = len(sel & mal)
            rec["honest_selected"] = len(sel - mal)
        if cfg.uav.enabled:
            rec["alive"] = sum(u.alive for u in self.network.uavs)
            rec["mean_energy_j"] = float(np.mean([u.energy_j for u in self.network.uavs]))
        if t % cfg.train.eval_every == 0 or t == cfg.train.rounds - 1:
            rec.update({f"test_{k}": v for k, v in self._eval(self.test_loader).items()})
            rec.update({f"val_{k}": v for k, v in self._eval(self.val_loader).items()})
        ts = self.trust.snapshot()
        rec["mean_trust_honest"] = float(np.mean([ts[j]["T"] for j in ts if j not in mal])) if len(ts) > len(mal) else None
        rec["mean_trust_malicious"] = float(np.mean([ts[j]["T"] for j in mal])) if mal else None
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
