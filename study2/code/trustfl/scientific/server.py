"""Prospective server adapter; native equations/attacker transitions are reused.

No import executes training. All optimizer steps have explicit checked budgets.
The copied round body is instrumented, with the native source preserved intact.
"""
from __future__ import annotations
import copy
import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from ..fl.server import Server
from ..fl.client import Client,flatten_state,unflatten_like
from ..trust.selection import allocate_resources
from .methods import aggregate
from .context import ScientificFailure


class ScientificClient(Client):
    def __init__(self,*args,context,**kwargs):
        # The allowlisted real-data transforms are deterministic; do not invoke
        # engineering per-sample RNG scopes for millions of deterministic reads.
        super().__init__(*args,context=None,**kwargs)
        self.context=context

    def local_update(self,model,malicious_hook=None):
        ctx=self.context;cfg=self.cfg
        with ctx.timer.region('local_training',sync=True):
            local=copy.deepcopy(model).to(self.device);local.train()
            initial=flatten_state(model.state_dict()).cpu()
            opt=torch.optim.SGD(local.parameters(),lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
            total_loss=0.0;batches=0
            for epoch in range(cfg.local_epochs):
                ctx.current_client=self.uid;ctx.local_epoch=epoch
                indices=ctx.permutation(self.uid,epoch,self.n)
                loader=DataLoader(Subset(self.subset,indices),batch_size=cfg.batch_size,
                   shuffle=False,num_workers=0,drop_last=False,
                   generator=ctx.rng.torch_generator('minibatch_permutation',ctx.round_idx,self.uid,epoch=epoch,operation='loader'))
                if hasattr(ctx, 'wrap_training_loader'):
                    loader = ctx.wrap_training_loader(loader, 'client')
                for batch,(x,y) in enumerate(loader):
                    ctx.local_batch=batch
                    x,y=x.to(self.device),y.to(self.device)
                    if malicious_hook is not None:y=malicious_hook('label',y)
                    ctx.memory.check_budget()
                    with ctx.rng.scope('model_forward_stochastic_operations',ctx.round_idx,self.uid,
                             epoch=epoch,batch=batch,operation='local',device=self.device):
                        opt.zero_grad();loss=F.cross_entropy(local(x),y);loss.backward()
                        ctx.step_budget.before_step('client')
                        opt.step();ctx.step_budget.after_step('client')
                    ctx.count('client_sgd_steps');ctx.count('client_examples',len(y));ctx.count('client_batches')
                    total_loss+=float(loss.detach());batches+=1
            return {'uid':self.uid,'delta':flatten_state(local.state_dict()).cpu()-initial,
                    'n':self.n,'train_loss':total_loss/max(batches,1)}


class ScientificServer(Server):
    def __init__(self,*args,context,**kwargs):
        super().__init__(*args,context=None,**kwargs)
        self.context=context;context.server=self
        self.method=context.method;self._fd_state={}
        # Preserve deterministic transforms, but isolate DataLoader seed draws.
        self.val_loader.generator=context.rng.torch_generator('server_validation_and_evaluation',-1,operation='loader:root')
        self.test_loader.generator=context.rng.torch_generator('server_validation_and_evaluation',-1,operation='loader:test')

    @torch.no_grad()
    def _eval(self,loader):
        ctx=self.context
        purpose='test' if loader is self.test_loader else ctx.query_purpose
        ctx.count(purpose+'_queries')
        loader.generator=ctx.rng.torch_generator('server_validation_and_evaluation',ctx.round_idx,operation='loader:'+purpose)
        correct=total=0;loss_sum=0.0
        with ctx.timer.region('query_'+purpose, sync=True),ctx.rng.scope('server_validation_and_evaluation',ctx.round_idx,operation=purpose,device=self.device):
            self.model.eval()
            for x,y in loader:
                ctx.memory.check_budget() if ctx.memory else None
                x,y=x.to(self.device),y.to(self.device);out=self.model(x)
                loss_sum+=float(F.cross_entropy(out,y,reduction='sum'))
                correct+=int((out.argmax(1)==y).sum());total+=len(y)
                ctx.count(purpose+'_batches');ctx.count(purpose+'_examples',len(y))
        result={'acc':correct/max(total,1),'loss':loss_sum/max(total,1)}
        if not all(np.isfinite(v) for v in result.values()):
            if ctx.record is not None:ctx.record['failed_query']={'purpose':purpose,**result}
            raise ScientificFailure('numerical_evaluation','nonfinite '+purpose,{'nonfinite_evaluation_recorded':True})
        return result

    def _server_update(self):
        ctx=self.context
        with ctx.timer.region('intrinsic_root_training',sync=True):
            local=copy.deepcopy(self.model).to(self.device);local.train()
            opt=torch.optim.SGD(local.parameters(),lr=self.cfg.train.lr,momentum=self.cfg.train.momentum)
            self.val_loader.generator=ctx.rng.torch_generator('server_validation_and_evaluation',ctx.round_idx,operation='loader:root_training')
            for epoch in range(self.cfg.train.local_epochs):
                loader = ctx.wrap_training_loader(self.val_loader, 'root') if hasattr(ctx, 'wrap_training_loader') else self.val_loader
                for batch,(x,y) in enumerate(loader):
                    ctx.memory.check_budget()
                    x,y=x.to(self.device),y.to(self.device)
                    with ctx.rng.scope('model_forward_stochastic_operations',ctx.round_idx,-1,
                                      epoch=epoch,batch=batch,operation='root_training',device=self.device):
                        opt.zero_grad();F.cross_entropy(local(x),y).backward()
                        ctx.step_budget.before_step('root');opt.step();ctx.step_budget.after_step('root')
                    ctx.count('root_sgd_steps');ctx.count('root_training_examples',len(y));ctx.count('root_training_batches')
            return flatten_state(local.state_dict()).cpu()-flatten_state(self.model.state_dict()).cpu()

    def _aggregate(self,deltas,n_map,grad_cons=None):
        ctx=self.context;ctx.query_purpose='intrinsic_root_validation'
        with ctx.timer.region('intrinsic_method_inclusive',sync=True):
            root=self._server_update() if self.method=='fltrust_declared_root_fallback' else None
            out,meta=aggregate(self.method,deltas,n_map,cfg=self.cfg,
                trust_scores={j:self.trust.overall(j) for j in deltas},grad_cons=grad_cons,
                root_delta=root,loss_eval=self._val_loss_with_delta,state=self._fd_state)
            ctx.coefficients_meta(meta)
            self._last_influence={int(j):a for j,a in meta['alpha'].items()}
            if not meta['aggregate_finite']:
                ctx.record['aggregate_finite']=False
                raise ScientificFailure('numerical_aggregate','guarded nonfinite aggregate',{'nonfinite_aggregate_recorded':True})
            if not meta['reconstruction']['passed']:
                raise ScientificFailure('missing_corrupt_evidence','fixed coefficient reconstruction failed',{'evidence_integrity_failure_recorded':True})
            return out

    def _update_trust_weights(self,deltas):
        ctx=self.context
        ctx.query_purpose='trustfl_learner' if self.method=='trustfl' else 'shadow_attacker_feedback'
        with ctx.timer.region(ctx.query_purpose+'_inclusive',sync=True):
            return super()._update_trust_weights(deltas)

    def _log_round(self,t,invited,deltas,agg_wallclock):
        ctx=self.context
        result={'round':t,'agg_time_s':agg_wallclock}
        if ctx.record['evaluation_required']:
            ctx.query_purpose='scheduled_root'
            with ctx.timer.region('scheduled_evaluation',sync=True):
                result.update({'test_'+k:v for k,v in self._eval(self.test_loader).items()})
                result.update({'val_'+k:v for k,v in self._eval(self.val_loader).items()})
        return result

    def run_round(self, t: int) -> Dict:
        cfg = self.cfg
        net_on = cfg.uav.enabled
        if self.context is not None:
            self.context.begin_round(t, self.network)
            self._last_influence = None
            self._last_delta_norms = {}
            self._last_krum_nn = None
            self._last_nonfinite_dropped = []
        if net_on:
            self.network.step_mobility(cfg.uav.round_duration_s)

        # adaptive adversaries peek at their own trust before acting
        for j in self.attack.malicious:
            if self.context is not None:
                prior = {"own_trust_observed": float(self.trust.overall(j)),
                         "prior_striking": bool(self.attack._adaptive_attacking.get(j, False)),
                         "prior_mode": self.attack._sys_mode.get(j),
                         "prior_rotation": self.attack._sys_rotation.get(j),
                         "available_information": ["own_method_diagnostic_trust", "round", "own_attack_state", "own_true_capability", "attack_configuration"],
                         "absolute_delay_deadline_s": float(cfg.uav.comm_deadline_s)}
            self.attack.update_adaptive(j, self.trust.overall(j), t,
                                        network=self.network if self.cfg.uav.enabled else None)
            if self.context is not None:
                prior.update(striking=bool(self.attack._adaptive_attacking.get(j, False)),
                             mode=self.attack._sys_mode.get(j), rotation=self.attack._sys_rotation.get(j),
                             probe_active=t < cfg.attack.adaptive_probe_rounds)
                state = next((u for u in self.network.uavs if u.uid == j), None)
                prior["reported_frequency_after_hz"] = float(state.f_reported_hz) if state is not None else None
                self.context.record["adaptive"][str(j)] = prior

        invited = self._invite()
        deltas, n_map, arrivals = {}, {}, {}
        round_obs = {}  # uid -> physical observations for trust update
        nonfinite_dropped = []  # delivered updates quarantined for NaN/Inf

        for j in invited:
            participates = self.attack.will_participate(j, invited=True)
            u = next((x for x in self.network.uavs if x.uid == j), None) if net_on else None

            if self.context is not None:
                self.context.record["clients"][str(j)] = {"participated": bool(participates),
                    "exclusion_reason": None if participates else "attack_refusal",
                    "physical_arrival": False, "usable_delivery": False, "quarantined": False,
                    "resources_status": "pending" if participates else "not_executed_refusal"}
            if not participates:
                round_obs[j] = dict(participated=False)
                continue

            # --- resource allocation (Algorithm 3) ---
            if net_on and self.context is not None:
                f_hz, p_tx = self.context.resources(u)
            elif net_on and cfg.method.name == "trustfl":
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
                if self.context is not None:
                    base_t_comm = t_comm
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
                if self.context is not None:
                    self.context.record["clients"][str(j)].update(
                        committed_s=float(committed), actual_s=float(actual), base_comm_s=float(base_t_comm),
                        attack_delay_s=float(t_comm - base_t_comm), total_comm_s=float(t_comm),
                        absolute_delay_deadline_s=float(cfg.uav.comm_deadline_s),
                        effective_comm_deadline_s=float(comm_ddl), compute_ok=bool(compute_ok),
                        comm_ok=bool(comm_ok), samples=int(self.clients[j].n),
                        local_epochs=int(cfg.train.local_epochs), num_params=int(self.num_params))
            else:
                committed = actual = t_comm = 0.0
                delivered = True

            # A delivered payload containing NaN/Inf (diverged local training,
            # overflowed attack payload) is unusable: one such delta poisons
            # the median reference, every cosine against it, and through
            # grad_cons the trust EWMAs of ALL participants. A ZERO-NORM delta
            # is equally unusable (no gradient information, degenerate
            # direction against the cosine reference). Quarantine both: the
            # update reaches neither the consistency reference nor
            # aggregation, and trust scores the round as maximally suspicious
            # (all three instantaneous scores 0 — see TrustManager.update).
            # Finite nonzero deltas — however large — are untouched.
            if self.context is not None:
                physical_arrival = bool(delivered)
                self.context.record["clients"][str(j)]["physical_arrival"] = physical_arrival
                if physical_arrival:
                    self.context.record["physical_arrivals"].append(j)
            quarantined = delivered and (not bool(torch.isfinite(delta).all())
                                         or not bool(delta.any()))
            if quarantined:
                delivered = False
                nonfinite_dropped.append(j)
                if self.context is not None:
                    reason = "nonfinite_payload" if not bool(torch.isfinite(delta).all()) else "zero_payload"
                    self.context.record["quarantined"][str(j)] = reason
                    self.context.record["guarded_events"].append(reason + ":" + str(j))
            if self.context is not None:
                self.context.record["clients"][str(j)].update(usable_delivery=bool(delivered),
                                                           quarantined=bool(quarantined))
            round_obs[j] = dict(participated=True, committed=committed, actual=actual,
                                t_comm=t_comm, delivered=delivered, quarantined=quarantined)
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
                    if self.context is not None:
                        self.context.energy_components(u.uid, 0.0, 0.0,
                            p_mob * cfg.uav.round_duration_s, idle=True)
                    self.network.drain(u, p_mob * cfg.uav.round_duration_s)

        self.context.timer.start('reference_consistency')
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

        self.context.timer.stop('reference_consistency')
        self._last_delta_norms = {j: float(d.norm()) for j, d in deltas.items()}
        if self.context is not None:
            self.context.record["usable_candidates"] = sorted(deltas)
            self.context.record["delta_norms"] = dict(self._last_delta_norms)

        # --- aggregation (timed for flagship claim #3) ---
        agg_delta, agg_wallclock = None, 0.0
        self._last_agg_nonfinite = False
        if deltas:
            t0 = time.perf_counter()
            agg_delta = self._aggregate(deltas, n_map, grad_cons)
            agg_wallclock = time.perf_counter() - t0
            # last-resort guard: inputs are finite by construction, but an
            # aggregator can still overflow in float32; a non-finite aggregate
            # would destroy the global model irreversibly, so skip the round.
            if self.context is not None:
                self.context.record.update(aggregate_available=True,
                    aggregate_finite=bool(torch.isfinite(agg_delta).all()),
                    aggregate_norm=float(agg_delta.norm()))
                errors = self.context.record["coefficient_status"] in ("missing", "nonfinite_coefficients")
                if errors:
                    self.context.record["errors"].append("invalid_actual_coefficients")
                    self.context.finish_round({}, self.network)
            if bool(torch.isfinite(agg_delta).all()):
                sd = self.model.state_dict()
                if self.context is not None:
                    before_flat = flatten_state(sd).cpu()
                    self.context.record["aggregate_application_attempted"] = True
                new_flat = flatten_state(sd).cpu() + agg_delta
                self.model.load_state_dict(
                    {k: v.to(self.device) for k, v in unflatten_like(new_flat, sd).items()})
                if self.context is not None:
                    after_flat = flatten_state(self.model.state_dict()).cpu()
                    finite_model = bool(torch.isfinite(after_flat).all())
                    self.context.record.update(aggregate_application_completed=True,
                        post_model_finite=finite_model, model_changed=not torch.equal(before_flat, after_flat))
                    if finite_model:
                        self.context.record["accepted_valid_ids"] = list(
                            self.context.record["positive_coefficient_candidate_ids"])
                    else:
                        self.context.record["errors"].append("nonfinite_model_after_application")
                        self.context.finish_round({}, self.network)
            else:
                self._last_agg_nonfinite = True
                if self.context is not None:
                    self.context.record["guarded_events"].append("nonfinite_aggregate")

        self.context.timer.start('trust_update')
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
                                  c_expected=obs["committed"], c_actual=obs["actual"],
                                  quarantined=obs.get("quarantined", False))

        self.context.timer.stop('trust_update')
        # --- learnable trust weights (modernization #1) ---
        if cfg.trust.weight_mode == "learnable" and len(deltas) >= 2:
            self._update_trust_weights(deltas)

        self._last_nonfinite_dropped = nonfinite_dropped
        rec = self._log_round(t, invited, deltas, agg_wallclock)
        if self.context is not None:
            self.context.record["trust_after"] = self.trust.snapshot()
            self.context.record["trust_weights"] = self.trust.weights.tolist()
            self.context.finish_round(rec, self.network)
        return rec
