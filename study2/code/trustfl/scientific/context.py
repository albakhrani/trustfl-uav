"""Prospective hooks using legacy physical/attack equations with compact traces."""
from __future__ import annotations
import hashlib
import os
import math
import time
from pathlib import Path
import numpy as np
from ..comparison.context import CommonContext
from ..comparison.records import fresh_round as legacy_fresh_round
from ..comparison.policy import executable_invitations, realize_resources, resource_request
from ..trust.selection import allocate_resources, select_participants
from .randomness import CompactEventRandomness
from .instrumentation import Timing, MemorySampler
from .records import fresh_round, validate_round, write_json_once
from .policy import select_invitations, requested_resources, execution_order


class ScientificFailure(RuntimeError):
    def __init__(self, kind, message, evidence=None):
        super().__init__(message)
        self.kind=kind;self.evidence=evidence or {}


class ScientificContext(CommonContext):
    def __init__(self,bundle,arrays,cfg,writer,method,policy,step_budget,host_limit_bytes):
        # Do not relax CommonContext's two-method allowlist; this separate
        # context implements only the separately validated scientific schema.
        self.bundle=bundle;self.arrays=arrays;self.cfg=cfg;self.writer=writer
        self.method=method;self.policy=policy;self.step_budget=step_budget
        self.host_limit_bytes=host_limit_bytes
        self.rng=CompactEventRandomness(bundle['rng']['master_seed'],bundle['rng']['namespace'])
        self.device=cfg.train.device;self.run_dir=Path(writer.path)
        self.round_idx=-1;self.local_epoch=self.local_batch=0;self.current_client=-1
        self.record=None;self.server=None;self.method_meta=None
        self.query_purpose='scheduled_root';self.counters={};self.permutation_records=[]
        self.timer=Timing(self.device);self.memory=None
        (self.run_dir/'permutations').mkdir(exist_ok=False)

    def begin_round(self,t,network):
        if t!=self.round_idx+1:raise ValueError('sequential rounds required')
        self.round_started_ns=time.perf_counter_ns()
        self.round_idx=t;self.rng.begin_round(t)
        self.record=legacy_fresh_round(t,network)
        self.record['evaluation_required']=((t+1)%self.cfg.train.eval_every==0 or t==self.cfg.train.rounds-1)
        self.method_meta=None;self.counters={};self.permutation_records=[]
        self.timer=Timing(self.device);self.timer.start('round_inclusive',sync=True)
        self.memory=MemorySampler(self.device,self.host_limit_bytes);self.memory.start()
        self.local_epoch=self.local_batch=0;self.current_client=-1

    def count(self,name,n=1):
        self.counters[name]=self.counters.get(name,0)+n

    def event_value(self,event,uid,value,**counters):
        digest=self.rng.digest(event,self.round_idx,uid,**counters)
        a=np.asarray(value)
        if self.record is not None:
            self.record['events'].append({'event':event,'uid':uid,'key_digest':digest,
              'shape':list(a.shape),'dtype':str(a.dtype),
              'payload_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest(),
              'value':a.tolist() if a.size<=3 else None})
        return value

    def mobility_noise(self,uid,scale):
        return np.asarray(self.event_value('mobility_innovation',uid,
                    self.arrays['mobility'][self.round_idx,uid]),dtype=float)*scale

    def channel_fading(self,uid):
        return float(self.event_value('channel_unit_exponential_fading',uid,
                    self.arrays['channel'][self.round_idx,uid]))

    def attack_uniform(self,event,uid,*,size=None,**counters):
        if event in ('attack_participation','attack_delay_branch') and size is None:
            col=0 if event=='attack_participation' else 1
            value=float(self.arrays['attack_coins'][self.round_idx,uid,col])
        else:value=self.rng.numpy(event,self.round_idx,uid,**counters).random(size)
        return self.event_value(event,uid,value,**counters)

    def permutation(self,uid,epoch,n):
        a=self.rng.numpy('minibatch_permutation',self.round_idx,uid,epoch=epoch).permutation(n).astype('<i8')
        path=self.run_dir/'permutations'/f'r{self.round_idx:04d}-c{uid:04d}-e{epoch:03d}.npy'
        with path.open('xb') as f:
            np.save(f,a,allow_pickle=False)
            if self.bundle['protocol']['purpose']=='v5':f.flush();os.fsync(f.fileno())
        self.permutation_records.append({'round':self.round_idx,'client':uid,'epoch':epoch,
          'length':n,'path':str(path.relative_to(self.run_dir)),
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return a.tolist()

    def invite(self,network):
        with self.timer.region('selection'):
            priority=list(map(int,self.arrays['priority'][self.round_idx]))
            def native_selector():
                alive=[u.uid for u in network.uavs if u.alive]
                energy={u.uid:network.eligible_energy(u) for u in network.uavs}
                srv=self.server
                selected,_=select_participants({j:srv.trust.overall(j) for j in alive},energy,
                  {j:srv.clients[j].n for j in alive},{j:srv.trust.records[j].staleness for j in alive},
                  {j:srv.sigma2[j] for j in alive},self.cfg.trust,alive)
                return selected
            scheduled,executable,excluded=select_invitations(priority,network,self.cfg.trust.k_max,
                     self.policy,self.method,native_trustfl=native_selector)
            self.record.update(scheduled=sorted(scheduled),executable=sorted(executable),exclusions=excluded)
            self.record['invitation_priority_sha256']=hashlib.sha256(np.asarray(priority,dtype='<i8').tobytes()).hexdigest()
            self.record['selection_policy']=self.policy
            self.record['execution_order']=execution_order(self.policy,self.method,executable)
            return self.record['execution_order']

    def resources(self,uav):
        with self.timer.region('resource_rule'):
            srv=self.server
            def native_allocator():
                return allocate_resources(srv.network,uav,srv.clients[uav.uid].n,
                         self.cfg.train.local_epochs,srv.num_params,self.cfg.resources,self.cfg.uav)
            values=requested_resources(self.policy,self.method,self.cfg.uav,uav,
                                       native_allocation=native_allocator)
            self.record['clients'][str(uav.uid)].update(resources=values,resources_status='executed')
            return values['realized_frequency_hz'],values['realized_power_w']

    def coefficients_meta(self,meta):
        self.method_meta=meta
        alpha=meta['alpha']
        self.record.update(coefficients=alpha,coefficient_round=self.round_idx,
            coefficient_status='valid_positive_coefficients' if any(v>0 for v in alpha.values()) else 'valid_zero_coefficients',
            positive_coefficient_candidate_ids=sorted(int(k) for k,v in alpha.items() if v>0))

    def export_record(self,metrics,network):
        old=self.record
        r=fresh_round(self.round_idx,[u.uid for u in network.uavs])
        for name in ('scheduled','executable','exclusions','quarantined','adaptive'):
            r[name]=old[name]
        r['participating']=sorted(int(j) for j,c in old['clients'].items() if c['participated'])
        r['arrived']=sorted(old['physical_arrivals']);r['usable']=sorted(old['usable_candidates'])
        r['positive_contribution']=old['accepted_valid_ids']
        r['aggregate_applied']=old['aggregate_application_completed']
        r['aggregate_finite']=old['aggregate_finite'];r['post_model_finite']=old['post_model_finite']
        if self.method_meta:
            m=self.method_meta
            r.update(alpha=m['alpha'],q=m['q'],reconstruction=m['reconstruction'])
            r['root_term']={**m['root_term'],'fallback':bool(m['root_term']['coefficient'])}
            r['method_details']=m['details'];r['delta_norms']=m['delta_norms']
            r['actual_aggregation_device']=m['actual_tensor_device'];r['coefficient_dtype']=m['dtype']
            r['coefficient_status']='valid_positive' if any(a>0 for a in m['alpha'].values()) else 'valid_zero'
        for u in network.uavs:
            e=old['energy'][str(u.uid)];components=e['components']
            r['energy'][str(u.uid)]={'pre_j':e['pre_j'],'post_j':float(u.energy_j),
                'modeled_j':e['modeled_j'],'depletion_j':e['depletion_j'],
                **{k:math.fsum(c[k] for c in components) for k in ('compute_j','tx_j','mobility_j')}}
        r['evaluation_required']=old['evaluation_required']
        r['evaluations']={k.replace('_acc','_accuracy').replace('val_','root_'):v for k,v in metrics.items()
                           if k in ('test_acc','test_loss','val_acc','val_loss')}
        purposes=('test','scheduled_root','intrinsic_root_validation','trustfl_learner','shadow_attacker_feedback')
        for purpose in purposes:
            for unit in ('queries','examples','batches'):
                self.counters.setdefault(purpose+'_'+unit,0)
        for unit in ('queries','examples','batches'):
            self.counters['total_query_'+unit]=sum(self.counters[p+'_'+unit] for p in purposes)
        for name in ('client_sgd_steps','client_examples','client_batches','root_sgd_steps','root_training_examples','root_training_batches'):
            self.counters.setdefault(name,0)
        r['counters']=dict(self.counters);r['timers']=dict(self.timer.values)
        for name in ('selection','resource_rule','reference_consistency','intrinsic_method_inclusive','intrinsic_root_training','trust_update','trustfl_learner_inclusive','shadow_attacker_feedback_inclusive','scheduled_evaluation','local_training','round_inclusive')+tuple('query_'+p for p in purposes):
            r['timers'].setdefault(name,0.0)
        r['execution_order']=old.get('execution_order',[])
        r['resources']={j:c['resources'] for j,c in old['clients'].items() if 'resources' in c}
        r['clients']=old['clients'];r['events']=old['events'];r['rng']=self.rng.round_summary()
        r['permutations']=self.permutation_records;r['policy']=self.policy;r['method']=self.method
        r['priority_sha256']=old.get('invitation_priority_sha256')
        r['trust_after']=old.get('trust_after',{});r['trust_weights']=old.get('trust_weights',[])
        r['guarded_events']=old.get('guarded_events',[])
        return r

    def finish_round(self,metrics,network):
        if self.timer and 'round_inclusive' in self.timer.active:self.timer.stop('round_inclusive',sync=True)
        publication_started=time.perf_counter_ns()
        r=self.export_record(metrics,network)
        if self.memory:r['memory']=self.memory.stop();self.memory=None
        total=r['timers'].get('round_inclusive',0.0)
        r['timers']['server_work_before_publication_excluding_local_and_scheduled_eval']=max(0.0,total-
            r['timers'].get('local_training',0.0)-r['timers'].get('scheduled_evaluation',0.0))
        r['timing_scope']='named regions may nest; inclusive totals are not sums; round/server work excludes serialization, validation, durable publication and memory sampler stop; publication timed separately in immutable round timing sidecar; excludes physical waiting in sequential simulation'
        errors=validate_round(r, self.writer.metadata)
        if errors:
            self.writer.record_failure_round(r)
            if not r['post_model_finite']:kind='numerical_model';ev={'nonfinite_model_recorded':True}
            elif old_nonfinite(self.record):kind='numerical_aggregate';ev={'nonfinite_aggregate_recorded':True}
            elif any('nonfinite_evaluation' in e for e in errors):kind='numerical_evaluation';ev={'nonfinite_evaluation_recorded':True}
            else:kind='missing_corrupt_evidence';ev={'evidence_integrity_failure_recorded':True}
            raise ScientificFailure(kind,';'.join(errors),ev)
        self.writer.write_round(r)
        duration=(time.perf_counter_ns()-publication_started)/1e9
        write_json_once(self.run_dir/f'round-timing-{self.round_idx:05d}.json',{
            'round':self.round_idx,'publication_s':duration,
            'inclusive_wall_s':(time.perf_counter_ns()-self.round_started_ns)/1e9,
            'scope':'publication_s covers export, sampler stop, validation, JSON serialization and atomic round publication; inclusive_wall_s covers begin_round through round publication; both exclude this sidecar publication'})
        self.final_record=r
        return r


def old_nonfinite(record):
    return record.get('aggregate_finite') is False
