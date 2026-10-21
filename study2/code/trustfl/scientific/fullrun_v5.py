"""Narrow full-run allowance hooks. Import uses only the standard library.

No method equations, loaders' order, or random scopes are replaced. Numerical
work remains in the existing scientific runner, behind its execution gate.
"""
import copy
import math
import hashlib
from pathlib import Path
import time
from .instrumentation import StepBudget


class BrokerStepBudget(StepBudget):
    def __init__(self, broker, maximum):
        super().__init__(maximum)
        self.broker = broker
        self.pending = {}

    def before_step(self, kind):
        if kind not in ('client', 'root') or kind in self.pending:
            raise ValueError('invalid or outstanding optimizer reservation')
        if self.reserved >= self.maximum:
            raise RuntimeError('optimizer allowance exhausted')
        token = self.broker.reserve(kind + '_step', detail={'purpose': kind + '_training'})
        self.pending[kind] = token
        self.reserved += 1
        self.by_kind.setdefault(kind, {'reserved': 0, 'completed': 0})['reserved'] += 1

    def after_step(self, kind):
        if kind not in self.pending:
            raise ValueError('optimizer completion without reservation')
        self.broker.acknowledge(self.pending[kind])
        del self.pending[kind]
        self.completed += 1
        self.by_kind[kind]['completed'] += 1


class BudgetLoader:
    def __init__(self, loader, broker, kind):
        self.loader, self.broker, self.kind = loader, broker, kind

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        # The size of the next batch is known from the deterministic loader;
        # reserve before reading it, not after a model/data operation.
        remaining = len(self.loader.dataset)
        iterator = iter(self.loader)
        for index in range(len(self.loader)):
            count = min(int(self.loader.batch_size), remaining)
            detail = {'purpose': self.kind + '_training', 'examples': count, 'batches': 1}
            if self.kind == 'root':
                detail['first_of_pass'] = index == 0
            token = self.broker.reserve(self.kind + '_batch', detail=detail)
            batch = next(iterator)
            if len(batch[1]) != count:
                raise ValueError('actual batch differs from reserved retained population')
            yield batch
            self.broker.acknowledge(token)
            remaining -= count
        if remaining != 0:
            raise ValueError('loader did not cover retained population')


class FullRunAdapter:
    """One source-bound manifest cell; parent remains the budget authority."""
    def __init__(self, broker, cell, binding, resolved_config):
        self.broker, self.cell, self.binding = broker, copy.deepcopy(cell), copy.deepcopy(binding)
        self.resolved_config=copy.deepcopy(resolved_config)
        self.started = time.perf_counter()
        self.round_token = None
        self.bound = False

    def validate(self, bundle, method, policy, step_budget, scenario):
        p = bundle['protocol']
        expected = (self.cell['dataset'], self.cell['scenario_seed'], self.cell['condition'], self.cell['method'], 'common')
        actual = (p['dataset'], p['scenario_seed'], p['condition'], method, policy)
        if actual != expected or p['purpose'] != 'v5' or not isinstance(step_budget, BrokerStepBudget):
            raise ValueError('v5 fixed cell/allowance mismatch')
        if step_budget.broker is not self.broker or self.cell['rounds'] != 60:
            raise ValueError('v5 broker or horizon mismatch')
        self.scenario=Path(scenario)
        self.sealed_payloads=dict(bundle["payloads"])
        self.sealed_payloads["scenario.json"]=hashlib.sha256((self.scenario/"scenario.json").read_bytes()).hexdigest()
        self.bound = True
        return {'campaign_v5': self.binding, 'fixed_attempt_id_v5': self.cell['id'], 'claim_eligible': True}

    def install(self, server, context):
        if not self.bound:
            raise ValueError('adapter was not validated')
        if server.cfg.to_dict()!=self.resolved_config:raise ValueError('runtime config differs from predeclared fixed cell')
        counts = context.bundle['indices']['counts']['clients']
        if [server.clients[j].n for j in range(30)] != counts:
            raise ValueError('client n differs from sealed retained indices')
        sigmas=context.bundle['indices']['selection']['sigma2']
        if any(not math.isclose(float(server.sigma2[j]),sigmas[j],rel_tol=1e-12,abs_tol=1e-12) for j in range(30)):
            raise ValueError('sigma2 differs from sealed retained population')
        self.client_setup_s = 0.0
        self.client_calls = 0
        self.local_started = None
        self.first_loader = False
        def wrap_loader(loader, kind):
            if kind == 'client' and self.first_loader:
                self.client_setup_s += time.perf_counter() - self.local_started
                self.first_loader = False
            return BudgetLoader(loader, self.broker, kind)
        context.wrap_training_loader = wrap_loader
        for client in server.clients.values():
            original_local = client.local_update
            def measured_local(*args, _original=original_local, **kwargs):
                self.local_started = time.perf_counter()
                self.first_loader = True
                self.client_calls += 1
                return _original(*args, **kwargs)
            client.local_update = measured_local
        original_export = context.export_record
        def exported(*args, **kwargs):
            record = original_export(*args, **kwargs)
            record['v5_accounting'] = {'actual_client_counts': counts,
                'client_setup_before_first_loader_s_cumulative': self.client_setup_s,
                'client_calls_cumulative': self.client_calls,
                'campaign_protocol_sha256': self.binding['protocol_sha256']}
            return record
        context.export_record = exported
        original_eval = server._eval
        def guarded_eval(loader):
            kind = 'test_query' if loader is server.test_loader else 'root_query'
            purpose = 'test' if kind == 'test_query' else context.query_purpose
            started = time.perf_counter()
            token = self.broker.reserve(kind, detail={'purpose': purpose, 'examples': len(loader.dataset), 'batches': len(loader)})
            value = original_eval(loader)
            self.broker.acknowledge(token)
            label='v5_eval_inclusive_'+purpose
            context.timer.values[label]=context.timer.values.get(label,0.)+time.perf_counter()-started
            return value
        server._eval = guarded_eval

    def before_round(self, round_id):
        for name,expected in self.sealed_payloads.items():
            if hashlib.sha256((self.scenario/name).read_bytes()).hexdigest()!=expected:
                raise ValueError("sealed scenario reference changed before round")
        if self.round_token is not None:
            raise ValueError('unacknowledged prior round')
        self.round_token = self.broker.reserve('round', detail={'round': round_id})

    def after_round(self, round_id):
        if self.round_token is None:
            raise ValueError('unreserved round')
        self.broker.acknowledge(self.round_token)
        self.round_token = None
