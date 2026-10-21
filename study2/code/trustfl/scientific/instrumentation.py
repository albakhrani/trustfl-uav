"""Explicit execution counters and clock/memory boundaries, usable without Torch."""
from __future__ import annotations
from contextlib import contextmanager
import math
import copy
import sys
import time
import threading


class StepBudget:
    def __init__(self, maximum):
        if type(maximum) is not int or maximum < 0:
            raise ValueError('Nonnegative fixed optimizer-step budget required')
        self.maximum = maximum
        self.reserved = self.completed = 0
        self.by_kind = {}

    def before_step(self, kind):
        if self.reserved >= self.maximum:
            raise RuntimeError('optimizer_step_budget_exhausted BEFORE optimizer.step')
        self.reserved += 1
        self.by_kind.setdefault(kind, {'reserved': 0, 'completed': 0})['reserved'] += 1

    def after_step(self, kind):
        self.completed += 1
        self.by_kind[kind]['completed'] += 1

    def snapshot(self):
        return {'maximum': self.maximum, 'reserved': self.reserved,
                'completed': self.completed, 'by_kind': copy.deepcopy(self.by_kind)}


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


class Timing:
    """Named inclusive/nested timings; never sum overlapping regions as total."""
    def __init__(self, device='cpu'):
        self.device = device
        self.values = {}
        self.active = {}

    def start(self, name, sync=False):
        if name in self.active:
            raise RuntimeError('overlapping same-name timer: '+name)
        if sync:
            synchronize(self.device)
        self.active[name] = time.perf_counter_ns()

    def stop(self, name, sync=False):
        if sync:
            synchronize(self.device)
        elapsed = (time.perf_counter_ns() - self.active.pop(name)) / 1e9
        self.values[name] = self.values.get(name, 0.0) + elapsed
        return elapsed

    @contextmanager
    def region(self, name, sync=False):
        self.start(name, sync)
        try:
            yield
        finally:
            self.stop(name, sync)


def process_memory():
    if sys.platform == 'win32':
        from .windows_memory import process_memory as windows_process_memory
        return windows_process_memory()
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak=int(peak if sys.platform=='darwin' else peak*1024)
    # This platform's stdlib supplies a lifetime peak, not exact current RSS.
    return {'rss_bytes':None,'lifetime_peak_rss_bytes':peak}


class MemorySampler:
    def __init__(self, device, host_limit_bytes, vram_fraction=.8):
        self.device=device;self.host_limit_bytes=host_limit_bytes;self.vram_fraction=vram_fraction
        self.peak=0;self.samples=0;self.errors=[];self.stopped=threading.Event();self.thread=None

    def start(self):
        self.initial=process_memory()
        if str(self.device).startswith('cuda'):
            import torch
            torch.cuda.reset_peak_memory_stats(self.device)
        def poll():
            while not self.stopped.wait(.01):
                try:
                    m=process_memory();value=m['rss_bytes']
                    if value is not None:self.peak=max(self.peak,value)
                    self.samples+=1
                except Exception as exc:
                    self.errors.append(type(exc).__name__+':'+str(exc));break
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()

    def check_budget(self):
        current=process_memory()
        if current['rss_bytes'] is not None and current['rss_bytes']>self.host_limit_bytes:
            raise MemoryError('declared host memory ceiling exceeded')
        if str(self.device).startswith('cuda'):
            import torch
            free,total=torch.cuda.mem_get_info(self.device)
            if total-free>self.vram_fraction*total:
                # Device-total includes unrelated processes: cause remains ambiguous
                # unless PyTorch's own allocation independently exceeds its budget.
                if torch.cuda.memory_allocated(self.device)>self.vram_fraction*total:
                    raise MemoryError('declared model GPU allocation ceiling exceeded')
                raise RuntimeError('external_device_pressure_or_reserved_memory; cause ambiguous')

    def stop(self):
        self.stopped.set()
        if self.thread:self.thread.join(timeout=1)
        out={'start':self.initial,'end':process_memory(),'sampled_phase_peak_rss_bytes':self.peak or None,
             'sampling_interval_s':.01,'samples':self.samples,'errors':self.errors,
             'scope':'process lifetime peak differs from approximate sampled phase peak'}
        if str(self.device).startswith('cuda'):
            import torch
            out['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(self.device)
            out['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved(self.device)
            out['cuda_device']=str(self.device)
        return out
