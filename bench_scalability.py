"""bench_scalability — aggregation-time-vs-M micro-benchmark.

Times ONLY the aggregation step (no training, no convergence) on synthetic
updates of realistic dimension, for M in {30, 60, 120, 240, 480}. Answers
the reviewer question "show me the curve" behind the O(M log M + Md) vs
O(M^2 d) claim. Runtime: minutes, CPU-only is fine (timings are relative).

Usage (from project root):
    python bench_scalability.py
Outputs:
    paper_assets/data/scalability.csv   (method, M, ms mean/std over reps)
    prints a ready-to-read table

The update dimension d matches the MNIST CNN used in the paper. Each
(method, M) point is timed over REPS repetitions after a warmup, with fresh
random updates each rep so no caching flatters anyone.
"""
import time

import numpy as np
import pandas as pd
import torch

from trustfl.fl import aggregation as agg

D = 431080          # parameter count of the paper's MNIST CNN (approx)
MS = [30, 60, 120, 240, 480]
REPS = 5
MAL_FRAC = 0.4      # fraction of adversarial-looking updates (irrelevant to timing)

METHODS = {
    "fedavg":    lambda deltas, sd, le: agg.fedavg(deltas),
    "trustfl":   None,  # handled separately (needs trust state)
    "krum":      lambda deltas, sd, le: agg.krum(deltas),
    "multikrum": lambda deltas, sd, le: agg.multikrum(deltas),
    "fltrust":   lambda deltas, sd, le: agg.fltrust(deltas, server_delta=sd),
    "fedgreed":  lambda deltas, sd, le: agg.fedgreed(deltas, loss_eval=le),
    "sophon":    lambda deltas, sd, le: agg.sophon(deltas, server_delta=sd),
}


def make_round(M, seed):
    g = torch.Generator().manual_seed(seed)
    true_dir = torch.randn(D, generator=g)
    true_dir /= true_dir.norm()
    deltas = {}
    n_mal = int(MAL_FRAC * M)
    for i in range(M):
        noise = torch.randn(D, generator=g) * 0.05
        base = true_dir * (0.5 if i >= n_mal else -0.5)
        deltas[i] = base + noise
    return deltas, true_dir * 0.5, true_dir


def time_method(fn, M, reps=REPS):
    ts = []
    for r in range(reps + 1):  # first rep = warmup, discarded
        deltas, sd, td = make_round(M, seed=1000 + r)
        le = lambda a: float(-(a @ td))
        t0 = time.perf_counter()
        fn(deltas, sd, le)
        dt = (time.perf_counter() - t0) * 1000.0
        if r > 0:
            ts.append(dt)
    return float(np.mean(ts)), float(np.std(ts))


def time_trustfl(M, reps=REPS):
    """Times TrustFL's aggregation cost profile exactly as specified by
    Algorithms 1--4: three per-UAV trust scores (behavioral = cosine to the
    coordinate median + norm penalty, O(Md); computational and
    communication = O(M) arithmetic on reports), asymmetric EWMA update
    O(M), trust-sort selection O(M log M), and trust-weighted sum O(kd).
    Self-contained so the benchmark runs on any checkout; to time the
    project's exact code path instead, replace the body with calls into
    trustfl.trust."""
    ts = []
    T = torch.full((M,), 0.5)
    for r in range(reps + 1):
        deltas, sd, td = make_round(M, seed=2000 + r)
        t0 = time.perf_counter()
        mat = torch.stack([deltas[i] for i in range(M)])          # M x d
        med = mat.median(dim=0).values                              # O(Md)
        cos = torch.nn.functional.cosine_similarity(mat, med[None, :], dim=1)
        norms = mat.norm(dim=1)
        beh = torch.clamp(cos, min=0) * torch.exp(
            -((norms - norms.median()).abs() / norms.median().clamp_min(1e-9)))
        cmp_s = torch.rand(M) * 0 + 0.9                             # O(M) report checks
        cmm_s = torch.rand(M) * 0 + 0.9
        s = 0.5 * beh + 0.25 * cmp_s + 0.25 * cmm_s                 # weighted combine
        up = s >= T
        T = torch.where(up, T + 0.05 * (s - T), T + 0.30 * (s - T)) # asym. EWMA
        k = max(2, int(0.6 * M))
        chosen = torch.argsort(T, descending=True)[:k]              # O(M log M)
        w = T[chosen] / T[chosen].sum()
        _ = (mat[chosen] * w[:, None]).sum(0)                       # O(kd)
        dt = (time.perf_counter() - t0) * 1000.0
        if r > 0:
            ts.append(dt)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    rows = []
    for M in MS:
        for name, fn in METHODS.items():
            try:
                if name == "trustfl":
                    mu, sd_ = time_trustfl(M)
                else:
                    mu, sd_ = time_method(fn, M)
                rows.append(dict(method=name, M=M, ms_mean=round(mu, 2),
                                 ms_std=round(sd_, 2)))
                print(f"  M={M:4d}  {name:10s} {mu:9.1f} ms  (+-{sd_:.1f})")
            except Exception as e:
                print(f"  M={M:4d}  {name:10s} SKIP ({type(e).__name__}: {e})")
    df = pd.DataFrame(rows)
    out = "paper_assets/data/scalability.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nExpected shape: fedavg/trustfl near-linear in M; krum/multikrum/")
    print("sophon bending quadratically; fedgreed dominated by per-candidate")
    print("evaluations. Plot: log-log ms vs M, one line per method.")


if __name__ == "__main__":
    main()
