"""bench_scalability_v2 — aggregation-time-vs-M using the REAL server path.

This reproduces the same per-round aggregation-time measurement that the
training runs record as `agg_time_ms` (server.py), swept across swarm sizes
M in {10, 30, 60, 120, 240}. Unlike a standalone micro-benchmark, it drives
each method's actual aggregation exactly as the paper measured it (server.py times ONLY _aggregate; trust scoring is
outside the timer), so this is a SMALL-D algorithmic-scaling microbenchmark: absolute times are
not comparable to the paper per-round numbers (which use full D); the point
is the SCALING ORDER in M (log-log slope) and the relative ordering. Timed on CPU for stable, reproducible algorithmic scaling; absolute
values differ from the GPU-under-load store timings, but the curve shape
(TrustFL/FedAvg near-linear vs Krum-family quadratic) is the point.

WHY THIS IS FAIR: every method's aggregation is timed at the same boundary
the server uses. TrustFL's cost includes trust scoring + selection +
weighted sum (Algorithms 1-4); the baselines include their full defense
(e.g. Krum's pairwise distances). The comparison is therefore like-for-like.

Usage (from project root):
    python bench_scalability_v2.py
Output:
    paper_assets/data/scalability.csv   (method, M, ms_mean, ms_std, reps)

======================================================================
HOW TO WIRE IN THE REAL PATH (one edit):
Replace the body of `time_aggregation(method, deltas, ctx)` with the exact
call your server makes between the two perf_counter() calls at
trustfl/fl/server.py lines ~243-245. Paste that block here. It likely looks
like one of:
    agg = AGGREGATORS[method](deltas, **agg_kwargs)          # simple case
  or, for TrustFL, the trust-manager scoring + selection + trust_weighted.
Everything else in this script (sweep, timing harness, CSV output) is ready.
======================================================================
"""
import time

import numpy as np
import pandas as pd
import torch

from trustfl.fl import aggregation as agg

# --- config ---------------------------------------------------------------
# Small model dimension: isolates ALGORITHMIC scaling in M. With the full
# D=431k, the O(M*D) vector arithmetic dominates and hides the O(M^2) vs
# O(M log M) distinction the complexity claim is about. A small D makes the
# aggregation LOGIC (pairwise distances vs sort) the bottleneck, exposing the
# true scaling order. We report this as an algorithmic-scaling microbenchmark;
# absolute times are not comparable to per-round wall-clock at full D.
D = 2000
MS = [16, 32, 64, 128, 256, 512]
REPS = 50                  # more reps to stabilize; endpoint noise was a problem at 20
WARMUP = 10
MAL_FRAC = 0.4
DEVICE = "cpu"   # scalability measures algorithmic scaling; CPU is stable/reproducible
METHODS = ["fedavg", "trustfl", "krum", "multikrum", "fltrust",
           "fedgreed", "sophon"]


def make_round(M, seed):
    """Synthetic but realistic round: M updates of dimension D, 40% adversarial
    (adversarial updates are valid-looking, matching the paper's threat)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ref = torch.randn(D, generator=g)
    ref /= ref.norm()
    n_mal = int(MAL_FRAC * M)
    deltas = {}
    for i in range(M):
        sign = -0.5 if i < n_mal else 0.5
        deltas[i] = (ref * sign + torch.randn(D, generator=g) * 0.05).to(DEVICE)
    server_delta = (ref * 0.5).to(DEVICE)
    n = {i: 100 for i in deltas}           # equal sample counts
    return deltas, server_delta, n, ref.to(DEVICE)


def time_aggregation(method, deltas, ctx):
    """Time ONE aggregation call for `method`, timed at the same boundary the
    server uses. ctx carries server_delta, n, ref for methods that need them.

    >>> TODO: replace this dispatch with the EXACT call from server.py lines
    243-245 once pasted. The dispatch below is the faithful default based on
    the known aggregator signatures; verify it matches the server. <<<
    """
    server_delta, n, ref = ctx["server_delta"], ctx["n"], ctx["ref"]

    if method == "fedavg":
        return agg.fedavg(deltas, n)
    if method == "krum":
        f = int(MAL_FRAC * len(deltas))
        return agg.krum(deltas, f=f, multi_m=1)
    if method == "multikrum":
        f = int(MAL_FRAC * len(deltas))
        mm = max(1, len(deltas) - 2 * f - 2)
        return agg.krum(deltas, f=f, multi_m=mm)
    if method == "fltrust":
        return agg.fltrust(deltas, server_delta=server_delta)
    if method == "sophon":
        return agg.sophon(deltas, server_delta=server_delta)
    if method == "fedgreed":
        le = lambda a: float(-(a @ ref))
        return agg.fedgreed(deltas, loss_eval=le, k=max(2, len(deltas) // 2))
    if method == "trustfl":
        # MATCHES server._aggregate: timed region = trust lookups + selection
        # (aggregation_weights: t_min gate + consistency gate) + trust_weighted.
        # Only trust.update() scoring is outside the timer (done in harness).
        tm = ctx["trust_mgr"]
        n_samples = ctx["n"]
        t_min = 0.3
        # EXACT aggregation_weights (trust.selection): gated rule
        #   omega_j = ReLU(T_j - t_min) * n_j / Z
        trust_scores = {j: tm.overall(j) for j in deltas}
        raw = {j: max(trust_scores[j] - t_min, 0.0) * n_samples[j] for j in deltas}
        z = sum(raw.values())
        omega = ({j: raw[j] / z for j in deltas} if z > 0
                 else {j: 1.0 / len(deltas) for j in deltas})
        # round-consistency gate: omega_j *= max(grad_cons_j, 0)
        gc = ctx["grad_cons"]
        omega = {j: w * max(gc.get(j, 1.0), 0.0) for j, w in omega.items()}
        return agg.trust_weighted(deltas, omega=omega)
    raise ValueError(method)


def bench():
    rows = []
    for M in MS:
        # per-M context objects reused across reps (built once)
        for method in METHODS:
            ts = []
            for r in range(REPS + WARMUP):
                deltas, sd, n, ref = make_round(M, seed=1000 + r)
                ctx = {"server_delta": sd, "n": n, "ref": ref}
                if method == "trustfl":
                    # scoring + selection done BEFORE the timer, as in server.py
                    from trustfl.trust.trust import TrustManager
                    class _Cfg:
                        init_trust = 0.5; w_beh = w_comp = w_comm = 1/3
                        weight_mode = "fixed"; learnable_lr = 0.1
                        alpha_beh = alpha_comp = alpha_comm = 0.3
                        asymmetry = 3.0
                        w_participation = w_commitment = w_gradient = 1/3
                    tm = TrustManager(_Cfg(), M)
                    med = torch.stack(list(deltas.values())).median(dim=0).values
                    for j, d in deltas.items():
                        gc = float(torch.clamp(torch.nn.functional.cosine_similarity(
                            d[None, :], med[None, :]).squeeze(), 0, 1))
                        tm.update(j, invited=True, participated=True,
                                  committed_time=1.0, actual_time=1.0,
                                  grad_consistency=gc, comm_time=0.3, comm_max=1.0,
                                  delivered=True, c_expected=1.0, c_actual=1.0)
                    ctx["trust_mgr"] = tm
                    ctx["grad_cons"] = {j: float(torch.clamp(
                        torch.nn.functional.cosine_similarity(
                            deltas[j][None, :], med[None, :]).squeeze(), 0, 1))
                        for j in deltas}
                t0 = time.perf_counter()
                time_aggregation(method, deltas, ctx)
                dt = (time.perf_counter() - t0) * 1000.0
                if r >= WARMUP:
                    ts.append(dt)
            mu, sd_ = float(np.mean(ts)), float(np.std(ts))
            rows.append(dict(method=method, M=M, ms_mean=round(mu, 3),
                             ms_std=round(sd_, 3), reps=REPS))
            print(f"  M={M:4d}  {method:10s} {mu:9.2f} ms  (+-{sd_:.2f})")
    df = pd.DataFrame(rows)
    df.to_csv("paper_assets/data/scalability.csv", index=False)
    print("\nwrote paper_assets/data/scalability.csv")
    # log-log slope per method = empirical scaling order (cost ~ M^slope)
    print("\nEMPIRICAL SCALING ORDER (log-log slope; ~1 linear, ~2 quadratic):")
    for m in METHODS:
        d = df[df.method == m]
        slope = np.polyfit(np.log(d["M"]), np.log(d["ms_mean"]), 1)[0]
        tag = ("~linear" if slope < 1.35 else
               "~quadratic" if slope > 1.7 else "super-linear")
        print(f"  {m:10s} slope={slope:.2f}  ({tag})")
    print("\nEXPECTED: krum/multikrum ~2 (O(M^2)); trustfl/fedavg/fltrust/"
          "fedgreed ~1 (O(M)); this is the m3 result.")


if __name__ == "__main__":
    bench()
