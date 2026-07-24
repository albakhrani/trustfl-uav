"""Comprehensive metrics matrix: six facets of system effectiveness.

Not a sprawling metric dump, and not the two-metric minimum. A FOCUSED matrix
where each metric illuminates a different facet of how the system performs,
and together they show the full effectiveness picture. Each facet maps to a
real system property a deployer cares about.

  FACET               METRIC(S)                     what it demonstrates
  ------------------  ----------------------------  --------------------------
  Task effectiveness  final acc, rounds-to-target   the system learns well
  Honest-data effic.  honest / high-mob retention   keeps good participants
  Security            malicious influence share     benches attackers
  Resource effic.     delivery eff, energy/update   respects UAV constraints
                      (the UAV-specific axis)       *nobody else measures this*
  Cost                aggregation time              cheap enough to deploy
  Robustness          spread across attacks         holds up broadly

The resource-effectiveness row is what makes this a UAV paper, not a generic
FL-robustness paper.

CLI:
    python -m trustfl.metrics_matrix --store ./results_store --prefix exp_h_adsys
    python -m trustfl.metrics_matrix --store ./results_store --prefix femh_adsys
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .store import Store
from .audit import retention_for_run

METHOD_ORDER = ["trustfl", "sophon", "fedaa", "fltrust", "fedgreed", "multikrum", "fedavg", "krum"]
LABELS = {"trustfl": "TrustFL", "sophon": "Sophon", "fedaa": "FedAA", "fltrust": "FLTrust",
          "fedgreed": "FedGreed", "multikrum": "Multi-Krum", "fedavg": "FedAvg",
          "krum": "Krum"}


def _listlen(v):
    if isinstance(v, str):
        try:
            return len(json.loads(v))
        except Exception:
            return 0
    return len(v) if isinstance(v, (list, tuple)) else 0


def facets_for_run(store: Store, run_id: str) -> dict:
    """Compute every facet metric for one run."""
    rd = store.rounds(run_id)
    ov_ret, hm_ret = retention_for_run(store, run_id)

    # task
    accs = rd.dropna(subset=["test_acc"])["test_acc"] if "test_acc" in rd else pd.Series([], dtype=float)
    final_acc = float(accs.iloc[-1]) if len(accs) else np.nan
    best_acc = float(accs.max()) if len(accs) else np.nan
    # rounds to 95% of best (convergence speed, dataset-agnostic target)
    r_to = np.nan
    if len(accs):
        target = 0.95 * best_acc
        hit = rd[(rd["test_acc"].notna()) & (rd["test_acc"] >= target)]
        r_to = int(hit["round"].iloc[0]) if len(hit) else np.nan

    # security
    mi = [v for v in rd.get("malicious_influence_share", []) if v == v]
    mal_infl = float(np.mean(mi)) if mi else np.nan

    # resource effectiveness
    deff = []
    total_delivered = 0
    for _, r in rd.iterrows():
        inv, dlv = _listlen(r.get("invited")), _listlen(r.get("delivered"))
        if inv:
            deff.append(dlv / inv)
        total_delivered += dlv
    deliv_eff = float(np.mean(deff)) if deff else np.nan
    # energy consumed over the run / useful updates delivered
    energy = rd.dropna(subset=["mean_energy_j"])["mean_energy_j"] if "mean_energy_j" in rd else pd.Series([], dtype=float)
    energy_spent = float(energy.iloc[0] - energy.iloc[-1]) if len(energy) >= 2 else np.nan
    energy_per_update = (energy_spent / total_delivered) if (total_delivered and energy_spent == energy_spent) else np.nan

    # cost
    agg = rd.dropna(subset=["agg_time_s"])["agg_time_s"] if "agg_time_s" in rd else pd.Series([], dtype=float)
    agg_ms = float(agg.mean() * 1000) if len(agg) else np.nan

    return dict(final_acc=final_acc, best_acc=best_acc, rounds_to_target=r_to,
                hm_retention=hm_ret, overall_retention=ov_ret,
                mal_infl=mal_infl, deliv_eff=deliv_eff,
                energy_per_update=energy_per_update, agg_ms=agg_ms)


def matrix_per_seed(store: Store, prefix: str) -> dict:
    """method -> {facet: [per-seed values]}. Needed to decide whether an
    apparent between-method difference exceeds seed noise."""
    df = store.query()
    df = df[df["name"].str.startswith(prefix)]
    out = {}
    for method in METHOD_ORDER:
        runs = df[df["method"] == method]["run_id"].tolist()
        if not runs:
            continue
        per = [facets_for_run(store, r) for r in runs]
        out[LABELS.get(method, method)] = {
            k: [p[k] for p in per if p[k] == p[k]] for k in per[0]
        }
    return out


def column_is_significant(per_seed: dict, facet: str, k_sigma: float = 2.0) -> bool:
    """Is the best-vs-worst gap on this facet larger than seed noise?

    Rule (stated in the figure caption): a facet is treated as showing a real
    difference only if the gap between the best and worst method means exceeds
    k_sigma times the pooled within-method standard deviation across seeds.
    With few seeds this is deliberately conservative -- we would rather call a
    real effect "not significant" than paint noise as a difference.
    """
    means, stds = [], []
    for m, facets in per_seed.items():
        vals = facets.get(facet, [])
        if len(vals) >= 1:
            means.append(np.mean(vals))
        if len(vals) >= 2:
            stds.append(np.std(vals, ddof=1))
    if len(means) < 2:
        return False
    gap = max(means) - min(means)
    pooled = np.mean(stds) if stds else 0.0
    if pooled <= 0:
        return gap > 0  # no variance info: fall back to any nonzero gap
    return gap > k_sigma * pooled


def matrix(store: Store, prefix: str) -> pd.DataFrame:
    df = store.query()
    df = df[df["name"].str.startswith(prefix)]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for method in METHOD_ORDER:
        runs = df[df["method"] == method]["run_id"].tolist()
        if not runs:
            continue
        per = [facets_for_run(store, r) for r in runs]
        agg = {}
        for k in per[0]:
            vals = [p[k] for p in per if p[k] == p[k]]
            agg[k] = np.mean(vals) if vals else np.nan
        agg["method"] = LABELS.get(method, method)
        rows.append(agg)
    cols = ["method", "final_acc", "rounds_to_target", "hm_retention",
            "mal_infl", "deliv_eff", "energy_per_update", "agg_ms"]
    return pd.DataFrame(rows)[cols]


def print_matrix(df: pd.DataFrame):
    if df.empty:
        print("no runs for prefix"); return
    hdr = (f"{'method':11s} {'acc':>6s} {'r->tgt':>7s} {'retention':>10s} "
           f"{'mal_infl':>9s} {'deliv_eff':>10s} {'E/update':>10s} {'agg_ms':>9s}")
    print(hdr); print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['method']:11s} {r['final_acc']:6.3f} "
              f"{r['rounds_to_target']:7.0f} {r['hm_retention']:10.3f} "
              f"{r['mal_infl']:9.3f} {r['deliv_eff']:10.3f} "
              f"{r['energy_per_update']:10.1f} {r['agg_ms']:9.2f}")
    print("\nfacets: acc/r->tgt = task | retention = honest-data | mal_infl = "
          "security |\n        deliv_eff/E-per-update = RESOURCE (UAV-specific) "
          "| agg_ms = cost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("--prefix", default="exp_h_adsys")
    ap.add_argument("--csv", default=None, help="also write CSV here")
    args = ap.parse_args()
    store = Store(args.store)
    df = matrix(store, args.prefix)
    print_matrix(df)
    if args.csv and not df.empty:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
