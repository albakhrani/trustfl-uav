"""Summarize flagship experiment (a): honest-outlier retention.

Usage:
    python scripts/analyze_exp_a.py results/exp_a_trustfl results/exp_a_krum ...

For each results/<name> directory (containing seed*/metrics.jsonl +
groups.json), prints per-method, aggregated over seeds:
  - final test accuracy (mean +- std)
  - rounds to reach 0.95 accuracy (mean +- std; convergence speed)
  - mean influence share of high-mobility honest UAVs (the mechanism
    evidence: geometric methods should show this near zero, TrustFL should
    show it near the group's population share)
"""
import json
import os
import sys

import numpy as np


def load_seed(seed_dir):
    rows = [json.loads(l) for l in open(os.path.join(seed_dir, "metrics.jsonl"))]
    groups = json.load(open(os.path.join(seed_dir, "groups.json")))
    return rows, groups


def summarize(name_dir):
    finals, t95s, hm_shares = [], [], []
    hm_pop_share = None
    for sd in sorted(os.listdir(name_dir)):
        seed_dir = os.path.join(name_dir, sd)
        if not os.path.isdir(seed_dir):
            continue
        rows, groups = load_seed(seed_dir)
        evals = [(r["round"], r["test_acc"]) for r in rows if "test_acc" in r]
        if not evals:
            continue
        finals.append(evals[-1][1])
        t95 = next((rd for rd, a in evals if a >= 0.95), None)
        if t95 is not None:
            t95s.append(t95)
        shares = [r["highmob_influence_share"] for r in rows if "highmob_influence_share" in r]
        if shares:
            hm_shares.append(float(np.mean(shares)))
        n_uavs = max(max(groups["high_mobility"], default=0),
                     max(groups["malicious"], default=0)) + 1
        # population share needs total uav count; infer from config
        cfgp = os.path.join(seed_dir, "config_resolved.yaml")
        if os.path.exists(cfgp):
            import yaml
            c = yaml.safe_load(open(cfgp))
            n_uavs = c["uav"]["num_uavs"]
        hm_pop_share = len(groups["high_mobility"]) / n_uavs

    def ms(x):
        return f"{np.mean(x):.4f} +- {np.std(x):.4f}" if x else "n/a"

    print(f"\n{os.path.basename(name_dir)}  ({len(finals)} seeds)")
    print(f"  final acc:            {ms(finals)}")
    print(f"  rounds to 0.95:       {ms(t95s)}" + ("" if len(t95s) == len(finals)
          else f"  ({len(finals)-len(t95s)} seed(s) never reached 0.95)"))
    print(f"  high-mob infl share:  {ms(hm_shares)}"
          + (f"   (population share: {hm_pop_share:.3f})" if hm_pop_share else ""))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summarize(d)
