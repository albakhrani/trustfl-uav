"""Diagnose influence concentration in experiment (a) logs.

Usage:
    python scripts/diagnose_influence.py results/exp_a_krum results/exp_a_trustfl ...

For each method directory, aggregated over seeds, prints:
  1. per-UAV cumulative influence, top-10, tagged [HM] high-mobility / [--],
     with per-seed winner concentration (does ONE UAV dominate, or the group?)
  2. per-group delivery rate (delivered rounds / invited rounds) -- tests
     whether high-mobility UAVs deliver less often
  3. per-group mean influence per delivered round -- influence conditional
     on actually delivering (separates "chosen when present" from
     "present more often")

Reads metrics.jsonl + groups.json written by the v2 server. No reruns needed.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np


def analyze_seed(seed_dir):
    rows = [json.loads(l) for l in open(os.path.join(seed_dir, "metrics.jsonl"))]
    groups = json.load(open(os.path.join(seed_dir, "groups.json")))
    hm = set(groups["high_mobility"])

    infl_sum = defaultdict(float)      # uid -> cumulative influence
    invited_ct = defaultdict(int)
    delivered_ct = defaultdict(int)
    infl_when_delivered = defaultdict(list)

    for r in rows:
        inv = r.get("invited", [])
        dlv = set(r.get("delivered", []))
        infl = {int(k): v for k, v in r.get("influence", {}).items()}
        for j in inv:
            invited_ct[j] += 1
            if j in dlv:
                delivered_ct[j] += 1
                infl_when_delivered[j].append(infl.get(j, 0.0))
        for j, w in infl.items():
            infl_sum[j] += w

    return hm, infl_sum, invited_ct, delivered_ct, infl_when_delivered


def summarize(name_dir):
    print(f"\n===== {os.path.basename(name_dir)} =====")
    seed_dirs = [os.path.join(name_dir, d) for d in sorted(os.listdir(name_dir))
                 if os.path.isdir(os.path.join(name_dir, d))]

    for sd in seed_dirs:
        hm, infl, inv, dlv, iwd = analyze_seed(sd)
        total = sum(infl.values()) or 1.0
        top = sorted(infl.items(), key=lambda kv: -kv[1])[:10]
        top_str = "  ".join(
            f"{'HM' if j in hm else '--'}u{j}:{v/total:.2f}" for j, v in top)
        print(f"  {os.path.basename(sd)} top10 influence: {top_str}")

        # group stats
        def grp(d, uids):
            vals = [d[j] for j in uids if j in d]
            return vals

        hm_del = [dlv[j] / inv[j] for j in hm if inv.get(j)]
        lo_del = [dlv[j] / inv[j] for j in dlv if j not in hm and inv.get(j)]
        hm_iwd = [np.mean(iwd[j]) for j in hm if iwd.get(j)]
        lo_iwd = [np.mean(iwd[j]) for j in iwd if j not in hm and iwd.get(j)]
        print(f"      delivery rate  HM={np.mean(hm_del):.3f}  other={np.mean(lo_del):.3f}"
              f"   |   infl/delivered-round  HM={np.mean(hm_iwd):.4f}  other={np.mean(lo_iwd):.4f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summarize(d)