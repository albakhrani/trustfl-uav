"""Audit: why do high-mobility (uids 0-8) updates have smaller norms?

Recomputes the exact data partitions for exp_a seeds (same code path, same
seeds -> identical partitions) and tests:
  T1: do uids 0-8 get systematically smaller datasets? (positional artifact
      in dirichlet_partition -> would be a BUG to fix)
  T2: does per-UAV dataset size explain the logged update norms?
      (Pearson r between n_j and mean delta norm from exp_a_krum_diag)

Usage (from project root; needs MNIST already in ./data):
    python scripts/partition_audit.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from trustfl.config import ExperimentConfig            # noqa: E402
from trustfl.data import (load_dataset, split_server_val,  # noqa: E402
                          dirichlet_partition)

CFG = "configs/exp_a_honest_outlier.yaml"
DIAG = "results/exp_a_krum_diag"
N_UAVS, HM = 30, set(range(9))


def partition_sizes(seed):
    cfg = ExperimentConfig.load(CFG)
    train, _ = load_dataset(cfg.data, seed)
    train, _ = split_server_val(train, cfg.data.server_val_fraction, seed)
    parts = dirichlet_partition(train, N_UAVS, cfg.data.dirichlet_alpha,
                                cfg.data.min_samples_per_client, seed)
    return np.array([len(p) for p in parts])


def mean_norms(seed_dir):
    sums, cts = {}, {}
    for line in open(os.path.join(seed_dir, "metrics.jsonl")):
        r = json.loads(line)
        for k, v in r.get("delta_norms", {}).items():
            j = int(k)
            sums[j] = sums.get(j, 0.0) + v
            cts[j] = cts.get(j, 0) + 1
    return {j: sums[j] / cts[j] for j in sums}


print("T1: partition sizes by group (uids 0-8 = HM positions)")
ratios = []
for seed in range(5):
    n = partition_sizes(seed)
    hm_mean, other_mean = n[:9].mean(), n[9:].mean()
    ratios.append(hm_mean / other_mean)
    print(f"  seed{seed}: HM mean n={hm_mean:7.1f}  other mean n={other_mean:7.1f}"
          f"  ratio={hm_mean/other_mean:.3f}")
print(f"  => mean ratio over 5 seeds: {np.mean(ratios):.3f}"
      f"  (1.0 = no positional artifact)")

if os.path.isdir(DIAG):
    print("\nT2: does dataset size explain update norms? (exp_a_krum_diag)")
    for sd in sorted(os.listdir(DIAG)):
        seed_dir = os.path.join(DIAG, sd)
        if not os.path.isdir(seed_dir):
            continue
        seed = int(sd.replace("seed", ""))
        n = partition_sizes(seed)
        norms = mean_norms(seed_dir)
        uids = sorted(norms)
        x = np.array([n[j] for j in uids], dtype=float)
        y = np.array([norms[j] for j in uids])
        r = np.corrcoef(x, y)[0, 1]
        hm_n = [norms[j] for j in uids if j in HM]
        ot_n = [norms[j] for j in uids if j not in HM]
        print(f"  {sd}: corr(n_j, norm_j) = {r:+.3f}   "
              f"mean norm HM={np.mean(hm_n):.3f} other={np.mean(ot_n):.3f}")
    print("  interpretation: strong positive corr + T1 ratio < 1 -> size-driven"
          "\n                  weak corr -> size is NOT the cause; report back")