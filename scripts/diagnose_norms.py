"""Test WHY Krum concentrates on high-mobility UAVs.

Usage:
    python scripts/diagnose_norms.py results/exp_a_krum_diag [more dirs...]

Requires runs made with the norm-logging server (delta_norms / krum_nn in
metrics.jsonl). Per seed, reports:

  H1 (norm bias): winner's norm rank among delivered updates (0 = smallest,
     1 = largest; 0.5 = no bias), plus mean update norm HM vs other.
  H2 (pairing): how often the winner's nearest neighbor is in the same
     group (HM-HM pairs vs expected under independence).
"""
import json
import os
import sys

import numpy as np


def summarize(name_dir):
    print(f"\n===== {os.path.basename(name_dir)} =====")
    for sd in sorted(os.listdir(name_dir)):
        seed_dir = os.path.join(name_dir, sd)
        if not os.path.isdir(seed_dir):
            continue
        rows = [json.loads(l) for l in open(os.path.join(seed_dir, "metrics.jsonl"))]
        hm = set(json.load(open(os.path.join(seed_dir, "groups.json")))["high_mobility"])

        win_ranks, hmhm_pairs, cross_pairs = [], 0, 0
        hm_norms, other_norms = [], []
        win_is_hm = 0
        n_rounds = 0

        for r in rows:
            norms = {int(k): v for k, v in r.get("delta_norms", {}).items()}
            infl = {int(k): v for k, v in r.get("influence", {}).items()}
            if not norms or not infl:
                continue
            n_rounds += 1
            for j, v in norms.items():
                (hm_norms if j in hm else other_norms).append(v)
            winner = max(infl, key=infl.get)
            ranked = sorted(norms, key=norms.get)
            win_ranks.append(ranked.index(winner) / max(len(ranked) - 1, 1))
            win_is_hm += winner in hm
            nn = {int(k): v for k, v in r.get("krum_nn", {}).items()}
            if winner in nn:
                if (winner in hm) and (nn[winner] in hm):
                    hmhm_pairs += 1
                elif (winner in hm) != (nn[winner] in hm):
                    cross_pairs += 1

        if not n_rounds:
            print(f"  {sd}: no norm logs (rerun with the patched server)")
            continue
        print(f"  {sd}: rounds={n_rounds}  winner_norm_rank={np.mean(win_ranks):.3f} "
              f"(0.5=unbiased)  winner_HM_rate={win_is_hm/n_rounds:.2f}")
        print(f"      mean update norm  HM={np.mean(hm_norms):.3f}  "
              f"other={np.mean(other_norms):.3f}  "
              f"ratio={np.mean(hm_norms)/max(np.mean(other_norms),1e-9):.3f}")
        if hmhm_pairs + cross_pairs:
            print(f"      winner-NN pairs: HM-HM={hmhm_pairs}  mixed={cross_pairs}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summarize(d)
