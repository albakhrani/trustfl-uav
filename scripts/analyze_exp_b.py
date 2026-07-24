"""Analyze experiment (b): system-dimension attacks vs defenses.

Usage:
    python scripts/analyze_exp_b.py results/exp_b_*

Per run directory (aggregated over seeds) prints:
  acc_final / acc_last10   accuracy (final round; mean of last 10 evals)
  mal_sel_frac             mean fraction of delivered updates that are
                           malicious (ground truth; server never used it)
  mal_sel_late             same, over the LAST 20 rounds only -> lock-in
                           check: a working defense should drive this to ~0
  mal_infl                 mean malicious influence share in aggregation
  deliv_eff                delivered / invited (round efficiency)
"""
import glob
import json
import os
import sys

import numpy as np


def seed_stats(seed_dir):
    rows = [json.loads(l) for l in open(os.path.join(seed_dir, "metrics.jsonl"))]
    accs = [(r["round"], r["test_acc"]) for r in rows if "test_acc" in r]
    if not accs:
        return None  # crashed/empty run; skip
    acc_final = accs[-1][1]
    acc_last10 = float(np.mean([a for _, a in accs[-10:]]))
    msel, mlate, minfl, deff = [], [], [], []
    n_rounds = rows[-1]["round"]
    for r in rows:
        ms, hs = r.get("malicious_selected"), r.get("honest_selected")
        if ms is not None and (ms + hs) > 0:
            frac = ms / (ms + hs)
            msel.append(frac)
            if r["round"] > n_rounds - 20:
                mlate.append(frac)
        if r.get("malicious_influence_share") is not None:
            minfl.append(r["malicious_influence_share"])
        inv, dlv = len(r.get("invited", [])), len(r.get("delivered", []))
        if inv:
            deff.append(dlv / inv)
    return dict(acc_final=acc_final, acc_last10=acc_last10,
                mal_sel=np.mean(msel) if msel else float("nan"),
                mal_late=np.mean(mlate) if mlate else float("nan"),
                mal_infl=np.mean(minfl) if minfl else float("nan"),
                deff=np.mean(deff) if deff else float("nan"))


def summarize(name_dir):
    stats = []
    for sd in sorted(os.listdir(name_dir)):
        p = os.path.join(name_dir, sd)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "metrics.jsonl")):
            st = seed_stats(p)
            if st is not None:
                stats.append(st)
    if not stats:
        return
    agg = {k: (np.mean([s[k] for s in stats]), np.std([s[k] for s in stats]))
           for k in stats[0]}
    name = os.path.basename(name_dir)
    print(f"{name:44s} acc={agg['acc_final'][0]:.4f}±{agg['acc_final'][1]:.4f} "
          f"last10={agg['acc_last10'][0]:.4f} "
          f"mal_sel={agg['mal_sel'][0]:.3f} mal_late={agg['mal_late'][0]:.3f} "
          f"mal_infl={agg['mal_infl'][0]:.3f} deliv_eff={agg['deff'][0]:.3f}")


if __name__ == "__main__":
    paths = []
    for a in sys.argv[1:]:
        paths.extend(glob.glob(a))
    print(f"{'run':44s} {'metrics (mean over seeds)'}")
    for d in sorted(paths):
        if os.path.isdir(d):
            summarize(d)
    print("\nread: mal_sel = malicious fraction of delivered updates (0.30 = "
          "no filtering);\n      mal_late = same over final 20 rounds (lock-in "
          "check: should -> 0 if defense adapts);\n      mal_infl = malicious "
          "share of aggregation weight.")
