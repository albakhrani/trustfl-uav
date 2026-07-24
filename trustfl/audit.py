"""Thesis-metric audit (Option C) — the decisive evaluation.

Computes the THREE metrics that operationalize the flagship claims, and only
those. No sprawling matrix: each metric is the direct measure of one claim,
and the two core metrics measure exactly what the thesis says competing
methods cannot see.

  M1  Honest-data retention  (Claim 1): fraction of INVITED honest updates
      that survive into aggregation. Reported overall and for HIGH-MOBILITY
      honest UAVs specifically (the ones Krum discards). Geometric methods
      that reject honest outliers -> low retention.

  M2  Malicious influence share (Claims 2 & 4): aggregation weight captured
      by attackers. Gradient-only defenses blind to system-dimension attacks
      -> attackers keep weight -> high share. Reported as a mean and, for the
      adaptive case, as a per-round TRAJECTORY.

  M3  Per-round aggregation cost (Claim 3): wall-clock ms. Already stored;
      surfaced here for the cost-vs-robustness Pareto view.

Retention is computed from `invited`, `delivered`, `influence`, and
groups.json (honest = not malicious; high-mobility from groups). A run's
retention for a round = |honest invited with POSITIVE influence weight| /
|honest invited|.  (Positive weight == survived into the aggregate; zero-weight
entries are discards.) Rounds where no update receives positive weight are
skipped — retention is undefined when nothing was aggregated.

CLI:
    python -m trustfl.audit --store ./results_store claim1
    python -m trustfl.audit --store ./results_store claim2
    python -m trustfl.audit --store ./results_store trajectory --attack adaptive_system
    python -m trustfl.audit --store ./results_store all
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .store import Store


def _groups(store: Store, run_id: str):
    """Recover (malicious set, high_mobility set) for a run from its seed dir.
    groups.json lives in the original results tree, keyed by run's seed dir."""
    name, seed = run_id.split("/seed")
    # try common results locations
    for base in ("./results", store.root):
        gp = os.path.join(base, name, f"seed{seed}", "groups.json")
        if os.path.exists(gp):
            g = json.load(open(gp))
            return set(g.get("malicious", [])), set(g.get("high_mobility", []))
    return set(), set()


def _parse_listcol(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return v if isinstance(v, list) else []


def _parse_dictcol(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return v if isinstance(v, dict) else {}


def retention_for_run(store: Store, run_id: str):
    """Return (overall_retention, highmob_retention) averaged over rounds
    where honest UAVs were invited."""
    mal, hm = _groups(store, run_id)
    df = store.rounds(run_id)
    over, hmob = [], []
    for _, r in df.iterrows():
        invited = set(_parse_listcol(r.get("invited")))
        infl = _parse_dictcol(r.get("influence"))
        # Rounds where nothing received positive weight (e.g. cosine-fallback
        # rounds returning an all-zero influence dict) aggregate nothing from
        # clients: retention is undefined there, exactly as mal_influence is
        # NaN when tot_infl == 0. Skip them.
        if not infl or sum(float(w) for w in infl.values()) == 0:
            continue
        survived = {int(k) for k, w in infl.items() if float(w) > 0}
        honest_inv = {j for j in invited if j not in mal}
        if honest_inv:
            over.append(len(honest_inv & survived) / len(honest_inv))
        hm_honest_inv = {j for j in honest_inv if j in hm}
        if hm_honest_inv:
            hmob.append(len(hm_honest_inv & survived) / len(hm_honest_inv))
    return (float(np.mean(over)) if over else np.nan,
            float(np.mean(hmob)) if hmob else np.nan)


def _canon(store: Store, where=""):
    """Query canonical runs only if paper_tag exists."""
    try:
        base = "paper_tag IS NOT NULL AND paper_tag != 'exploratory'"
        w = f"({where}) AND {base}" if where else base
        return store.query(w)
    except Exception:
        return store.query(where)


class Audit:
    def __init__(self, store_root):
        self.s = Store(store_root)

    # ---- Claim 1: honest-data retention (the metric Krum fails) ----
    def claim1(self):
        df = _canon(self.s, "attack_type='none' AND high_mob_frac > 0")
        if df.empty:
            print("no honest-outlier runs"); return
        rows = []
        for _, r in df.iterrows():
            ov, hm = retention_for_run(self.s, r["run_id"])
            rows.append((r["dataset"], r["method"], r["seed"], ov, hm, r["final_acc"]))
        d = pd.DataFrame(rows, columns=["dataset", "method", "seed",
                                        "retention", "hm_retention", "final_acc"])
        g = d.groupby(["dataset", "method"]).agg(
            retention=("retention", "mean"),
            hm_retention=("hm_retention", "mean"),
            final_acc=("final_acc", "mean"), n=("seed", "count")).reset_index()
        print("=== CLAIM 1: honest-data retention (higher = keeps honest UAVs) ===")
        print(g.round(4).to_string(index=False))
        print("\nread: hm_retention is the decisive number — fraction of honest "
              "HIGH-MOBILITY\n      updates that survive aggregation. Geometric "
              "methods (krum) should be\n      low; trustfl high. Accuracy is the "
              "downstream consequence.")
        return g

    # ---- Claims 2 & 4: malicious influence share (the metric FLTrust fails) ----
    def claim2(self, attack=None):
        cond = "malicious_frac > 0"
        if attack:
            cond += f" AND attack_type='{attack}'"
        df = _canon(self.s, cond)
        if df.empty:
            print(f"no runs for {attack or 'any attack'}"); return
        g = df.groupby(["attack_type", "method"]).agg(
            mal_infl=("mal_infl_mean", "mean"),
            mal_infl_std=("mal_infl_mean", "std"),
            final_acc=("final_acc", "mean"),
            deliv_eff=("deliv_eff", "mean"), n=("seed", "count")).reset_index()
        print("=== CLAIMS 2&4: malicious influence share (lower = benches attackers) ===")
        print(g.round(4).to_string(index=False))
        print("\nread: mal_infl is the decisive number — aggregation weight "
              "captured by\n      attackers. Gradient-only defenses (fltrust, "
              "fedgreed) are blind to\n      system attacks -> high share; trustfl "
              "benches them -> low share.\n      pop. share at 30%% malicious "
              "≈ 0.30; below that = active suppression.")
        return g

    # ---- Claim 4: influence trajectory under adaptive adversary ----
    def trajectory(self, attack="adaptive_system"):
        df = _canon(self.s, f"attack_type='{attack}' AND malicious_frac > 0")
        if df.empty:
            print(f"no {attack} runs"); return
        print(f"=== CLAIM 4: malicious-influence TRAJECTORY under {attack} ===")
        print("(mean malicious influence share per 10-round window, by method)\n")
        for method in sorted(df["method"].unique()):
            runs = df[df["method"] == method]["run_id"].tolist()
            # average the per-round malicious_influence_share across seeds
            series = {}
            for rid in runs:
                rd = self.s.rounds(rid)
                if "malicious_influence_share" not in rd.columns:
                    continue
                for _, r in rd.iterrows():
                    v = r.get("malicious_influence_share")
                    if pd.notna(v):
                        series.setdefault(int(r["round"]) // 10, []).append(float(v))
            windows = sorted(series)
            vals = [f"{np.mean(series[w]):.2f}" for w in windows]
            print(f"  {method:10s} " + " ".join(vals))
        print("\nread: does the attacker's influence stay suppressed as it "
              "maneuvers?\n      trustfl should hold low across windows; blind "
              "methods let it persist near 0.30.")

    def all(self):
        self.claim1(); print()
        self.claim2(); print()
        self.trajectory("adaptive_system")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("--attack", default=None)
    ap.add_argument("cmd", choices=["claim1", "claim2", "trajectory", "all"])
    args = ap.parse_args()
    a = Audit(args.store)
    if args.cmd == "claim1":
        a.claim1()
    elif args.cmd == "claim2":
        a.claim2(args.attack)
    elif args.cmd == "trajectory":
        a.trajectory(args.attack or "adaptive_system")
    else:
        a.all()


if __name__ == "__main__":
    main()
