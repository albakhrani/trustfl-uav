"""claims_audit.py — machine-check every number the paper quotes.

Recomputes each quoted claim from the exported CSVs and compares against a
hard-coded expected value with a tolerance.

  Group A  (must MATCH, not retention-derived): accuracy, influence, delivery,
           energy, sign-flip, system attacks, fidelity jump, ablation,
           scalability, clean accuracy, EuroSAT aggregates. A mismatch FAILS
           the run (exit 1).
  Group B  (retention-derived): Table I retention, Mann-Whitney p-values,
           clean-setting retention. Expecteds hold the post-patch numbers
           quoted in the paper (unified retention metric); printed as
           expected-vs-computed and never fails the run.

Usage:
    python claims_audit.py --data paper_assets/data              # post-patch
    python claims_audit.py --data <pre-patch csv dir>            # pre-patch
    python claims_audit.py --data paper_assets/data --matrix     # + Table III grid
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ loading -
def load(data_dir):
    dfs = {}
    for key, fn in [("hard", "hardened_dual.csv"), ("runs", "runs_summary.csv"),
                    ("jump", "fidelity_jump.csv"), ("sysatk", "system_attacks.csv"),
                    ("sweep", "sweep.csv"), ("scal", "scalability.csv")]:
        dfs[key] = pd.read_csv(f"{data_dir}/{fn}")
    return dfs


# --------------------------------------------------------------- accessors -
def hd(dfs, fam, method):
    h = dfs["hard"]
    return h[(h["family"] == fam) & (h["method"] == method)]


def hd_mean(dfs, fam, method, col):
    return float(hd(dfs, fam, method)[col].mean())


def hd_std(dfs, fam, method, col):
    return float(hd(dfs, fam, method)[col].std())


def runs_named(dfs, name):
    r = dfs["runs"]
    return r[r["name"] == name]


def sysatk_mean(dfs, fam, method, col="infl_late"):
    s = dfs["sysatk"]
    sub = s[(s["family"] == fam) & (s["method"] == method)]
    return float(sub[col].mean())


def sysatk_std(dfs, fam, method, col="infl_late"):
    s = dfs["sysatk"]
    sub = s[(s["family"] == fam) & (s["method"] == method)]
    return float(sub[col].std())


def jump_of(dfs, method):
    j = dfs["jump"]
    return float(j[j["method"] == method]["jump_factor"].iloc[0])


def signflip_mean(dfs, method):
    r = dfs["runs"]
    sub = r[r["name"].str.startswith("fid_signflip") & (r["method"] == method)]
    return float(sub["infl_late"].mean())


def energy_hardened(dfs):
    """trustfl energy/update on exp_h_adsys + baseline min/max + reduction range."""
    h = dfs["hard"]
    sub = h[h["family"] == "exp_h_adsys"]
    means = sub.groupby("method")["energy_per_update_j"].mean()
    tf = means["trustfl"]
    base = means.drop("trustfl")
    red = 1.0 - tf / base
    return dict(trustfl=float(tf), base_min=float(base.min()),
                base_max=float(base.max()),
                red_min=float(red.min()), red_max=float(red.max()))


def scal_slope(dfs, method, m_min=128):
    s = dfs["scal"]
    sub = s[(s["method"] == method) & (s["M"] >= m_min)].sort_values("M")
    return float(np.polyfit(np.log(sub["M"]), np.log(sub["ms_mean"]), 1)[0])


def scal_gap512(dfs):
    s = dfs["scal"]
    at = s[s["M"] == 512].set_index("method")["ms_mean"]
    return float(at["krum"] / at["trustfl"])


def mannwhitney(dfs, fam, baseline):
    from scipy.stats import mannwhitneyu
    tf = hd(dfs, fam, "trustfl")["ret_highmob"].dropna().values
    bl = hd(dfs, fam, baseline)["ret_highmob"].dropna().values
    _, p = mannwhitneyu(tf, bl, alternative="greater", method="exact")
    return float(p), tf, bl


def eur(dfs, method):
    """EuroSAT adaptive-system rows (family eur_adsys) for one method."""
    return dfs["runs"][dfs["runs"]["name"] == f"eur_adsys_{method}"]


def eur_mean(dfs, method, col):
    return float(eur(dfs, method)[col].mean())


def eur_extreme(dfs, method, col, which):
    s = eur(dfs, method)[col]
    return float(s.min() if which == "min" else s.max())


# ------------------------------------------------------------------ claims -
# Each claim: (id, description, compute(dfs) -> value, expected, tol, note)
# expected as float -> |computed - expected| <= tol
# expected as (lo, hi) -> lo <= computed <= hi (tol ignored)

CLAIMS_A = [
    # -- hardened accuracy (must be untouched by a retention fix)
    ("A1a", "hardened acc: trustfl MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "trustfl", "final_acc"), 0.988, 0.001, ""),
    ("A1b", "hardened acc: trustfl FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "trustfl", "final_acc"), 0.824, 0.001, ""),
    ("A1c", "hardened acc: fltrust FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "fltrust", "final_acc"), 0.843, 0.001, ""),
    # -- hardened influence, Table I, all seven methods x both datasets
    #    (frozen from the PRE-patch hardened_dual.csv; influence must not move)
    ("A2a", "Table I infl_late: trustfl MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "trustfl", "infl_late"), 0.2880, 0.001, ""),
    ("A2b", "Table I infl_late: sophon MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "sophon", "infl_late"), 0.3755, 0.001, ""),
    ("A2c", "Table I infl_late: fedaa MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "fedaa", "infl_late"), 0.2810, 0.001, ""),
    ("A2d", "Table I infl_late: fltrust MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "fltrust", "infl_late"), 0.3508, 0.001, ""),
    ("A2e", "Table I infl_late: fedgreed MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "fedgreed", "infl_late"), 0.3910, 0.001, ""),
    ("A2f", "Table I infl_late: multikrum MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "multikrum", "infl_late"), 0.3342, 0.001, ""),
    ("A2g", "Table I infl_late: fedavg MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "fedavg", "infl_late"), 0.3533, 0.001, ""),
    ("A2h", "Table I infl_late: trustfl FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "trustfl", "infl_late"), 0.2678, 0.001, ""),
    ("A2i", "Table I infl_late: sophon FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "sophon", "infl_late"), 0.3452, 0.001, ""),
    ("A2j", "Table I infl_late: fedaa FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "fedaa", "infl_late"), 0.1427, 0.001, ""),
    ("A2k", "Table I infl_late: fltrust FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "fltrust", "infl_late"), 0.4032, 0.001, ""),
    ("A2l", "Table I infl_late: fedgreed FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "fedgreed", "infl_late"), 0.3186, 0.001, ""),
    ("A2m", "Table I infl_late: multikrum FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "multikrum", "infl_late"), 0.2907, 0.001, ""),
    ("A2n", "Table I infl_late: fedavg FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "fedavg", "infl_late"), 0.3360, 0.001, ""),
    # -- delivery (delivered/invited, not weight-based -> unchanged)
    ("A3a", "delivery: trustfl MNIST",
     lambda d: hd_mean(d, "exp_h_adsys", "trustfl", "deliv_eff"), 0.909, 0.001, ""),
    ("A3b", "delivery: trustfl FEMNIST",
     lambda d: hd_mean(d, "femh_adsys", "trustfl", "deliv_eff"), 0.879, 0.001, ""),
    # -- energy
    ("A4a", "energy/update: trustfl exp_h",
     lambda d: energy_hardened(d)["trustfl"], 5214.0, 5.0, ""),
    ("A4b", "energy/update: baseline min (exp_h)",
     lambda d: energy_hardened(d)["base_min"], 7045.0, 5.0, ""),
    ("A4c", "energy/update: baseline max (exp_h)",
     lambda d: energy_hardened(d)["base_max"], 7959.0, 5.0, ""),
    ("A4d", "energy reduction min (vs baselines)",
     lambda d: energy_hardened(d)["red_min"], (0.25, 0.27), None,
     "paper quotes 26%"),
    ("A4e", "energy reduction max (vs baselines)",
     lambda d: energy_hardened(d)["red_max"], (0.33, 0.35), None,
     "paper quotes 34% (computed 34.5%, truncated in prose)"),
    # -- sign-flip fidelity
    ("A5a", "fid_signflip infl_late: trustfl",
     lambda d: signflip_mean(d, "trustfl"), 0.109, 0.001, ""),
    ("A5b", "fid_signflip infl_late: fltrust",
     lambda d: signflip_mean(d, "fltrust"), (0.037, 0.0505), None, ""),
    ("A5c", "fid_signflip infl_late: sophon",
     lambda d: signflip_mean(d, "sophon"), (0.037, 0.0505), None, ""),
    ("A5d", "fid_signflip infl_late: fedaa",
     lambda d: signflip_mean(d, "fedaa"), (0.037, 0.0505), None, ""),
    # -- system attacks (exp_b families, infl_late as quoted)
    ("A6a", "selective: trustfl mean",
     lambda d: sysatk_mean(d, "exp_b_selective_participation", "trustfl"),
     0.000, 0.001, ""),
    ("A6b", "selective: trustfl std",
     lambda d: sysatk_std(d, "exp_b_selective_participation", "trustfl"),
     0.000, 0.001, ""),
    ("A6c", "selective: fltrust",
     lambda d: sysatk_mean(d, "exp_b_selective_participation", "fltrust"),
     0.179, 0.001, ""),
    ("A6d", "selective: fedgreed",
     lambda d: sysatk_mean(d, "exp_b_selective_participation", "fedgreed"),
     0.136, 0.001, ""),
    ("A6e", "selective: fedavg",
     lambda d: sysatk_mean(d, "exp_b_selective_participation", "fedavg"),
     0.154, 0.001, ""),
    ("A6f", "delay: trustfl mean",
     lambda d: sysatk_mean(d, "exp_b2_delay", "trustfl"), 0.085, 0.001, ""),
    ("A6g", "delay: trustfl std",
     lambda d: sysatk_std(d, "exp_b2_delay", "trustfl"), 0.075, 0.001, ""),
    ("A6h", "delay: fltrust in 0.19-0.20",
     lambda d: sysatk_mean(d, "exp_b2_delay", "fltrust"), (0.185, 0.205), None, ""),
    ("A6i", "delay: fedavg in 0.19-0.20",
     lambda d: sysatk_mean(d, "exp_b2_delay", "fedavg"), (0.185, 0.205), None, ""),
    ("A6j", "misreport: trustfl mean",
     lambda d: sysatk_mean(d, "exp_b_resource_misreport", "trustfl"),
     0.261, 0.001, ""),
    ("A6k", "misreport: trustfl std",
     lambda d: sysatk_std(d, "exp_b_resource_misreport", "trustfl"),
     0.090, 0.001, ""),
    ("A6l", "misreport: fltrust in 0.30-0.34",
     lambda d: sysatk_mean(d, "exp_b_resource_misreport", "fltrust"),
     (0.295, 0.345), None, ""),
    ("A6m", "misreport: fedgreed in 0.30-0.34",
     lambda d: sysatk_mean(d, "exp_b_resource_misreport", "fedgreed"),
     (0.295, 0.345), None, ""),
    ("A6n", "misreport: fedavg in 0.30-0.34",
     lambda d: sysatk_mean(d, "exp_b_resource_misreport", "fedavg"),
     (0.295, 0.345), None, ""),
    # -- fidelity jump factors (paper: "7-10x"; computed 6.97/6.97/10.09,
    #    rounded to one decimal in prose)
    ("A7a", "jump factor: fltrust",
     lambda d: jump_of(d, "fltrust"), (6.9, 10.15), None, "quoted as 7x"),
    ("A7b", "jump factor: sophon",
     lambda d: jump_of(d, "sophon"), (6.9, 10.15), None, "quoted as 10x"),
    ("A7c", "jump factor: fedaa",
     lambda d: jump_of(d, "fedaa"), (6.9, 10.15), None, "quoted as 7x"),
    # -- ablation
    ("A8a", "ablation nogate: final accs (sorted)",
     lambda d: sorted(runs_named(d, "exp_c_ablate_nogate")["final_acc"].round(3)),
     [0.098, 0.103, 0.930], None, "set match"),
    ("A8b", "ablation symewma: mean final acc",
     lambda d: float(runs_named(d, "exp_c_ablate_symewma")["final_acc"].mean()),
     0.989, 0.001, ""),
    # -- scalability
    ("A9a", "scalability slope (M>=128): trustfl",
     lambda d: scal_slope(d, "trustfl"), 1.1, 0.05, ""),
    ("A9b", "scalability slope (M>=128): krum",
     lambda d: scal_slope(d, "krum"), (1.4, 1.6), None, ""),
    ("A9c", "scalability slope (M>=128): multikrum",
     lambda d: scal_slope(d, "multikrum"), (1.35, 1.5), None, ""),
    ("A9d", "scalability gap krum/trustfl at M=512",
     lambda d: scal_gap512(d), 4.9, 0.2, ""),
    # -- clean accuracy
    ("A10a", "clean acc: exp_a trustfl",
     lambda d: float(runs_named(d, "exp_a_trustfl")["final_acc"].mean()),
     0.989, 0.001, ""),
    ("A10b", "clean acc: exp_a fedavg",
     lambda d: float(runs_named(d, "exp_a_fedavg")["final_acc"].mean()),
     0.991, 0.001, ""),
]

# -- EuroSAT (family eur_adsys; 40% adaptive system+model attack; 3 seeds).
#    Group-A aggregates over ret_highmob / infl_late / deliv_eff / final_acc,
#    plus the two boundary seeds quoted in the text.
_EUR = {  # method -> (ret_highmob, infl_late, deliv_eff, final_acc)
    "trustfl": (0.920, 0.291, 0.913, 0.829),
    "fedavg": (0.634, 0.429, 0.719, 0.774),
    "multikrum": (0.563, 0.385, 0.728, 0.735),
    "sophon": (0.443, 0.382, 0.738, 0.643),
    "fedaa": (0.382, 0.276, 0.708, 0.626),
}
_EUR_COLS = ["ret_highmob", "infl_late", "deliv_eff", "final_acc"]
CLAIMS_A += [
    (f"A11-{m}-{c[:4]}", f"EuroSAT {c}: {m}",
     (lambda d, m=m, c=c: eur_mean(d, m, c)), e, 0.005, "")
    for m, vals in _EUR.items() for c, e in zip(_EUR_COLS, vals)
] + [
    ("A11-bnd-tf", "EuroSAT worst-seed ret_highmob: trustfl (min)",
     lambda d: eur_extreme(d, "trustfl", "ret_highmob", "min"), 0.8806, 0.001, ""),
    ("A11-bnd-fa", "EuroSAT best-seed ret_highmob: fedavg (max)",
     lambda d: eur_extreme(d, "fedavg", "ret_highmob", "max"), 0.6454, 0.001, ""),
]

# Group B expected values, updated to the post-patch numbers now quoted in the
# paper (unified retention metric, 2026-07). Printed as expected-vs-computed;
# Group B never fails the run.
_RET_EXP = {  # (family, method) -> post-patch ret_highmob mean
    ("exp_h_adsys", "trustfl"): 0.7963, ("femh_adsys", "trustfl"): 0.8476,
    ("exp_h_adsys", "sophon"): 0.4586, ("femh_adsys", "sophon"): 0.5385,
    ("exp_h_adsys", "fedaa"): 0.2876, ("femh_adsys", "fedaa"): 0.3613,
    ("exp_h_adsys", "fltrust"): 0.4340, ("femh_adsys", "fltrust"): 0.5293,
    ("exp_h_adsys", "fedgreed"): 0.3934, ("femh_adsys", "fedgreed"): 0.3640,
    ("exp_h_adsys", "multikrum"): 0.4616, ("femh_adsys", "multikrum"): 0.5239,
    ("exp_h_adsys", "fedavg"): 0.5566, ("femh_adsys", "fedavg"): 0.5988,
}
_P_EXP = {("exp_h_adsys", "sophon"): 0.0040, ("exp_h_adsys", "fedaa"): 0.0040,
          ("femh_adsys", "sophon"): 0.0079, ("femh_adsys", "fedaa"): 0.0079}
_CLEAN_RET_EXP = {"trustfl": 0.911, "fedavg": 0.820, "fltrust": 0.640,
                  "multikrum": 0.637, "krum": 0.173}

CLAIMS_B = (
    [(f"B1-{f[:4]}-{m}", f"Table I ret_highmob: {m} {'MNIST' if f=='exp_h_adsys' else 'FEMNIST'}",
      (lambda d, f=f, m=m: hd_mean(d, f, m, "ret_highmob")), exp, 0.0015, "")
     for (f, m), exp in _RET_EXP.items()]
    + [(f"B2-{f[:4]}-{m}", f"Mann-Whitney p (trustfl > {m}, {'MNIST' if f=='exp_h_adsys' else 'FEMNIST'})",
        (lambda d, f=f, m=m: mannwhitney(d, f, m)[0]), exp, 0.0015, "exact, greater")
       for (f, m), exp in _P_EXP.items()]
    + [(f"B3-{m}", f"clean ret_highmob: exp_a {m}",
        (lambda d, m=m: float(runs_named(d, f"exp_a_{m}")["ret_highmob"].mean())),
        exp, 0.0015, "")
       for m, exp in _CLEAN_RET_EXP.items()]
)


# ------------------------------------------------------------------ runner -
def check(computed, expected, tol):
    if isinstance(expected, tuple):
        return expected[0] <= computed <= expected[1]
    if isinstance(expected, list):
        return list(computed) == expected
    return abs(computed - expected) <= tol


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                               for x in v) + "]"
    return str(v)


def run_claims(dfs, claims, group):
    fails = 0
    print(f"\n{'='*74}\nGROUP {group}\n{'='*74}")
    for cid, desc, fn, expected, tol, note in claims:
        try:
            v = fn(dfs)
            ok = check(v, expected, tol)
        except Exception as e:
            v, ok = f"ERROR {type(e).__name__}: {e}", False
        if group == "A":
            status = "PASS" if ok else "FAIL"
            if not ok:
                fails += 1
        else:
            status = "unchanged" if ok else "CHANGED -> update paper"
        extra = f"  ({note})" if note else ""
        print(f"  [{status:>22s}] {cid:12s} {desc:48s} "
              f"expected {fmt(expected)}  computed {fmt(v)}{extra}")
    return fails


def print_retention_detail(dfs):
    """Per-seed retention + std, and the per-seed arrays behind the p-values."""
    print(f"\n{'-'*74}\nRetention detail (per family x method: mean +/- std, per-seed)")
    for fam, ds in [("exp_h_adsys", "MNIST"), ("femh_adsys", "FEMNIST")]:
        print(f"  [{ds}]")
        for m in ["trustfl", "sophon", "fedaa", "fltrust", "fedgreed",
                  "multikrum", "fedavg"]:
            sub = hd(dfs, fam, m)["ret_highmob"].dropna()
            seeds = " ".join(f"{v:.3f}" for v in sub)
            ov = hd(dfs, fam, m)["ret_overall"].dropna()
            print(f"    {m:10s} hm {sub.mean():.3f}+/-{sub.std():.3f} "
                  f"[{seeds}]   overall {ov.mean():.3f}+/-{ov.std():.3f}")
    print(f"\n  clean setting (exp_a) ret_highmob / ret_overall:")
    for m in ["trustfl", "fedavg", "fltrust", "multikrum", "krum"]:
        sub = runs_named(dfs, f"exp_a_{m}")
        print(f"    {m:10s} hm {sub['ret_highmob'].mean():.3f}   "
              f"overall {sub['ret_overall'].mean():.3f}")
    print(f"\n  Mann-Whitney per-seed arrays (ret_highmob):")
    for (fam, m) in _P_EXP:
        p, tf, bl = mannwhitney(dfs, fam, m)
        ds = "MNIST" if fam == "exp_h_adsys" else "FEMNIST"
        print(f"    {ds:8s} trustfl {np.sort(tf).round(3).tolist()} "
              f"vs {m} {np.sort(bl).round(3).tolist()}  -> p={p:.4f}")


def print_matrix(store_root):
    """Table III facet grid from the (patched) metrics_matrix pipeline."""
    from trustfl.store import Store
    from trustfl.metrics_matrix import matrix
    LOWER_BETTER = {"rounds_to_target", "mal_infl", "energy_per_update", "agg_ms"}
    print(f"\n{'='*74}\nTable III facet grid (patched metrics_matrix; "
          f"ranks: T=top-3 M=middle-2 B=bottom-2)\n{'='*74}")
    s = Store(store_root)
    for prefix in ["exp_h_adsys", "femh_adsys"]:
        df = matrix(s, prefix)
        if df.empty:
            print(f"  {prefix}: no runs"); continue
        print(f"\n  [{prefix}]")
        print("  " + df.round(3).to_string(index=False))
        facets = [c for c in df.columns if c != "method"]
        marks = pd.DataFrame({"method": df["method"]})
        for fc in facets:
            asc = fc in LOWER_BETTER
            order = df[fc].rank(ascending=asc, method="min")
            n = order.notna().sum()
            def mark(r):
                if r != r:
                    return "-"
                if r <= 3:
                    return "T"
                if r <= 5:
                    return "M"
                return "B"
            marks[fc] = order.map(mark)
        print("\n  rank marks (per facet):")
        print("  " + marks.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./paper_assets/data")
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("--matrix", action="store_true",
                    help="also print the Table III facet grid from the store")
    args = ap.parse_args()
    dfs = load(args.data)
    print(f"claims audit against: {args.data}")
    fails = run_claims(dfs, CLAIMS_A, "A")
    run_claims(dfs, CLAIMS_B, "B")
    print_retention_detail(dfs)
    if args.matrix:
        print_matrix(args.store)
    print(f"\n{'='*74}")
    if fails:
        print(f"GROUP A: {fails} FAILURE(S) — influence/delivery/accuracy/energy "
              f"drifted. STOP and investigate.")
        sys.exit(1)
    print("GROUP A: all claims PASS.")


if __name__ == "__main__":
    main()
