"""claims_audit.py — machine-check every number the paper quotes.

Recomputes each quoted claim from the exported CSVs (plus the tables written
by the revision scripts) and compares against a hard-coded expected value at
the paper's printed precision.

Revision r1 (20 Oct 2026): expected values are the revised Section VI
numbers, regenerated from the single revision store (revision-r1 branch,
revision_assets_20261020/store; composition in REGEN_LOG.md). Every quoted
value of the revised Section VI is covered:

  Group A  (must MATCH): Table V (all 56 cells, mean and sd), energy per
           delivered update, aggregation time, Mann-Whitney p-values,
           Table IV marks, clean setting (R13), magnitude attack (R13),
           isolated attacks (R13 + stored delay), sign-flip fidelity and jump
           ratios, Table VI secondary metrics, scalability exponents,
           ablation, EuroSAT. A mismatch FAILS the run (exit 1).
  Group B  (informational): the pre-revision (submitted) Table V retention
           values, printed as submitted-vs-regenerated and never failing.

Usage:
    python claims_audit.py --data paper_assets/data --tables paper_assets/tables \
                           --supp paper_assets/supplement
    python claims_audit.py ... --matrix --store <store>   # + Table III/IV facet grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ loading -
def load(data_dir, tables_dir, supp_dir):
    dfs = {}
    for key, fn in [("hard", "hardened_dual.csv"), ("runs", "runs_summary.csv"),
                    ("jump", "fidelity_jump.csv"), ("sysatk", "system_attacks.csv"),
                    ("sweep", "sweep.csv"), ("scal", "scalability.csv")]:
        dfs[key] = pd.read_csv(f"{data_dir}/{fn}")
    for key, fn in [("t4", "table4_marks.csv")]:
        p = f"{tables_dir}/{fn}"
        dfs[key] = pd.read_csv(p) if os.path.exists(p) else None
    for key, fn in [("sec", "secondary_summary.csv")]:
        p = f"{supp_dir}/{fn}"
        dfs[key] = pd.read_csv(p) if os.path.exists(p) else None
    return dfs


# --------------------------------------------------------------- accessors -
def hd(dfs, fam, method):
    h = dfs["hard"]
    return h[(h["family"] == fam) & (h["method"] == method)]


def hd_mean(dfs, fam, method, col):
    return float(hd(dfs, fam, method)[col].mean())


def hd_std(dfs, fam, method, col):
    return float(hd(dfs, fam, method)[col].std(ddof=1))


def runs_named(dfs, name):
    r = dfs["runs"]
    return r[r["name"] == name]


def sysatk_mean(dfs, fam, method, col="infl_late"):
    s = dfs["sysatk"]
    return float(s[(s["family"] == fam) & (s["method"] == method)][col].mean())


def sysatk_std(dfs, fam, method, col="infl_late"):
    s = dfs["sysatk"]
    return float(s[(s["family"] == fam) & (s["method"] == method)][col].std(ddof=1))


def jump_of(dfs, method):
    j = dfs["jump"]
    return float(j[j["method"] == method]["jump_factor"].iloc[0])


def signflip_mean(dfs, method):
    r = dfs["runs"]
    sub = r[r["name"].str.startswith("fid_signflip") & (r["method"] == method)]
    return float(sub["infl_late"].mean())


def energy(dfs, fam):
    sub = dfs["hard"][dfs["hard"]["family"] == fam]
    means = sub.groupby("method")["energy_per_update_j"].mean()
    base = means.drop("trustfl")
    return dict(trustfl=float(means["trustfl"]), base_min=float(base.min()),
                base_max=float(base.max()))


def scal_slope(dfs, method, m_min=None):
    s = dfs["scal"]
    sub = s[s["method"] == method]
    if m_min is not None:
        sub = sub[sub["M"] >= m_min]
    sub = sub.sort_values("M")
    return float(np.polyfit(np.log(sub["M"]), np.log(sub["ms_mean"]), 1)[0])


def _mwu_exact_greater(x, y):
    """Exact one-sided Mann-Whitney (x > y) by enumeration; equals scipy's
    method='exact' for samples without ties."""
    x, y = [float(v) for v in x], [float(v) for v in y]
    pooled, n, m = x + y, len(x), len(y)

    def ustat(idx):
        xs = [pooled[i] for i in idx]
        ys = [pooled[i] for i in range(n + m) if i not in idx]
        return sum(1.0 if a > b else (0.5 if a == b else 0.0) for a in xs for b in ys)

    u_obs = ustat(frozenset(range(n)))
    ge = sum(1 for idx in combinations(range(n + m), n)
             if ustat(frozenset(idx)) >= u_obs - 1e-12)
    return ge / comb(n + m, n)


def mannwhitney(dfs, fam, baseline, col="ret_highmob", alternative="greater"):
    tf = hd(dfs, fam, "trustfl")[col].dropna().values
    bl = hd(dfs, fam, baseline)[col].dropna().values
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(tf, bl, alternative=alternative, method="exact")
        return float(p), tf, bl
    except ImportError:
        p = _mwu_exact_greater(tf, bl) if alternative == "greater" else _mwu_exact_greater(bl, tf)
        return float(p), tf, bl


def t4_marks(dfs, method):
    t = dfs["t4"].set_index("method")
    return [int(t.loc[method, c]) for c in ["acc", "ret", "infl", "deliv", "energy", "cost"]]


def sec_mean(dfs, ds, metric, method):
    s = dfs["sec"]
    return float(s[(s.dataset == ds) & (s.metric == metric) & (s.method == method)]["mean"].iloc[0])


def eur(dfs, method):
    return dfs["runs"][dfs["runs"]["name"] == f"eur_adsys_{method}"]


# ------------------------------------------------------------------ claims -
# Each claim: (id, description, compute(dfs) -> value, expected, nd_or_tol, note)
#   expected float + int nd      -> printed-precision match: f"{v:.{nd}f}" equal
#   expected (lo, hi)            -> lo <= computed <= hi
#   expected list                -> exact list match
MNIST, FEMNIST = "exp_h_adsys", "femh_adsys"
METHODS = ["trustfl", "sophon", "fedaa", "fltrust", "fedgreed", "multikrum", "fedavg"]

# Table V, revised: (ret, sd, infl, sd, deliv, sd, acc, sd)
_TV = {
    MNIST: {"trustfl": (.796, .09, .288, .10, .909, .05, .988, .00),
            "sophon": (.354, .17, .357, .08, .648, .11, .981, .00),
            "fedaa": (.322, .17, .365, .17, .651, .14, .986, .00),
            "fltrust": (.434, .19, .351, .07, .645, .08, .980, .00),
            "fedgreed": (.393, .14, .391, .09, .687, .13, .988, .00),
            "multikrum": (.462, .19, .334, .09, .626, .13, .989, .00),
            "fedavg": (.557, .19, .353, .09, .653, .11, .989, .00)},
    FEMNIST: {"trustfl": (.848, .14, .268, .19, .879, .05, .824, .01),
              "sophon": (.392, .20, .321, .07, .650, .09, .820, .02),
              "fedaa": (.318, .21, .174, .08, .593, .09, .814, .01),
              "fltrust": (.529, .17, .403, .08, .677, .11, .843, .01),
              "fedgreed": (.364, .10, .319, .12, .641, .09, .818, .02),
              "multikrum": (.523, .22, .271, .12, .619, .08, .828, .03),
              "fedavg": (.599, .22, .336, .18, .660, .11, .823, .01)},
}
_COLS = [("ret_highmob", "ret"), ("infl_late", "infl"), ("deliv_eff", "deliv"), ("final_acc", "acc")]

CLAIMS_A = []
for fam, ds in [(MNIST, "MNIST"), (FEMNIST, "FEMNIST")]:
    for m in METHODS:
        e = _TV[fam][m]
        for i, (col, short) in enumerate(_COLS):
            CLAIMS_A.append((f"A1-{ds[:3]}-{m}-{short}", f"Table V {ds} {m} {short} mean",
                             (lambda d, fam=fam, m=m, col=col: hd_mean(d, fam, m, col)), e[2 * i], 3, ""))
            CLAIMS_A.append((f"A1-{ds[:3]}-{m}-{short}-sd", f"Table V {ds} {m} {short} sd",
                             (lambda d, fam=fam, m=m, col=col: hd_std(d, fam, m, col)), e[2 * i + 1], 2, ""))

# energy per delivered update (J) and aggregation time (ms), hardened families
CLAIMS_A += [
    ("A2a", "energy/update MNIST trustfl", lambda d: energy(d, MNIST)["trustfl"], 5214.0, 0, ""),
    ("A2b", "energy/update MNIST baselines min", lambda d: energy(d, MNIST)["base_min"], 7045.0, 0, ""),
    ("A2c", "energy/update MNIST baselines max", lambda d: energy(d, MNIST)["base_max"], 7825.0, 0, ""),
    ("A2d", "energy/update FEMNIST trustfl", lambda d: energy(d, FEMNIST)["trustfl"], 5464.0, 0, ""),
    ("A2e", "energy/update FEMNIST baselines min", lambda d: energy(d, FEMNIST)["base_min"], 7226.0, 0, ""),
    ("A2f", "energy/update FEMNIST baselines max", lambda d: energy(d, FEMNIST)["base_max"], 8274.0, 0, ""),
]
_AGG = {"trustfl": (8.5, 7.5), "fedavg": (5.3, 5.5), "multikrum": (30.5, 31.6),
        "sophon": (349, 2216), "fltrust": (277, 1754), "fedgreed": (1253, 14415),
        "fedaa": (8332, 47709)}
for m, (a, b) in _AGG.items():
    for fam, ds, e in [(MNIST, "MNIST", a), (FEMNIST, "FEMNIST", b)]:
        CLAIMS_A.append((f"A3-{ds[:3]}-{m}", f"aggregation ms {ds} {m} (hardened family)",
                         (lambda d, fam=fam, m=m: hd_mean(d, fam, m, "agg_time_ms")),
                         float(e), 1 if isinstance(e, float) else 0, ""))

# Mann-Whitney, exact one-sided, TrustFL better
_P_RET = {MNIST: {m: .004 for m in METHODS[1:]},
          FEMNIST: {"sophon": .004, "fedgreed": .004, "fedaa": .0079, "fltrust": .0079,
                    "multikrum": .0159, "fedavg": .0278}}
for fam, d_ in _P_RET.items():
    ds = "MNIST" if fam == MNIST else "FEMNIST"
    for m, e in d_.items():
        CLAIMS_A.append((f"A4-ret-{ds[:3]}-{m}", f"MWU retention p {ds} trustfl > {m}",
                         (lambda d, fam=fam, m=m: mannwhitney(d, fam, m)[0]), e, 3 if e == .004 else 4, "exact"))
        CLAIMS_A.append((f"A4-del-{ds[:3]}-{m}", f"MWU delivery p {ds} trustfl > {m}",
                         (lambda d, fam=fam, m=m: mannwhitney(d, fam, m, "deliv_eff")[0]), .004, 3, "exact"))
CLAIMS_A.append(("A4-inf-MNI-fedgreed", "MWU influence p MNIST trustfl < fedgreed",
                 lambda d: mannwhitney(d, MNIST, "fedgreed", "infl_late", "less")[0], .0476, 4, "exact"))
for fam in (MNIST, FEMNIST):
    ds = "MNIST" if fam == MNIST else "FEMNIST"
    for m in METHODS[1:]:
        if fam == MNIST and m == "fedgreed":
            continue
        CLAIMS_A.append((f"A4-inf-{ds[:3]}-{m}", f"MWU influence p {ds} trustfl < {m} in 0.11-0.79",
                         (lambda d, fam=fam, m=m: mannwhitney(d, fam, m, "infl_late", "less")[0]),
                         (0.11, 0.79), None, "exact"))

# Table IV marks (acc, ret, infl, deliv, energy, cost; 2 filled / 1 half / 0 empty)
_T4 = {"TrustFL": [2, 2, 2, 2, 2, 2], "Sophon": [0, 0, 1, 1, 1, 1], "FedAA": [0, 0, 2, 0, 0, 0],
       "FLTrust": [2, 1, 0, 2, 2, 1], "FedGreed": [1, 1, 0, 2, 2, 0],
       "Multi-Krum": [2, 2, 2, 0, 0, 2], "FedAvg": [2, 2, 1, 1, 1, 2]}
CLAIMS_A += [(f"A5-{m}", f"Table IV marks {m}", (lambda d, m=m: t4_marks(d, m)), e, None, "list")
             for m, e in _T4.items()]

# clean setting (exp_a, R13)
_CLEAN_RET = {"trustfl": .839, "fedavg": .776, "fltrust": .657, "multikrum": .447, "krum": .043}
CLAIMS_A += [(f"A6-ret-{m}", f"clean ret_highmob exp_a {m}",
              (lambda d, m=m: float(runs_named(d, f"exp_a_{m}")["ret_highmob"].mean())), e, 3, "")
             for m, e in _CLEAN_RET.items()]
CLAIMS_A += [
    ("A6-acc-trustfl", "clean acc exp_a trustfl",
     lambda d: float(runs_named(d, "exp_a_trustfl")["final_acc"].mean()), .989, 3, ""),
    ("A6-acc-fedavg", "clean acc exp_a fedavg",
     lambda d: float(runs_named(d, "exp_a_fedavg")["final_acc"].mean()), .991, 3, ""),
    ("A6-acc-krum-min", "clean acc exp_a krum worst seed",
     lambda d: float(runs_named(d, "exp_a_krum")["final_acc"].min()), .82, 2, ""),
    ("A6-acc-krum-max", "clean acc exp_a krum best seed",
     lambda d: float(runs_named(d, "exp_a_krum")["final_acc"].max()), .96, 2, ""),
    # magnitude attack (mnist_trustfl_debug, R13)
    ("A7", "magnitude attack trustfl final acc per seed 42/43/44",
     lambda d: [round(v, 4) for v in runs_named(d, "mnist_trustfl_debug").sort_values("seed")["final_acc"]],
     [0.1009, 0.9895, 0.9908], None, "list"),
    # isolated attacks
    ("A8a", "selective: trustfl mean", lambda d: sysatk_mean(d, "exp_b_selective_participation", "trustfl"), .000, 3, ""),
    ("A8b", "selective: trustfl std", lambda d: sysatk_std(d, "exp_b_selective_participation", "trustfl"), .000, 3, ""),
    ("A8c", "misreport: trustfl mean", lambda d: sysatk_mean(d, "exp_b_resource_misreport", "trustfl"), .284, 3, ""),
    ("A8d", "misreport: trustfl std", lambda d: sysatk_std(d, "exp_b_resource_misreport", "trustfl"), .103, 3, ""),
    ("A8e", "misreport: fltrust in 0.30-0.34", lambda d: sysatk_mean(d, "exp_b_resource_misreport", "fltrust"), (0.295, 0.345), None, ""),
    ("A8f", "misreport: fedgreed in 0.30-0.34", lambda d: sysatk_mean(d, "exp_b_resource_misreport", "fedgreed"), (0.295, 0.345), None, ""),
    ("A8g", "misreport: fedavg in 0.30-0.34", lambda d: sysatk_mean(d, "exp_b_resource_misreport", "fedavg"), (0.295, 0.345), None, ""),
    ("A8h", "delay: trustfl mean", lambda d: sysatk_mean(d, "exp_b2_delay", "trustfl"), .085, 3, ""),
    ("A8i", "delay: trustfl std", lambda d: sysatk_std(d, "exp_b2_delay", "trustfl"), .075, 3, ""),
    ("A8j", "delay: fltrust in 0.19-0.20", lambda d: sysatk_mean(d, "exp_b2_delay", "fltrust"), (0.185, 0.205), None, ""),
    ("A8k", "delay: fedavg in 0.19-0.20", lambda d: sysatk_mean(d, "exp_b2_delay", "fedavg"), (0.185, 0.205), None, ""),
    # sign-flip fidelity
    ("A9a", "fid_signflip infl_late: sophon", lambda d: signflip_mean(d, "sophon"), .131, 3, ""),
    ("A9b", "fid_signflip infl_late: fedaa", lambda d: signflip_mean(d, "fedaa"), .042, 3, ""),
    ("A9c", "fid_signflip infl_late: fltrust", lambda d: signflip_mean(d, "fltrust"), .050, 3, ""),
    ("A9d", "fid_signflip infl_late: trustfl", lambda d: signflip_mean(d, "trustfl"), .109, 3, ""),
    ("A9e", "jump factor: fltrust", lambda d: jump_of(d, "fltrust"), 7.0, 1, ""),
    ("A9f", "jump factor: fedaa", lambda d: jump_of(d, "fedaa"), 8.6, 1, ""),
    ("A9g", "jump factor: sophon", lambda d: jump_of(d, "sophon"), 2.7, 1, ""),
    ("A9h", "jump factor: trustfl", lambda d: jump_of(d, "trustfl"), 2.6, 1, ""),
    ("A9i", "fid_signflip trustfl final acc per seed",
     lambda d: [round(v, 3) for v in runs_named(d, "fid_signflip_trustfl").sort_values("seed")["final_acc"]],
     [0.834, 0.859, 0.946], None, "list"),
    # scalability (R06), fitted exponents
    ("A10a", "scalability exponent M>=128: trustfl", lambda d: scal_slope(d, "trustfl", 128), 1.08, 2, 'quoted "1.1"'),
    ("A10b", "scalability exponent M>=128: krum", lambda d: scal_slope(d, "krum", 128), 1.51, 2, ""),
    ("A10c", "scalability exponent M>=128: multikrum", lambda d: scal_slope(d, "multikrum", 128), 1.41, 2, ""),
    ("A10d", "scalability exponent all M: trustfl", lambda d: scal_slope(d, "trustfl"), 1.12, 2, ""),
    ("A10e", "scalability exponent all M: krum", lambda d: scal_slope(d, "krum"), 1.18, 2, ""),
    ("A10f", "scalability exponent all M: multikrum", lambda d: scal_slope(d, "multikrum"), 1.19, 2, ""),
    # ablation (stored)
    ("A11a", "ablation nogate: final accs (sorted)",
     lambda d: sorted(runs_named(d, "exp_c_ablate_nogate")["final_acc"].round(3)), [0.098, 0.103, 0.930], None, "list"),
    ("A11b", "ablation symewma: mean final acc",
     lambda d: float(runs_named(d, "exp_c_ablate_symewma")["final_acc"].mean()), .989, 3, ""),
]

# Table VI (Addendum A secondary metrics): S3, S4, S5a, D1
_S3 = {"trustfl": (0, 0), "sophon": (.366, .377), "fedaa": (.398, .344), "fltrust": (.238, .131),
       "fedgreed": (.385, .323), "multikrum": (.062, .035), "fedavg": (0, 0)}
for m, (a, b) in _S3.items():
    for ds, e in [("MNIST", a), ("FEMNIST", b)]:
        CLAIMS_A.append((f"A12-S3-{ds[:3]}-{m}", f"Table VI S3 honest discarded {ds} {m}",
                         (lambda d, ds=ds, m=m: sec_mean(d, ds, "S3_honest_false_excl", m)), float(e), 3, ""))
for m, (a, b) in {"trustfl": (.263, .289), "fedavg": (.230, .248)}.items():
    for ds, e in [("MNIST", a), ("FEMNIST", b)]:
        CLAIMS_A.append((f"A12-S4-{ds[:3]}-{m}", f"Table VI S4 swarm-wide retention {ds} {m}",
                         (lambda d, ds=ds, m=m: sec_mean(d, ds, "S4_inv_indep_hm_ret", m)), e, 3, ""))
CLAIMS_A += [
    ("A12-S5a-MNI", "Table VI S5a J/honest delivered MNIST trustfl",
     lambda d: sec_mean(d, "MNIST", "S5a_J_per_honest_delivered", "trustfl"), 8202.0, 0, ""),
    ("A12-S5a-FEM", "Table VI S5a J/honest delivered FEMNIST trustfl",
     lambda d: sec_mean(d, "FEMNIST", "S5a_J_per_honest_delivered", "trustfl"), 8874.0, 0, ""),
    ("A12-D1-MNI", "Table VI D1 Jain MNIST trustfl", lambda d: sec_mean(d, "MNIST", "D1_jain_honest", "trustfl"), .462, 3, ""),
    ("A12-D1-FEM", "Table VI D1 Jain FEMNIST trustfl", lambda d: sec_mean(d, "FEMNIST", "D1_jain_honest", "trustfl"), .490, 3, ""),
    ("A12-D1-base-min", "Table VI D1 Jain baselines min in .530-.744",
     lambda d: min(sec_mean(d, ds, "D1_jain_honest", m) for ds in ("MNIST", "FEMNIST") for m in METHODS[1:]),
     (0.5295, 0.7445), None, ""),
    ("A12-D1-base-max", "Table VI D1 Jain baselines max in .530-.744",
     lambda d: max(sec_mean(d, ds, "D1_jain_honest", m) for ds in ("MNIST", "FEMNIST") for m in METHODS[1:]),
     (0.5295, 0.7445), None, ""),
]

# EuroSAT (stored, unchanged): (ret_highmob, infl_late, deliv_eff, final_acc)
_EUR = {"trustfl": (0.920, 0.291, 0.913, 0.829), "fedavg": (0.634, 0.429, 0.719, 0.774),
        "multikrum": (0.563, 0.385, 0.728, 0.735), "sophon": (0.443, 0.382, 0.738, 0.643),
        "fedaa": (0.382, 0.276, 0.708, 0.626)}
for m, vals in _EUR.items():
    for c, e in zip(["ret_highmob", "infl_late", "deliv_eff", "final_acc"], vals):
        CLAIMS_A.append((f"A13-{m}-{c[:4]}", f"EuroSAT {c}: {m}",
                         (lambda d, m=m, c=c: float(eur(d, m)[c].mean())), e, 3, ""))

# Group B: the submitted (pre-revision) Table V retention, for the record
_RET_SUBMITTED = {(MNIST, "trustfl"): .796, (FEMNIST, "trustfl"): .848,
                  (MNIST, "sophon"): .459, (FEMNIST, "sophon"): .539,
                  (MNIST, "fedaa"): .288, (FEMNIST, "fedaa"): .361,
                  (MNIST, "fltrust"): .434, (FEMNIST, "fltrust"): .529,
                  (MNIST, "fedgreed"): .393, (FEMNIST, "fedgreed"): .364,
                  (MNIST, "multikrum"): .462, (FEMNIST, "multikrum"): .524,
                  (MNIST, "fedavg"): .557, (FEMNIST, "fedavg"): .599}
CLAIMS_B = [(f"B1-{f[:4]}-{m}", f"submitted Table V ret_highmob: {m} {'MNIST' if f == MNIST else 'FEMNIST'}",
             (lambda d, f=f, m=m: hd_mean(d, f, m, "ret_highmob")), e, 3, "submitted value")
            for (f, m), e in _RET_SUBMITTED.items()]


# ------------------------------------------------------------------ runner -
def check(computed, expected, nd):
    if isinstance(expected, tuple):
        return expected[0] <= computed <= expected[1]
    if isinstance(expected, list):
        return list(computed) == expected
    return f"{float(computed):.{nd}f}" == f"{float(expected):.{nd}f}"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v) + "]"
    return str(v)


def run_claims(dfs, claims, group):
    fails = 0
    print(f"\n{'='*74}\nGROUP {group}\n{'='*74}")
    for cid, desc, fn, expected, nd, note in claims:
        try:
            v = fn(dfs)
            ok = check(v, expected, nd)
        except Exception as e:
            v, ok = f"ERROR {type(e).__name__}: {e}", False
        if group == "A":
            status = "PASS" if ok else "FAIL"
            if not ok:
                fails += 1
        else:
            status = "unchanged" if ok else "CHANGED in revision"
        extra = f"  ({note})" if note else ""
        print(f"  [{status:>20s}] {cid:22s} {desc:52s} expected {fmt(expected)}  computed {fmt(v)}{extra}")
    return fails


def print_retention_detail(dfs):
    print(f"\n{'-'*74}\nRetention detail (per family x method: mean +/- std, per-seed)")
    for fam, ds in [(MNIST, "MNIST"), (FEMNIST, "FEMNIST")]:
        print(f"  [{ds}]")
        for m in METHODS:
            sub = hd(dfs, fam, m)["ret_highmob"].dropna()
            ov = hd(dfs, fam, m)["ret_overall"].dropna()
            print(f"    {m:10s} hm {sub.mean():.3f}+/-{sub.std():.3f} "
                  f"[{' '.join(f'{v:.3f}' for v in sub)}]   overall {ov.mean():.3f}+/-{ov.std():.3f}")
    print("\n  clean setting (exp_a) ret_highmob / ret_overall:")
    for m in ["trustfl", "fedavg", "fltrust", "multikrum", "krum"]:
        sub = runs_named(dfs, f"exp_a_{m}")
        print(f"    {m:10s} hm {sub['ret_highmob'].mean():.3f}   overall {sub['ret_overall'].mean():.3f}")


def print_matrix(store_root):
    from trustfl.store import Store
    from trustfl.metrics_matrix import matrix
    print(f"\n{'='*74}\nMetrics-matrix facet grid per dataset (Table IV pools both; see "
          f"paper_assets/make_table4_marks.py)\n{'='*74}")
    s = Store(store_root)
    for prefix in [MNIST, FEMNIST]:
        df = matrix(s, prefix)
        print(f"\n  [{prefix}]")
        print("  " + df.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./paper_assets/data")
    ap.add_argument("--tables", default="./paper_assets/tables")
    ap.add_argument("--supp", default="./paper_assets/supplement")
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("--matrix", action="store_true")
    args = ap.parse_args()
    dfs = load(args.data, args.tables, args.supp)
    print(f"claims audit against: {args.data} | {args.tables} | {args.supp}")
    try:
        import scipy  # noqa: F401
        print("Mann-Whitney: scipy exact")
    except ImportError:
        print("Mann-Whitney: built-in exact enumeration (scipy not installed)")
    fails = run_claims(dfs, CLAIMS_A, "A")
    run_claims(dfs, CLAIMS_B, "B")
    print_retention_detail(dfs)
    if args.matrix:
        print_matrix(args.store)
    print(f"\n{'='*74}")
    print(f"GROUP A: {len(CLAIMS_A)} claims, {fails} failure(s)")
    if fails:
        print("STOP and investigate: a quoted value does not match the regenerated store.")
        sys.exit(1)
    print("GROUP A: all claims PASS.")


if __name__ == "__main__":
    main()
