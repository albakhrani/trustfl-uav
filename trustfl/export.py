"""export — process the ENTIRE results store into inspectable CSV artifacts.

One command walks every run, computes every paper metric with a single
audited code path, and writes tidy files to paper_assets/data/:

  runs_summary.csv       one row per run (all per-run metrics)
  experiment_groups.csv  one row per experiment x method (mean/std/min/max, n)
  hardened_dual.csv      Table-1 view: exp_h_adsys + femh_adsys, per seed
  fidelity_jump.csv      sign-flip vs system influence per method + jump
  system_attacks.csv     selective / delay (recalibrated) / misreport
  sweep.csv              rs_f* threat spectrum, per fraction x method x seed
  timeseries_key.csv     long-format per-round data for the dynamics figures
  table1.tex             the paper's Table 1, generated (never hand-edited)
  claims_report.txt      committed claim sentences recomputed fresh + p-values

Open the CSVs, verify the numbers against reality FIRST; figures and prose
then quote only these files. Data bugs and plot bugs become distinguishable.
"""
import json
import os

import numpy as np
import pandas as pd

from .audit import _groups, _parse_dictcol, _parse_listcol, retention_for_run
from .store import Store

OUT = "./paper_assets/data"

METHODS_ORDER = ["trustfl", "sophon", "fedaa", "fltrust", "fedgreed",
                 "multikrum", "fedavg", "krum", "median", "trimmed_mean"]
LABELS = {"trustfl": "TrustFL (ours)", "sophon": "Sophon (TDSC'25)",
          "fedaa": "FedAA (AAAI'25)", "fltrust": "FLTrust (NDSS'21)",
          "fedgreed": "FedGreed (preprint)", "multikrum": "Multi-Krum",
          "fedavg": "FedAvg", "krum": "Krum"}


# ----------------------------------------------------------- per-run core -
def _per_round(store, run_id):
    """Everything the paper needs, per round, with the int-cast fix."""
    mal, hm = _groups(store, run_id)
    rd = store.rounds(run_id)
    rows = []
    for _, r in rd.iterrows():
        invited = {int(j) for j in _parse_listcol(r.get("invited"))}
        delivered = {int(j) for j in _parse_listcol(r.get("delivered"))}
        infl = {int(k): float(v)
                for k, v in _parse_dictcol(r.get("influence")).items()}
        survived = {j for j, w in infl.items() if w > 0}
        mal_infl = sum(w for j, w in infl.items() if j in mal)
        tot_infl = sum(infl.values())
        honest_inv = {j for j in invited if j not in mal}
        hm_inv = {j for j in honest_inv if j in hm}
        row = dict(
            round=int(r["round"]),
            test_acc=r.get("test_acc", np.nan),
            test_loss=r.get("test_loss", np.nan),
            n_invited=len(invited),
            n_delivered=len(delivered),
            deliv_eff=(len(delivered) / len(invited)) if invited else np.nan,
            mal_influence=(mal_infl / tot_infl) if tot_infl > 0 else np.nan,
            ret_overall=(len(honest_inv & survived) / len(honest_inv))
                        if honest_inv else np.nan,
            ret_highmob=(len(hm_inv & survived) / len(hm_inv))
                        if hm_inv else np.nan,
            mean_energy_j=r.get("mean_energy_j", np.nan),
            trust_honest=r.get("mean_trust_honest", np.nan),
            trust_malicious=r.get("mean_trust_malicious", np.nan),
        )
        tw = r.get("trust_weights")
        if isinstance(tw, str):
            try:
                tw = json.loads(tw)
            except Exception:
                tw = None
        if isinstance(tw, (list, tuple)) and len(tw) >= 3:
            row.update(w_beh=tw[0], w_comp=tw[1], w_comm=tw[2])
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_run(store, meta, pr):
    late = pr[pr["round"] >= pr["round"].max() / 2]
    e = pr.dropna(subset=["mean_energy_j"])["mean_energy_j"]
    nuav = float(meta.get("num_uavs", np.nan))
    total_deliv = int(pr["n_delivered"].sum())
    consumed = ((float(e.iloc[0]) - float(e.iloc[-1])) * nuav
                if len(e) >= 2 and nuav == nuav else np.nan)
    acc = pr.dropna(subset=["test_acc"])["test_acc"]
    ov, hmob = retention_for_run(store, meta["run_id"])
    return dict(
        run_id=meta["run_id"], name=meta["name"], method=meta["method"],
        dataset=meta["dataset"], attack=meta["attack_type"],
        mal_frac=meta["malicious_frac"], seed=meta["seed"],
        rounds=int(pr["round"].max()) + 1,
        final_acc=float(acc.iloc[-1]) if len(acc) else np.nan,
        best_acc=float(acc.max()) if len(acc) else np.nan,
        ret_overall=ov, ret_highmob=hmob,
        infl_mean=float(pr["mal_influence"].mean()),
        infl_late=float(late["mal_influence"].mean()),
        deliv_eff=float(pr["deliv_eff"].mean()),
        total_delivered=total_deliv,
        energy_consumed_j=consumed,
        energy_per_update_j=(consumed / total_deliv)
                            if consumed == consumed and total_deliv else np.nan,
        agg_time_ms=meta.get("agg_time_ms", np.nan),
    )


# ---------------------------------------------------------------- exports -
def export_all(store_root="./results_store", out=OUT):
    os.makedirs(out, exist_ok=True)
    s = Store(store_root)
    df = s.query()
    summaries, ts_rows = [], []
    ts_families = ("exp_h_adsys", "femh_adsys", "fid_signflip", "exp_a_",
                   "fem_a_", "exp_b_selective", "exp_b2_delay",
                   "exp_b_resource", "mnist_attack_fedavg",
                   "mnist_trustfl_debug", "exp_c_", "rs_f")
    for _, meta in df.iterrows():
        try:
            pr = _per_round(s, meta["run_id"])
            if pr.empty:
                continue
            summaries.append(_summarize_run(s, meta, pr))
            if any(meta["name"].startswith(f) for f in ts_families):
                pr2 = pr.copy()
                pr2.insert(0, "seed", meta["seed"])
                pr2.insert(0, "method", meta["method"])
                pr2.insert(0, "name", meta["name"])
                pr2.insert(0, "run_id", meta["run_id"])
                ts_rows.append(pr2)
        except Exception as e:
            print(f"  ! {meta['run_id']}: {type(e).__name__}: {e}")
    runs = pd.DataFrame(summaries)
    runs.to_csv(f"{out}/runs_summary.csv", index=False)
    print(f"  runs_summary.csv           {len(runs)} rows")

    # family = experiment name minus trailing _{method}
    def family(row):
        suf = "_" + row["method"]
        return row["name"][: -len(suf)] if row["name"].endswith(suf) else row["name"]
    runs["family"] = runs.apply(family, axis=1)
    agg_cols = ["final_acc", "ret_highmob", "ret_overall", "infl_late",
                "deliv_eff", "energy_per_update_j", "agg_time_ms"]
    grp = runs.groupby(["family", "method", "dataset", "attack", "mal_frac"])
    g = grp[agg_cols].agg(["mean", "std", "min", "max"])
    g.columns = ["_".join(c) for c in g.columns]
    g["n_seeds"] = grp.size()
    g.reset_index().to_csv(f"{out}/experiment_groups.csv", index=False)
    print(f"  experiment_groups.csv      {len(g)} experiment x method rows")

    # Table-1 view
    hard = runs[runs["family"].isin(["exp_h_adsys", "femh_adsys"])].copy()
    hard.to_csv(f"{out}/hardened_dual.csv", index=False)
    print(f"  hardened_dual.csv          {len(hard)} rows")

    # fidelity + jump
    fid = runs[runs["family"] == "fid_signflip"][
        ["method", "seed", "infl_late", "final_acc"]].rename(
        columns={"infl_late": "grad_infl", "final_acc": "grad_acc"})
    sysm = hard[hard["family"] == "exp_h_adsys"][
        ["method", "seed", "infl_late"]].rename(columns={"infl_late": "sys_infl"})
    if len(fid) and len(sysm):
        gm = fid.groupby("method")["grad_infl"].mean()
        sm = sysm.groupby("method")["sys_infl"].mean()
        jump = pd.DataFrame({"grad_infl_mean": gm, "sys_infl_mean": sm})
        jump["jump_factor"] = jump["sys_infl_mean"] / jump["grad_infl_mean"].clip(lower=1e-6)
        jump.reset_index().to_csv(f"{out}/fidelity_jump.csv", index=False)
        print(f"  fidelity_jump.csv          {len(jump)} methods")

    # system attacks (recalibrated delay only)
    sysatk = runs[runs["family"].isin(
        ["exp_b_selective_participation", "exp_b2_delay",
         "exp_b_resource_misreport"])]
    sysatk.to_csv(f"{out}/system_attacks.csv", index=False)
    print(f"  system_attacks.csv         {len(sysatk)} rows")

    # sweep
    sweep = runs[runs["family"].str.startswith("rs_f")].copy()
    if len(sweep):
        sweep["fraction"] = sweep["family"].str.replace("rs_f", "").astype(float)
        sweep.to_csv(f"{out}/sweep.csv", index=False)
        print(f"  sweep.csv                  {len(sweep)} rows")

    # timeseries
    if ts_rows:
        ts = pd.concat(ts_rows, ignore_index=True)
        ts.to_csv(f"{out}/timeseries_key.csv", index=False)
        print(f"  timeseries_key.csv         {len(ts)} rows "
              f"({ts['run_id'].nunique()} runs)")

    _emit_table1(hard, out)
    _emit_claims(runs, out)
    print(f"\nAll artifacts in {out}/ — open the CSVs and verify BEFORE plotting.")


def _emit_table1(hard, out):
    lines = []
    for m in METHODS_ORDER:
        row = {"MNIST": None, "FEMNIST": None}
        for fam, ds in [("exp_h_adsys", "MNIST"), ("femh_adsys", "FEMNIST")]:
            sub = hard[(hard["family"] == fam) & (hard["method"] == m)]
            if len(sub):
                row[ds] = (sub["ret_highmob"].mean(), sub["infl_late"].mean(),
                           sub["deliv_eff"].mean(), sub["final_acc"].mean())
        if row["MNIST"] is None and row["FEMNIST"] is None:
            continue
        def cells(t):
            return " & ".join(f"{v:.3f}" for v in t) if t else "-- & -- & -- & --"
        bold = r"\textbf" if m == "trustfl" else ""
        name = LABELS.get(m, m)
        lines.append(f"{name} & {cells(row['MNIST'])} & & {cells(row['FEMNIST'])} \\\\")
    with open(f"{out}/table1.tex", "w") as f:
        f.write("% AUTO-GENERATED by trustfl.export -- never hand-edit.\n")
        f.write("% Columns per dataset: Retention / Influence / Delivery / Accuracy\n")
        f.write("\n".join(lines) + "\n")
    print("  table1.tex                 generated from hardened_dual.csv")


def _emit_claims(runs, out):
    rep = ["CLAIMS REPORT — recomputed fresh from the store", "=" * 60]
    def fam(f, m):
        return runs[(runs["family"] == f) & (runs["method"] == m)]
    for f, ds in [("exp_h_adsys", "MNIST"), ("femh_adsys", "FEMNIST")]:
        tf = fam(f, "trustfl")["ret_highmob"].dropna()
        rep.append(f"\n[{ds} hardened, retention]")
        if len(tf):
            rep.append(f"  TrustFL: mean {tf.mean():.3f}  per-seed "
                       f"{' '.join(f'{v:.3f}' for v in sorted(tf))}")
        for comp in ["sophon", "fedaa"]:
            cv = fam(f, comp)["ret_highmob"].dropna()
            if len(tf) and len(cv):
                sep = "CLEAN SEPARATION" if tf.min() > cv.max() else "overlap"
                line = (f"  vs {comp}: gap +{tf.mean()-cv.mean():.3f}  "
                        f"[TrustFL worst {tf.min():.3f} vs best {cv.max():.3f}] {sep}")
                try:
                    from scipy.stats import mannwhitneyu
                    _, p = mannwhitneyu(tf, cv, alternative="greater")
                    line += f"  Mann-Whitney p={p:.4f}"
                except ImportError:
                    pass
                rep.append(line)
    rep.append("\n[Committed sentence check]")
    rep.append("  'p<=0.036 across all four dataset x competitor comparisons,")
    rep.append("   clean worst-vs-best separation in three of the four'")
    rep.append("  -> verify the four lines above match before quoting.")
    with open(f"{out}/claims_report.txt", "w") as fh:
        fh.write("\n".join(rep) + "\n")
    print("  claims_report.txt          committed claims recomputed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    export_all(a.store, a.out)
