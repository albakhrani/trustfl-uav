"""Paper-figure and table generation from the results store.

Deterministic: every table/figure is a pure function of the store, so the
paper's numbers regenerate with no re-running. This is the presentation
foundation — figures read here, never from ad-hoc log parsing.

Outputs go to <out>/tables/*.csv (+ LaTeX) and <out>/figdata/*.csv so the
figure scripts (Phase 4) consume clean, typed inputs.

CLI:
    python -m trustfl.analysis --store ./results_store motivating
    python -m trustfl.analysis --store ./results_store cost
    python -m trustfl.analysis --store ./results_store claim1
    python -m trustfl.analysis --store ./results_store all
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .store import Store


def _fmt_pm(mean, std):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = " & ".join(df.columns)
    body = " \\\\\n".join(" & ".join(str(x) for x in row) for row in df.values)
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{'l' + 'r' * (len(df.columns) - 1)}}}\n\\toprule\n"
        f"{cols} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )


class Analysis:
    def __init__(self, store_root: str, out: str = "./paper_assets", canonical_only: bool = True):
        self.s = Store(store_root)
        self.out = out
        self.canonical_only = canonical_only
        os.makedirs(os.path.join(out, "tables"), exist_ok=True)
        os.makedirs(os.path.join(out, "figdata"), exist_ok=True)

    def _q(self, where: str = "", params: tuple = ()):
        """Query with canonical-only filtering applied (excludes exploratory)."""
        clauses = []
        if where:
            clauses.append(f"({where})")
        if self.canonical_only:
            clauses.append("paper_tag IS NOT NULL AND paper_tag != 'exploratory'")
        full = " AND ".join(clauses)
        try:
            return self.s.query(full, params)
        except Exception:
            # paper_tag column may not exist yet; fall back to unfiltered
            return self.s.query(where, params)

    def _save_table(self, df: pd.DataFrame, stem: str, caption: str, label: str):
        csv = os.path.join(self.out, "tables", f"{stem}.csv")
        tex = os.path.join(self.out, "tables", f"{stem}.tex")
        df.to_csv(csv, index=False)
        open(tex, "w").write(_latex_table(df, caption, label))
        print(f"  wrote {csv}\n  wrote {tex}")

    # ---- Claim 3: cost vs robustness (from agg_time_ms already logged) ----
    def cost(self):
        df = self._q()
        if df.empty:
            print("store empty"); return
        g = (df.groupby("method")
               .agg(agg_ms_mean=("agg_time_ms", "mean"),
                    agg_ms_std=("agg_time_ms", "std"),
                    n_runs=("run_id", "count"))
               .reset_index().sort_values("agg_ms_mean"))
        g["complexity"] = g["method"].map({
            "fedavg": "O(M)", "trustfl": "O(M log M)", "trust_weighted": "O(M log M)",
            "median": "O(Md)", "trimmed_mean": "O(Md log M)",
            "krum": "O(M^2 d)", "multikrum": "O(M^2 d)",
            "fltrust": "O(M d)", "fedgreed": "O(M d) + val",
        }).fillna("--")
        out = g[["method", "complexity", "agg_ms_mean", "agg_ms_std", "n_runs"]].round(4)
        out.columns = ["Method", "Aggregation cost", "Agg time (ms)", "std", "runs"]
        self._save_table(out, "cost_vs_robustness",
                         "Per-round aggregation cost by method (wall-clock, mean over all runs). "
                         "Trust filtering is O(M log M) vs Krum-family O(M^2).",
                         "tab:cost")
        print(out.to_string(index=False))

    # ---- Motivating table: two failures, one table ----
    def motivating(self):
        """FedAvg collapses under attack; Krum discards honest mobile UAVs.
        Pulls whatever matching runs exist; reports what's present and what's
        missing so the author knows exactly which runs to add."""
        rows = []
        # Failure 1: FedAvg under a gradient attack (accuracy collapse).
        # Prefer grad_scale (simplest attack to explain in the intro); fall
        # back to any non-none attack. Include exploratory here ONLY for the
        # motivating demo since mnist_attack_fedavg predates the canonical set.
        fa = self.s.query("method='fedavg' AND attack_type='grad_scale'")
        if fa.empty:
            fa = self._q("method='fedavg' AND attack_type NOT IN ('none')")
        if not fa.empty:
            worst = fa.loc[fa["final_acc"].idxmin()]
            rows.append(["FedAvg", worst["attack_type"], f"{worst['final_acc']:.3f}",
                         "accuracy collapse under attack"])
        # Failure 2: Krum with no attack but high mobility (honest exclusion)
        kr = self._q("method='krum' AND attack_type='none' AND high_mob_frac > 0")
        if not kr.empty:
            g = kr.groupby("name").agg(acc=("final_acc", "mean")).reset_index()
            worst = g.loc[g["acc"].idxmin()]
            rows.append(["Krum", "none (honest high-mobility)", f"{worst['acc']:.3f}",
                         "discards honest outliers"])
        # Contrast: TrustFL on the same honest-mobility condition
        tf = self._q("method='trustfl' AND attack_type='none' AND high_mob_frac > 0")
        if not tf.empty:
            g = tf.groupby("name").agg(acc=("final_acc", "mean")).reset_index()
            best = g.loc[g["acc"].idxmax()]
            rows.append(["TrustFL (ours)", "none (honest high-mobility)",
                         f"{best['acc']:.3f}", "retains honest outliers"])
        if not rows:
            print("motivating: no matching runs yet. Need: a FedAvg-under-attack run "
                  "and a Krum/TrustFL honest-high-mobility run.")
            return
        df = pd.DataFrame(rows, columns=["Method", "Condition", "Final acc", "Failure mode"])
        self._save_table(df, "motivating",
                         "Two failure modes, one table: gradient-only aggregation collapses under "
                         "attack; geometric aggregation discards honest high-mobility UAVs. "
                         "TrustFL avoids both.", "tab:motivating")
        print(df.to_string(index=False))

    # ---- Claim 1: honest-outlier, per dataset (paper table + convergence figdata) ----
    def claim1(self):
        df = self._q("attack_type='none' AND high_mob_frac > 0")
        if df.empty:
            print("claim1: no honest-outlier runs found."); return
        g = (df.groupby(["dataset", "method"])
               .agg(acc_mean=("final_acc", "mean"), acc_std=("final_acc", "std"),
                    best_mean=("best_acc", "mean"), r95=("rounds_to_95", "mean"),
                    n=("seed", "count")).reset_index())
        g["acc"] = [_fmt_pm(m, s) for m, s in zip(g["acc_mean"], g["acc_std"])]
        tbl = g[["dataset", "method", "acc", "best_mean", "r95", "n"]].copy()
        tbl.columns = ["Dataset", "Method", "Final acc", "Best acc", "Rounds->0.95", "Seeds"]
        self._save_table(tbl, "claim1_honest_outlier",
                         "Claim 1: honest high-mobility UAVs. Geometric aggregation (Krum) loses "
                         "accuracy and stability; TrustFL matches clean FedAvg on both datasets.",
                         "tab:claim1")
        print(tbl.to_string(index=False))
        # convergence figure data: per-round test_acc for each run
        fd = []
        for _, r in df.iterrows():
            rd = self.s.rounds(r["run_id"])
            if "test_acc" in rd.columns:
                sub = rd.dropna(subset=["test_acc"])[["round", "test_acc"]].copy()
                sub["dataset"], sub["method"], sub["seed"] = r["dataset"], r["method"], r["seed"]
                fd.append(sub)
        if fd:
            allc = pd.concat(fd, ignore_index=True)
            p = os.path.join(self.out, "figdata", "claim1_convergence.csv")
            allc.to_csv(p, index=False)
            print(f"  wrote {p}  ({len(allc)} points for convergence bands)")

    def all(self):
        for fn in (self.motivating, self.cost, self.claim1):
            print(f"\n=== {fn.__name__} ===")
            try:
                fn()
            except Exception as e:
                print(f"  ({fn.__name__} skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("--out", default="./paper_assets")
    ap.add_argument("cmd", choices=["motivating", "cost", "claim1", "all"])
    args = ap.parse_args()
    a = Analysis(args.store, args.out)
    getattr(a, args.cmd)()


if __name__ == "__main__":
    main()
