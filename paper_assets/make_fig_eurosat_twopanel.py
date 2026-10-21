#!/usr/bin/env python3
"""make_fig_eurosat_twopanel.py — Fig. 10 of the manuscript (F9_eurosat), the
two-panel EuroSAT pilot figure, generated ONLY from paper_assets/data/runs_summary.csv.

  (a) EuroSAT per-seed honest high-mobility retention, one marker per seed and
      the three-seed mean as a bar, with the worst TrustFL seed and the best
      baseline seed annotated (values read from the CSV, never typed);
  (b) mean honest high-mobility retention of the pilot's five methods on
      MNIST (exp_h_adsys), FEMNIST (femh_adsys) and EuroSAT (eur_adsys).

Style follows trustfl/figures_tmc.py (same colours, markers, dashes).
usage: python3 make_fig_eurosat_twopanel.py --data paper_assets/data/runs_summary.csv --out paper_assets/figures
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"trustfl": "#D55E00", "sophon": "#8B4789", "fedaa": "#B8860B",
          "fltrust": "#0072B2", "fedgreed": "#009E73", "multikrum": "#56B4E9",
          "fedavg": "#666666", "krum": "#CC79A7"}
LABELS = {"trustfl": "TrustFL (ours)", "sophon": "Sophon", "fedaa": "FedAA",
          "fltrust": "FLTrust", "fedgreed": "FedGreed", "multikrum": "Multi-Krum",
          "fedavg": "FedAvg", "krum": "Krum"}
LS = {"trustfl": "-", "sophon": "--", "fedaa": "-.", "fltrust": ":",
      "fedgreed": (0, (3, 1, 1, 1)), "multikrum": (0, (5, 2)),
      "fedavg": (0, (1, 1)), "krum": (0, (4, 1, 1, 1, 1, 1))}
MK = {"trustfl": "*", "sophon": "D", "fedaa": "P", "fltrust": "s",
      "fedgreed": "^", "multikrum": "v", "fedavg": "o", "krum": "X"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300, "lines.linewidth": 1.3, "grid.alpha": 0.22})

METHODS = ["trustfl", "fedavg", "multikrum", "sophon", "fedaa"]   # the pilot's five methods
FAMS = [("exp_h_adsys", "MNIST"), ("femh_adsys", "FEMNIST"), ("eur_adsys", "EuroSAT")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paper_assets/data/runs_summary.csv")
    ap.add_argument("--out", default="paper_assets/figures")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    fam = lambda f: df[df["name"].str.startswith(f + "_")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw={"wspace": 0.28})

    # (a) per-seed EuroSAT retention
    eur = fam("eur_adsys")
    xs = np.arange(len(METHODS))
    worst_trustfl = eur[eur.method == "trustfl"]["ret_highmob"].min()
    best_base = eur[eur.method != "trustfl"]["ret_highmob"].max()
    ax1.axhspan(best_base, worst_trustfl, color=COLORS["trustfl"], alpha=0.08, lw=0)
    for i, m in enumerate(METHODS):
        v = eur[eur.method == m].sort_values("seed")["ret_highmob"].values
        jitter = np.linspace(-0.12, 0.12, len(v))
        ax1.scatter(i + jitter, v, marker=MK[m], s=70 if m == "trustfl" else 40, color=COLORS[m],
                    edgecolor="black" if m == "trustfl" else "none", linewidth=0.6, zorder=4)
        ax1.hlines(v.mean(), i - 0.22, i + 0.22, color="black", lw=1.8, zorder=5)
    ax1.annotate(f"worst TrustFL seed {worst_trustfl:.2f}", xy=(0.12, worst_trustfl), xytext=(0.9, 0.955),
                 fontsize=7, color=COLORS["trustfl"], arrowprops=dict(arrowstyle="-", color=COLORS["trustfl"], lw=0.6))
    ax1.annotate(f"best baseline seed {best_base:.2f}", xy=(1.12, best_base), xytext=(2.1, 0.75),
                 fontsize=7, color="#444444", arrowprops=dict(arrowstyle="-", color="#444444", lw=0.6))
    ax1.set_xticks(xs); ax1.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)") for m in METHODS], rotation=20)
    ax1.set_ylabel("honest high-mob. retention"); ax1.set_ylim(0.2, 1.0); ax1.grid(True, axis="y")
    ax1.set_title("(a) EuroSAT: per-seed retention")

    # (b) retention across tasks (means over seeds)
    for m in METHODS:
        means = [fam(f)[lambda d: d.method == m]["ret_highmob"].mean() for f, _ in FAMS]
        ax2.plot(range(3), means, color=COLORS[m], ls=LS[m], marker=MK[m], lw=2.4 if m == "trustfl" else 1.3,
                 ms=11 if m == "trustfl" else 6, markeredgecolor="black" if m == "trustfl" else "none",
                 markeredgewidth=0.6, label=LABELS[m], zorder=5 if m == "trustfl" else 3)
    ax2.set_xticks(range(3)); ax2.set_xticklabels([n for _, n in FAMS]); ax2.set_ylim(0.2, 1.0)
    ax2.grid(True, axis="y"); ax2.set_title("(b) Retention transfers across tasks")
    handles = [plt.Line2D([0], [0], color=COLORS[m], ls=LS[m], marker=MK[m], ms=6, lw=1.6, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.06), columnspacing=1.2, handlelength=2.4)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"F9_eurosat.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.out}/F9_eurosat.pdf/.png; worst TrustFL seed {worst_trustfl:.4f}, best baseline seed {best_base:.4f}")


if __name__ == "__main__":
    main()
