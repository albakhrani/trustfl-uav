"""EuroSAT figure (F9): the retention-vs-influence plane under attack.

Standalone regenerator for the EuroSAT panel. Reads the exported per-run
aggregates (paper_assets/data/runs_summary.csv), keeps the eur_adsys family
(EuroSAT, 40% adaptive system+model attack, 3 seeds), and draws one point per
method on the paper's signature plane:

    y = honest-data retention (ret_highmob, higher = keeps honest UAVs)
    x = malicious influence share (infl_late, lower = benches attackers)

TrustFL alone occupies the top-left region (keeps honest data AND benches
attackers); every baseline sits lower. Per-seed scatter shows spread. Style
matches the main figure system (Okabe-Ito colorblind-safe palette, distinct
marker shapes so the figure survives grayscale print, IEEE column width).

    python paper_assets/make_fig_eurosat.py

Outputs paper_assets/figures/F9_eurosat.{pdf,png}.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "runs_summary.csv")
OUT = os.path.join(HERE, "figures")

# Okabe-Ito colorblind-safe palette; ours gets the single saturated highlight.
COLORS = {
    "trustfl": "#D55E00", "sophon": "#8B4789", "fedaa": "#B8860B",
    "fedavg": "#999999", "multikrum": "#56B4E9",
}
MARKERS = {
    "trustfl": "*", "sophon": "D", "fedaa": "P",
    "fedavg": "o", "multikrum": "v",
}
LABELS = {
    "trustfl": "TrustFL (ours)", "sophon": "Sophon", "fedaa": "FedAA",
    "fedavg": "FedAvg", "multikrum": "Multi-Krum",
}
ORDER = ["trustfl", "fedavg", "multikrum", "sophon", "fedaa"]

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "lines.linewidth": 1.4, "grid.alpha": 0.25,
})


def load_eurosat():
    df = pd.read_csv(DATA)
    df = df[df["name"].str.startswith("eur_adsys_")].copy()
    df["method"] = df["name"].str.replace("eur_adsys_", "", regex=False)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_eurosat()

    fig, ax = plt.subplots(figsize=(3.5, 3.2))          # IEEE single column
    for m in ORDER:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        ours = m == "trustfl"
        c = COLORS[m]
        # per-seed scatter (spread), muted
        ax.scatter(sub["infl_late"], sub["ret_highmob"], s=14 if ours else 10,
                   color=c, alpha=0.35, marker=MARKERS[m], zorder=2,
                   linewidths=0)
        # method mean (the point)
        mx, my = sub["infl_late"].mean(), sub["ret_highmob"].mean()
        ax.scatter([mx], [my], s=180 if ours else 70, color=c,
                   marker=MARKERS[m], edgecolors="white",
                   linewidths=0.8, zorder=5 if ours else 3,
                   alpha=1.0 if ours else 0.9, label=LABELS[m])

    ax.set_xlabel("malicious influence share (lower is better)")
    ax.set_ylabel("honest-data retention (higher is better)")
    ax.set_title("EuroSAT — 40% adaptive system+model attack")
    ax.set_ylim(0.30, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.3,
              borderaxespad=0.2)

    pdf = os.path.join(OUT, "F9_eurosat.pdf")
    png = os.path.join(OUT, "F9_eurosat.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
