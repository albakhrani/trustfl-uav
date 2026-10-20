"""Scalability figure (F8): aggregation cost versus swarm size.

Standalone regenerator for the scalability panel (R06). Reads the committed
micro-benchmark table paper_assets/data/scalability.csv (method, M, ms_mean,
ms_std, reps; produced by bench_scalability_v2.py and NOT regenerated here)
and draws aggregation time against swarm size M on log-log axes, one curve
per method, with the +/- one-std band of the repetitions.

Caption data are computed from the same CSV and printed / written to a JSON
sidecar next to the figure:
  * log-log slope (scaling exponent) per method, fitted over M >= 128
    (the regime the paper quotes: TrustFL ~1.1, Krum ~1.5, Multi-Krum ~1.4)
    and over all M;
  * the Krum/TrustFL cost ratio at M = 64 and M = 512.

    python paper_assets/make_fig_scalability.py [--data CSV] [--out DIR]

Outputs <out>/F8_scalability.{pdf,png} and <out>/F8_scalability_fit.json.
Style matches make_fig_eurosat.py / figures_tmc.py (Okabe-Ito palette,
distinct markers and dashes so the figure survives grayscale print, IEEE
single-column width, lossless PNG).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "scalability.csv")
OUT = os.path.join(HERE, "figures")

COLORS = {"trustfl": "#D55E00", "sophon": "#8B4789", "fedaa": "#B8860B",
          "fltrust": "#0072B2", "fedgreed": "#009E73", "multikrum": "#56B4E9",
          "fedavg": "#666666", "krum": "#CC79A7"}
LABELS = {"trustfl": "TrustFL (ours)", "sophon": "Sophon", "fedaa": "FedAA",
          "fltrust": "FLTrust", "fedgreed": "FedGreed",
          "multikrum": "Multi-Krum", "fedavg": "FedAvg", "krum": "Krum"}
LS = {"trustfl": "-", "sophon": "--", "fedaa": "-.", "fltrust": ":",
      "fedgreed": (0, (3, 1, 1, 1)), "multikrum": (0, (5, 2)),
      "fedavg": (0, (1, 1)), "krum": (0, (4, 1, 1, 1, 1, 1))}
MK = {"trustfl": "*", "sophon": "D", "fedaa": "P", "fltrust": "s",
      "fedgreed": "^", "multikrum": "v", "fedavg": "o", "krum": "X"}
ORDER = ["trustfl", "fedavg", "fltrust", "fedgreed", "sophon", "fedaa",
         "multikrum", "krum"]
FIT_MIN_M = 128

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "lines.linewidth": 1.4, "grid.alpha": 0.25,
})


def slope(sub: pd.DataFrame, m_min: int | None = None) -> float:
    """Log-log least-squares slope of ms_mean against M (scaling exponent)."""
    if m_min is not None:
        sub = sub[sub["M"] >= m_min]
    sub = sub.sort_values("M")
    return float(np.polyfit(np.log(sub["M"].values),
                            np.log(sub["ms_mean"].values), 1)[0])


def caption_data(df: pd.DataFrame) -> dict:
    out = {"fit_min_M": FIT_MIN_M, "exponent_M_ge_128": {}, "exponent_all_M": {},
           "ratio_krum_over_trustfl": {}}
    for m in sorted(df["method"].unique()):
        sub = df[df["method"] == m]
        out["exponent_M_ge_128"][m] = round(slope(sub, FIT_MIN_M), 4)
        out["exponent_all_M"][m] = round(slope(sub), 4)
    for M in (64, 512):
        at = df[df["M"] == M].set_index("method")["ms_mean"]
        if "krum" in at and "trustfl" in at:
            out["ratio_krum_over_trustfl"][str(M)] = round(float(at["krum"] / at["trustfl"]), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data)
    cap = caption_data(df)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))          # IEEE single column
    for m in ORDER:
        sub = df[df["method"] == m].sort_values("M")
        if sub.empty:
            continue
        ours = m == "trustfl"
        x, y, s = sub["M"].values, sub["ms_mean"].values, sub["ms_std"].values
        ax.plot(x, y, color=COLORS[m], ls=LS[m], marker=MK[m],
                lw=2.2 if ours else 1.2, ms=8 if ours else 4.5,
                markeredgecolor="black" if ours else "none", markeredgewidth=0.7,
                alpha=1.0 if ours else 0.85, zorder=5 if ours else 2,
                label=f"{LABELS[m]} ($\\alpha$={cap['exponent_M_ge_128'][m]:.2f})")
        ax.fill_between(x, np.maximum(y - s, 1e-3), y + s, color=COLORS[m],
                        alpha=0.12, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(df["M"].unique()))
    ax.set_xticklabels([str(int(v)) for v in sorted(df["M"].unique())])
    ax.set_xlabel("swarm size $M$ (UAVs)")
    ax.set_ylabel("aggregation time per round (ms)")
    ax.set_title("Aggregation cost vs. swarm size\n"
                 f"($\\alpha$: log-log exponent, fit over $M\\geq{FIT_MIN_M}$)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, handlelength=2.4,
              handletextpad=0.4, borderaxespad=0.2)

    pdf = os.path.join(a.out, "F8_scalability.pdf")
    png = os.path.join(a.out, "F8_scalability.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")   # PNG is lossless
    plt.close(fig)
    with open(os.path.join(a.out, "F8_scalability_fit.json"), "w") as f:
        json.dump(cap, f, indent=1)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print("exponents (M >= 128):", cap["exponent_M_ge_128"])
    print("exponents (all M):   ", cap["exponent_all_M"])
    print("krum/trustfl ratio:  ", cap["ratio_krum_over_trustfl"])


if __name__ == "__main__":
    main()
