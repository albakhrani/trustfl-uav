"""figures_tmc — the paper's figures, generated ONLY from paper_assets/data/*.csv.

Architecture: store -> trustfl.export (audited metrics, one code path) ->
CSV artifacts -> THIS module (pure presentation). If a number is wrong, fix
export once; if a panel is wrong, fix here. Never both blindly.

Seven 2x2 composites (TMC style), IEEE widths, 300 DPI, grayscale-safe
(unique dash + marker + hatch per method):
  F1 motivation   F2 distinction   F3 paradigm    F4 system attacks
  F5 sweep        F6 resource      F7 mechanism
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = "./paper_assets/data"
OUT = "./paper_assets/figures"

FULL = 7.16
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
HT = {"trustfl": None, "sophon": "//", "fedaa": "xx", "fltrust": "..",
      "fedgreed": "\\\\", "multikrum": "++", "fedavg": "||", "krum": "oo"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "lines.linewidth": 1.3, "grid.alpha": 0.22, "hatch.linewidth": 0.6,
})


def _panel(ax, letter, title=""):
    ax.text(0.02, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom")
    if title:
        ax.set_title(title, fontsize=8.5, pad=10)
    ax.grid(True)


def _line(ax, m, x, mean, lo=None, hi=None, every=10):
    ours = m == "trustfl"
    ax.plot(x, mean, color=COLORS[m], ls=LS[m], lw=2.2 if ours else 1.2,
            alpha=1.0 if ours else 0.85, marker=MK[m], markevery=every,
            ms=8 if ours else 4.5, markerfacecolor=COLORS[m],
            markeredgecolor="black" if ours else "none", markeredgewidth=0.7,
            zorder=5 if ours else 2, label=LABELS[m])
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.13, lw=0)


def _bars(ax, methods, vals, ylabel="", fmt="{:.2f}", ylim=None, rot=0):
    for i, m in enumerate(methods):
        ours = m == "trustfl"
        v = vals[m]
        ax.bar(i, np.mean(v), width=0.66, color=COLORS[m],
               alpha=1.0 if ours else 0.82, hatch=HT[m],
               edgecolor="black", linewidth=1.3 if ours else 0.5)
        ax.scatter([i] * len(v), v, s=9, color="black", alpha=0.55, zorder=4)
        top = max(np.mean(v), max(v))
        ax.text(i, top + (0.03 * (ylim[1] if ylim else max(top, 1))),
                fmt.format(np.mean(v)), ha="center", fontsize=6.5,
                fontweight="bold" if ours else "normal")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)")
                        for m in methods], fontsize=6.5, rotation=rot,
                       ha="right" if rot else "center")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def _smooth(y, win=5):
    if len(y) <= win:
        return np.asarray(y, dtype=float)
    return np.convolve(y, np.ones(win) / win, mode="valid")


class Hub:
    """Loads the CSV artifacts once; provides tidy accessors."""
    def __init__(self, data=DATA):
        self.runs = pd.read_csv(f"{data}/runs_summary.csv")
        suf = "_" + self.runs["method"].astype(str)
        name = self.runs["name"].astype(str)
        fam = [n[: -len(s)] if n.endswith(s) else n
               for n, s in zip(name, suf)]
        self.runs["family"] = fam
        self.ts = pd.read_csv(f"{data}/timeseries_key.csv")
        m = self.ts["name"].astype(str)
        suf2 = "_" + self.ts["method"].astype(str)
        self.ts["family"] = [n[: -len(s)] if n.endswith(s) else n
                             for n, s in zip(m, suf2)]
        try:
            self.jump = pd.read_csv(f"{data}/fidelity_jump.csv")
        except Exception:
            self.jump = None
        try:
            self.sweep = pd.read_csv(f"{data}/sweep.csv")
        except Exception:
            self.sweep = None

    def vals(self, family, method, col):
        r = self.runs[(self.runs["family"] == family)
                      & (self.runs["method"] == method)][col].dropna()
        return list(r.values)

    def band(self, family, method, col, smooth_win=None):
        sub = self.ts[(self.ts["family"] == family)
                      & (self.ts["method"] == method)].dropna(subset=[col])
        if sub.empty:
            return None
        curves = [g.sort_values("round").set_index("round")[col]
                  for _, g in sub.groupby("run_id")]
        L = min(len(c) for c in curves)
        if L == 0:
            return None
        arr = np.stack([c.values[:L] for c in curves])
        x = curves[0].index.values[:L]
        mu, lo, hi = arr.mean(0), arr.min(0), arr.max(0)
        if smooth_win:
            mu, lo, hi = (_smooth(mu, smooth_win), _smooth(lo, smooth_win),
                          _smooth(hi, smooth_win))
            x = x[smooth_win - 1:] if len(x) > smooth_win else x[: len(mu)]
        return x, mu, lo, hi


class TMC:
    def __init__(self, data=DATA, out=OUT):
        self.h = Hub(data)
        self.out = out
        os.makedirs(out, exist_ok=True)

    def _save(self, fig, name):
        p = os.path.join(self.out, name)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    def _figlegend(self, fig, methods, ncol=None):
        handles = [plt.Line2D([0], [0], color=COLORS[m], ls=LS[m],
                              marker=MK[m], ms=6, lw=1.6, label=LABELS[m])
                   for m in methods]
        fig.legend(handles=handles, loc="upper center",
                   ncol=ncol or len(methods), frameon=False,
                   bbox_to_anchor=(0.5, 1.04), columnspacing=1.2,
                   handlelength=2.4)

    # ---------------------------------------------------------------- F1 --
    def fig1(self):
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(FULL, 5.0))
        for fam, m in [("mnist_attack", "fedavg"), ("mnist_trustfl_debug", "trustfl")]:
            band = self.h.band(fam, m, "test_acc")
            if band:
                _line(a, m, *band)
        a.set_xlabel("round"); a.set_ylabel("test accuracy")
        a.set_ylim(0, 1.02); a.legend(frameon=False, loc="center right")
        _panel(a, "a", "No defense: magnitude attack\ncollapses FedAvg")

        methods = ["krum", "multikrum", "fltrust", "trustfl"]
        vals = {m: self.h.vals("exp_a", m, "ret_highmob") for m in methods}
        vals = {m: v for m, v in vals.items() if v}
        _bars(b, [m for m in methods if m in vals], vals,
              ylabel="honest high-mobility retention", ylim=(0, 1.15))
        _panel(b, "b", "Defense on, NO attack:\ngeometric filters discard honest UAVs")

        for m in ["krum", "multikrum", "trustfl"]:
            band = self.h.band("exp_a", m, "ret_highmob", smooth_win=5)
            if band:
                _line(c, m, *band)
        c.set_xlabel("round"); c.set_ylabel("retention (5-round mean)")
        c.set_ylim(-0.02, 1.05); c.legend(frameon=False, loc="center right")
        _panel(c, "c", "The exclusion, live")

        for m in ["krum", "trustfl"]:
            band = self.h.band("exp_a", m, "test_acc")
            if band:
                _line(d, m, *band)
        d.set_xlabel("round"); d.set_ylabel("test accuracy")
        d.legend(frameon=False, loc="lower right")
        _panel(d, "d", "The consequence:\nunstable, slower convergence")
        fig.tight_layout()
        self._save(fig, "F1_motivation.pdf")

    # ---------------------------------------------------------------- F2 --
    def fig2(self):
        methods = ["trustfl", "sophon", "fedaa", "fltrust",
                   "fedgreed", "multikrum", "fedavg"]
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(FULL, 5.4))
        for panel, fam, ds, letter in [(a, "exp_h_adsys", "MNIST", "a"),
                                       (b, "femh_adsys", "FEMNIST", "b")]:
            for m in methods:
                rets = self.h.vals(fam, m, "ret_highmob")
                infl = self.h.vals(fam, m, "infl_late")
                if not rets or not infl:
                    continue
                n = min(len(rets), len(infl))
                ours = m == "trustfl"
                panel.scatter(infl[:n], rets[:n], s=12, color=COLORS[m],
                              alpha=0.35, edgecolors="none", zorder=2)
                panel.scatter([np.mean(infl)], [np.mean(rets)],
                              s=190 if ours else 70, color=COLORS[m],
                              marker=MK[m], edgecolors="black",
                              linewidths=1.2 if ours else 0.6,
                              zorder=6 if ours else 3, label=LABELS[m])
            panel.axhspan(0.8, 1.05, color="green", alpha=0.05)
            panel.text(0.985, 0.97, "keeps honest data",
                       transform=panel.transAxes, ha="right", va="top",
                       fontsize=6.5, color="#2a7a2a")
            panel.axvline(0.4, ls="--", color="#888", lw=1)
            panel.set_xlim(0, 0.55); panel.set_ylim(0, 1.05)
            panel.set_xlabel("malicious influence share")
            panel.set_ylabel("honest high-mob. retention")
            _panel(panel, letter, f"{ds}: retention vs influence\n"
                                  "(40% adaptive system adversary)")
        for panel, fam, ds, letter in [(c, "exp_h_adsys", "MNIST", "c"),
                                       (d, "femh_adsys", "FEMNIST", "d")]:
            ms = [m for m in methods if self.h.vals(fam, m, "ret_highmob")]
            for i, m in enumerate(ms):
                rets = self.h.vals(fam, m, "ret_highmob")
                ours = m == "trustfl"
                panel.scatter([i] * len(rets), rets, s=26, color=COLORS[m],
                              marker=MK[m], alpha=0.85,
                              edgecolors="black" if ours else "none",
                              linewidths=0.7, zorder=4)
                panel.plot([i - 0.22, i + 0.22], [np.mean(rets)] * 2,
                           color="black", lw=1.6, zorder=5)
            panel.set_xticks(range(len(ms)))
            panel.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)")
                                   for m in ms], fontsize=6, rotation=14,
                                  ha="right")
            panel.set_ylabel("retention (per seed)")
            panel.set_ylim(0, 1.05)
            _panel(panel, letter, f"{ds}: per-seed spread")
        fig.tight_layout()
        self._save(fig, "F2_distinction.pdf")

    # ---------------------------------------------------------------- F3 --
    def fig3(self):
        methods = [m for m in ["sophon", "fedaa", "fltrust", "trustfl"]
                   if self.h.vals("fid_signflip", m, "infl_late")
                   and self.h.vals("exp_h_adsys", m, "infl_late")]
        if not methods:
            print("  fig3: fidelity data missing"); return
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(FULL, 5.0))
        w = 0.36
        for i, m in enumerate(methods):
            g = self.h.vals("fid_signflip", m, "infl_late")
            s = self.h.vals("exp_h_adsys", m, "infl_late")
            a.bar(i - w / 2, np.mean(g), width=w, color=COLORS[m], alpha=0.45,
                  edgecolor="black", linewidth=0.5,
                  label="gradient (sign-flip, 30%)" if i == 0 else None)
            a.bar(i + w / 2, np.mean(s), width=w, color=COLORS[m],
                  alpha=1.0 if m == "trustfl" else 0.85, hatch=HT[m],
                  edgecolor="black", linewidth=1.2 if m == "trustfl" else 0.5,
                  label="system (adaptive, 40%)" if i == 0 else None)
            for dx, vv in [(-w / 2, g), (w / 2, s)]:
                a.scatter([i + dx] * len(vv), vv, s=8, color="black",
                          alpha=0.55, zorder=4)
        a.axhline(0.3, ls=":", color="#888", lw=1)
        a.axhline(0.4, ls="--", color="#888", lw=1)
        a.set_xticks(range(len(methods)))
        a.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)")
                           for m in methods], fontsize=6.5)
        a.set_ylabel("malicious influence share")
        a.legend(frameon=False, loc="upper left", fontsize=6.5)
        _panel(a, "a", "Influence under each paradigm's\nblind-spot attack")

        if self.h.jump is not None:
            for i, m in enumerate(methods):
                row = self.h.jump[self.h.jump["method"] == m]
                if row.empty:
                    continue
                j = float(row["jump_factor"].iloc[0])
                ours = m == "trustfl"
                b.bar(i, j, width=0.62, color=COLORS[m], hatch=HT[m],
                      alpha=1.0 if ours else 0.82, edgecolor="black",
                      linewidth=1.3 if ours else 0.5)
                b.text(i, j + 0.2, f"{j:.1f}\u00d7", ha="center", fontsize=7.5,
                       fontweight="bold" if ours else "normal")
            b.set_xticks(range(len(methods)))
            b.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)")
                               for m in methods], fontsize=6.5)
            b.set_ylabel("influence jump (system / gradient)")
        _panel(b, "b", "The jump measures the blind spot")

        for panel, fam, letter, ttl in [
                (c, "fid_signflip", "c",
                 "Dynamics under sign-flip (their turf)\n5-round rolling mean"),
                (d, "exp_h_adsys", "d",
                 "Dynamics under system attack (our turf)\n5-round rolling mean")]:
            for m in methods:
                band = self.h.band(fam, m, "mal_influence", smooth_win=5)
                if band:
                    x, mu, lo, hi = band
                    if letter == "d":
                        _line(panel, m, x, mu, every=12)
                    else:
                        _line(panel, m, x, mu, lo, hi, every=12)
            panel.axhline(0.3 if fam == "fid_signflip" else 0.4,
                          ls="--", color="#888", lw=1)
            panel.set_xlabel("round"); panel.set_ylabel("malicious influence")
            panel.set_ylim(-0.02, 0.6)
            _panel(panel, letter, ttl)
        c.legend(frameon=False, fontsize=6, loc="upper right")
        fig.tight_layout()
        self._save(fig, "F3_paradigm.pdf")

    # ---------------------------------------------------------------- F4 --
    def fig4(self):
        methods = ["trustfl", "fltrust", "fedgreed", "fedavg"]
        fams = [("exp_b_selective_participation", "Selective participation"),
                ("exp_b2_delay", "Strategic delay (recalibrated)"),
                ("exp_b_resource_misreport", "Resource misreporting")]
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        panels = [ax[0][0], ax[0][1], ax[1][0]]
        for (fam, ttl), panel, letter in zip(fams, panels, "abc"):
            vals = {m: self.h.vals(fam, m, "infl_late") for m in methods}
            vals = {m: v for m, v in vals.items() if v}
            if vals:
                ymax = max(max(v) for v in vals.values())
                _bars(panel, [m for m in methods if m in vals], vals,
                      ylabel="malicious influence", fmt="{:.3f}",
                      ylim=(0, ymax * 1.4 + 1e-3))
            _panel(panel, letter, ttl)
        d = ax[1][1]
        for m in methods:
            band = self.h.band("exp_b_selective_participation", m,
                               "mal_influence", smooth_win=5)
            if band:
                x, mu, lo, hi = band
                _line(d, m, x, mu, every=12)
        d.set_xlabel("round"); d.set_ylabel("malicious influence")
        d.legend(frameon=False, fontsize=6)
        _panel(d, "d", "Selective participation: dynamics\n5-round rolling mean")
        fig.tight_layout()
        self._save(fig, "F4_system_attacks.pdf")

    # ---------------------------------------------------------------- F5 --
    def fig5(self):
        if self.h.sweep is None or self.h.sweep.empty:
            print("  fig5: sweep.csv missing"); return
        sw = self.h.sweep
        methods = ["trustfl", "fltrust", "fedgreed", "multikrum", "fedavg"]
        spec = [("ret_highmob", "honest retention", (0, 1.05)),
                ("deliv_eff", "delivery efficiency", (0, 1.05)),
                ("infl_late", "malicious influence", None),
                ("final_acc", "final accuracy", (0.9, 1.0))]
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        axes = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
        for (col, ylab, ylim), panel, letter in zip(spec, axes, "abcd"):
            for m in methods:
                sub = sw[sw["method"] == m]
                if sub.empty:
                    continue
                g = sub.groupby("fraction")[col]
                fr = sorted(g.groups.keys())
                mu = g.mean().reindex(fr).values
                lo = g.min().reindex(fr).values
                hi = g.max().reindex(fr).values
                _line(panel, m, fr, mu, lo, hi, every=1)
            panel.set_xlabel("malicious fraction")
            panel.set_ylabel(ylab)
            panel.set_xticks(sorted(sw["fraction"].unique()))
            if ylim:
                panel.set_ylim(*ylim)
            _panel(panel, letter, ylab.capitalize() + " vs. threat level")
        self._figlegend(fig, methods, ncol=5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._save(fig, "F5_sweep.pdf")

    # ---------------------------------------------------------------- F6 --
    def fig6(self, fam="exp_h_adsys"):
        methods = ["trustfl", "sophon", "fedaa", "fltrust",
                   "fedgreed", "multikrum", "fedavg"]
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(FULL, 5.2))
        for m in methods:
            sub = self.h.ts[(self.h.ts["family"] == fam)
                            & (self.h.ts["method"] == m)]
            if sub.empty:
                continue
            curves = []
            for _, g in sub.groupby("run_id"):
                g = g.sort_values("round")
                curves.append(pd.Series(np.cumsum(g["n_delivered"].values),
                                        index=g["round"].values))
            L = min(len(cv) for cv in curves)
            arr = np.stack([cv.values[:L] for cv in curves])
            _line(a, m, curves[0].index.values[:L], arr.mean(0),
                  arr.min(0), arr.max(0), every=14)
        a.set_xlabel("round"); a.set_ylabel("cumulative useful updates")
        _panel(a, "a", "Useful work delivered over time")

        for m in methods:
            band = self.h.band(fam, m, "mean_energy_j")
            if band:
                _line(b, m, *band, every=14)
        b.set_xlabel("round"); b.set_ylabel("mean remaining energy (J)")
        _panel(b, "b", "Swarm energy depletion")

        vals = {m: self.h.vals(fam, m, "energy_per_update_j") for m in methods}
        vals = {m: v for m, v in vals.items() if v}
        if vals:
            ymax = max(max(v) for v in vals.values())
            _bars(c, [m for m in methods if m in vals], vals,
                  ylabel="energy per delivered update (J)", fmt="{:.0f}",
                  ylim=(0, ymax * 1.3), rot=14)
        _panel(c, "c", "Energy per delivered update\n(lower is better)")

        # R05: timings from the hardened family of this panel only, not the
        # mean over every run in the store
        cost = (self.h.runs[self.h.runs["family"] == fam]
                .groupby("method")["agg_time_ms"].mean())
        for m in methods:
            rets = self.h.vals(fam, m, "ret_highmob")
            if not rets or m not in cost.index or cost[m] != cost[m]:
                continue
            ours = m == "trustfl"
            d.scatter([cost[m]], [np.mean(rets)], s=170 if ours else 60,
                      color=COLORS[m], marker=MK[m], edgecolors="black",
                      linewidths=1.2 if ours else 0.6,
                      zorder=5 if ours else 3)
            d.annotate(LABELS[m].replace(" (ours)", ""), (cost[m], np.mean(rets)),
                       textcoords="offset points", xytext=(5, 4), fontsize=6,
                       fontweight="bold" if ours else "normal")
        d.set_xscale("log")
        d.set_xlabel("aggregation time (ms, log)")
        d.set_ylabel("retention under attack")
        _panel(d, "d", "Robustness per unit compute")
        self._figlegend(fig, methods, ncol=7)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._save(fig, "F6_resource.pdf")

    # ---------------------------------------------------------------- F7 --
    def fig7(self):
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(FULL, 5.0))
        for col, lab, cc in [("trust_honest", "honest UAVs", "#0072B2"),
                             ("trust_malicious", "malicious UAVs", "#D55E00")]:
            band = self.h.band("exp_h_adsys", "trustfl", col)
            if band:
                x, mu, lo, hi = band
                a.plot(x, mu, color=cc, lw=1.8, label=lab)
                a.fill_between(x, lo, hi, color=cc, alpha=0.15, lw=0)
        a.set_xlabel("round"); a.set_ylabel("mean trust")
        a.legend(frameon=False)
        _panel(a, "a", "Trust separates the populations\n(system attack)")

        names = [("w_beh", "behavioral", "#009E73", "-"),
                 ("w_comp", "computational", "#CC79A7", "--"),
                 ("w_comm", "communication", "#56B4E9", ":")]
        for panel, fam, letter, ttl in [
                (b, "exp_h_adsys", "b",
                 "Weights under CYCLING system attack:\ntracking the moving dimension"),
                (c, "fid_signflip", "c",
                 "Weights under sign-flip: convergence\nto pure behavioral (Thm. 2 verified)")]:
            for col, lab, cc, ls in names:
                band = self.h.band(fam, "trustfl", col)
                if band:
                    x, mu, lo, hi = band
                    panel.plot(x, mu, color=cc, ls=ls, lw=1.6, label=lab)
                    panel.fill_between(x, lo, hi, color=cc, alpha=0.13, lw=0)
            panel.set_ylim(-0.02, 1.05)
            panel.set_xlabel("round"); panel.set_ylabel("dimension weight")
            panel.legend(frameon=False, fontsize=6)
            _panel(panel, letter, ttl)

        abl = [("exp_c_adaptive", "trustfl", "full"),
               ("exp_c_ablate_symewma", None, "sym.\nEWMA"),
               ("exp_c_ablate_nogate", None, "no\ngate")]
        vals = {}
        for fam, m, lab in abl:
            if m is None:
                sub = self.h.runs[self.h.runs["name"] == fam]
                v = list(sub["final_acc"].dropna().values)
            else:
                v = self.h.vals(fam, m, "final_acc")
            if v:
                vals[lab] = v
        for i, (lab, v) in enumerate(vals.items()):
            col = "#D55E00" if lab == "full" else "#888888"
            d.bar(i, np.mean(v), width=0.6, color=col,
                  alpha=1.0 if lab == "full" else 0.7,
                  hatch=None if lab == "full" else "//",
                  edgecolor="black", linewidth=1.2 if lab == "full" else 0.5)
            d.scatter([i] * len(v), v, s=14, color="black", alpha=0.6, zorder=4)
        d.set_xticks(range(len(vals)))
        d.set_xticklabels(list(vals.keys()), fontsize=7)
        d.set_ylabel("final accuracy"); d.set_ylim(0, 1.1)
        _panel(d, "d", "Ablation: removing the consistency\ngate collapses robustness")
        fig.tight_layout()
        self._save(fig, "F7_mechanism.pdf")

    def all(self):
        for f in [self.fig1, self.fig2, self.fig3, self.fig4,
                  self.fig5, self.fig6, self.fig7]:
            try:
                f()
            except Exception as e:
                print(f"  {f.__name__}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("cmd", choices=["fig1", "fig2", "fig3", "fig4", "fig5",
                                    "fig6", "fig7", "all"])
    a = ap.parse_args()
    t = TMC(a.data)
    getattr(t, a.cmd if a.cmd != "all" else "all")()


if __name__ == "__main__":
    main()
