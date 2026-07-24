"""figures_v2 — journal-grade composite figures for the TMC paper.

Seven 2x2 composites, each a complete argument (one figure = one claim arc):
  fig1  Motivation: the two failures, live and in consequence.
  fig2  The distinction: signature scatters + per-seed stability strips.
  fig3  Paradigm blindness: endpoints, jump factors, and dynamics.
  fig4  System-dimension attacks vs gradient-only defenses.
  fig5  Threat-spectrum sweep (retention/delivery/influence/accuracy).
  fig6  Resource effectiveness: sustainment, depletion, energy, cost.
  fig7  Mechanism: trust separation, weight dynamics (x2), ablation.

Encoding system (grayscale-safe by construction):
  ours = solid vermillion + star;  every baseline gets a UNIQUE dash
  pattern, marker, and bar hatch, so no panel depends on colour alone.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit import _groups, _parse_dictcol, _parse_listcol
from .store import Store

# ---------------------------------------------------------------- style ---
FULL = 7.16          # IEEE double-column width (in)
COL = 3.5
COLORS = {
    "trustfl": "#D55E00", "sophon": "#8B4789", "fedaa": "#B8860B",
    "fltrust": "#0072B2", "fedgreed": "#009E73", "multikrum": "#56B4E9",
    "fedavg": "#666666", "krum": "#CC79A7",
}
LABELS = {
    "trustfl": "TrustFL (ours)", "sophon": "Sophon", "fedaa": "FedAA",
    "fltrust": "FLTrust", "fedgreed": "FedGreed", "multikrum": "Multi-Krum",
    "fedavg": "FedAvg", "krum": "Krum",
}
LINESTYLES = {
    "trustfl": "-", "sophon": "--", "fedaa": "-.", "fltrust": ":",
    "fedgreed": (0, (3, 1, 1, 1)), "multikrum": (0, (5, 2)),
    "fedavg": (0, (1, 1)), "krum": (0, (4, 1, 1, 1, 1, 1)),
}
MARKERS = {
    "trustfl": "*", "sophon": "D", "fedaa": "P", "fltrust": "s",
    "fedgreed": "^", "multikrum": "v", "fedavg": "o", "krum": "X",
}
HATCHES = {
    "trustfl": None, "sophon": "//", "fedaa": "xx", "fltrust": "..",
    "fedgreed": "\\\\", "multikrum": "++", "fedavg": "||", "krum": "oo",
}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "lines.linewidth": 1.3, "grid.alpha": 0.22,
    "hatch.linewidth": 0.6,
})


def _panel(ax, letter, title=""):
    ax.text(0.02, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom")
    if title:
        ax.set_title(title, fontsize=8.5, pad=10)
    ax.grid(True)


def _line(ax, m, x, mean, lo=None, hi=None, every=8):
    ours = m == "trustfl"
    ax.plot(x, mean, color=COLORS[m], ls=LINESTYLES[m],
            lw=2.2 if ours else 1.2, alpha=1.0 if ours else 0.85,
            marker=MARKERS[m], markevery=every, ms=8 if ours else 4.5,
            markerfacecolor=COLORS[m], markeredgecolor="black" if ours else "none",
            markeredgewidth=0.7, zorder=5 if ours else 2, label=LABELS[m])
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.13, lw=0)


def _bars(ax, methods, vals, dots=None, ylabel="", fmt="{:.2f}", ylim=None):
    for i, m in enumerate(methods):
        ours = m == "trustfl"
        ax.bar(i, np.mean(vals[m]), width=0.66, color=COLORS[m],
               alpha=1.0 if ours else 0.82, hatch=HATCHES[m],
               edgecolor="black", linewidth=1.3 if ours else 0.5)
        top = np.mean(vals[m])
        if dots and m in dots:
            ax.scatter([i] * len(dots[m]), dots[m], s=9, color="black",
                       alpha=0.55, zorder=4)
            top = max(top, max(dots[m]))
        ax.text(i, top + (0.02 if not ylim else 0.03 * ylim[1]),
                fmt.format(np.mean(vals[m])), ha="center", fontsize=6.5,
                fontweight="bold" if ours else "normal")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)")
                        for m in methods], fontsize=6.5)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


# ------------------------------------------------------------- data layer -
class Data:
    def __init__(self, root):
        self.s = Store(root)
        self.df = self.s.query()

    def runs(self, name):
        return sorted(self.df[self.df["name"] == name]["run_id"].tolist())

    def acc_band(self, name):
        curves = []
        for rid in self.runs(name):
            rd = self.s.rounds(rid).dropna(subset=["test_acc"])
            if len(rd):
                curves.append(rd.set_index("round")["test_acc"])
        return self._band(curves)

    def col_band(self, name, col):
        curves = []
        for rid in self.runs(name):
            rd = self.s.rounds(rid)
            if col in rd:
                c = rd.dropna(subset=[col]).set_index("round")[col]
                if len(c):
                    curves.append(c)
        return self._band(curves)

    def retention_series(self, rid):
        """Per-round honest high-mobility retention (audit loop, as series)."""
        mal, hm = _groups(self.s, rid)
        rd = self.s.rounds(rid)
        xs, ys = [], []
        for _, r in rd.iterrows():
            invited = {int(j) for j in _parse_listcol(r.get("invited"))}
            infl = _parse_dictcol(r.get("influence"))
            survived = {int(j) for j, w in infl.items() if w and w > 0}
            hmi = {j for j in invited if j not in mal and j in hm}
            if hmi:
                xs.append(r["round"])
                ys.append(len(hmi & survived) / len(hmi))
        return xs, ys

    def retention_band(self, name, win=5):
        curves = []
        for rid in self.runs(name):
            xs, ys = self.retention_series(rid)
            if ys:
                y = np.convolve(ys, np.ones(win) / win, mode="valid")
                x = xs[win - 1:]
                import pandas as pd
                curves.append(pd.Series(y, index=x))
        return self._band(curves)

    def retention_vals(self, name):
        from .audit import retention_for_run
        out = []
        for rid in self.runs(name):
            _, hm = retention_for_run(self.s, rid)
            if hm == hm:
                out.append(hm)
        return out

    def infl_late(self, name):
        out = []
        for rid in self.runs(name):
            rd = self.s.rounds(rid)
            mi = [v for v in rd.get("malicious_influence_share", []) if v == v]
            if mi:
                out.append(float(np.mean(mi[len(mi) // 2:])))
        return out

    def deliv_vals(self, name):
        out = []
        for rid in self.runs(name):
            rd = self.s.rounds(rid)
            eff = []
            for _, r in rd.iterrows():
                inv = _parse_listcol(r.get("invited"))
                dlv = _parse_listcol(r.get("delivered"))
                if inv:
                    eff.append(len(dlv) / len(inv))
            if eff:
                out.append(float(np.mean(eff)))
        return out

    def acc_vals(self, name):
        out = []
        for rid in self.runs(name):
            rd = self.s.rounds(rid).dropna(subset=["test_acc"])
            if len(rd):
                out.append(float(rd["test_acc"].iloc[-1]))
        return out

    def weights_band(self, name):
        per_dim = [[], [], []]
        for rid in self.runs(name):
            rd = self.s.rounds(rid)
            if "trust_weights" not in rd:
                continue
            tw = rd.dropna(subset=["trust_weights"])
            if not len(tw):
                continue
            W = np.array([(json.loads(v) if isinstance(v, str) else v)
                          for v in tw["trust_weights"]])
            import pandas as pd
            for d in range(min(3, W.shape[1])):
                per_dim[d].append(pd.Series(W[:, d], index=tw["round"].values))
        return [self._band(c) for c in per_dim]

    def col_band_smooth(self, name, col, win=5):
        band = self.col_band(name, col)
        if not band:
            return None
        x, mu, lo, hi = band
        if len(mu) <= win:
            return band
        k = np.ones(win) / win
        sm = lambda a: np.convolve(a, k, mode="valid")
        return x[win - 1:], sm(mu), sm(lo), sm(hi)

    @staticmethod
    def _band(curves):
        if not curves:
            return None
        L = min(len(c) for c in curves)
        arr = np.stack([c.values[:L] for c in curves])
        x = curves[0].index.values[:L]
        return x, arr.mean(0), arr.min(0), arr.max(0)


# ---------------------------------------------------------------- figures -
class Figs:
    def __init__(self, root, out="./paper_assets/figures"):
        self.d = Data(root)
        self.out = out
        os.makedirs(out, exist_ok=True)

    def _save(self, fig, name):
        p = os.path.join(self.out, name)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    def _figlegend(self, fig, methods, y=1.02, ncol=None):
        handles = [plt.Line2D([0], [0], color=COLORS[m], ls=LINESTYLES[m],
                              marker=MARKERS[m], ms=6, lw=1.6,
                              label=LABELS[m]) for m in methods]
        fig.legend(handles=handles, loc="upper center", ncol=ncol or len(methods),
                   frameon=False, bbox_to_anchor=(0.5, y),
                   columnspacing=1.2, handlelength=2.4)

    # ---- Fig 1: motivation -------------------------------------------------
    def fig1(self):
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        (a, b), (c, dd) = ax
        # (a) attack collapse
        for name, m in [("mnist_attack_fedavg", "fedavg"),
                        ("mnist_trustfl_debug", "trustfl")]:
            band = self.d.acc_band(name)
            if band:
                x, mu, lo, hi = band
                _line(a, m, x, mu, lo, hi, every=10)
        _panel(a, "a", "No defense: magnitude attack\ncollapses FedAvg")
        a.set_xlabel("round"); a.set_ylabel("test accuracy"); a.set_ylim(0, 1.02)
        a.legend(frameon=False, loc="center right")
        # (b) retention bars, no attack
        methods = ["krum", "multikrum", "fltrust", "trustfl"]
        vals = {m: self.d.retention_vals(f"exp_a_{m}") for m in methods}
        vals = {m: v for m, v in vals.items() if v}
        _bars(b, [m for m in methods if m in vals], vals, dots=vals,
              ylabel="honest high-mobility retention", ylim=(0, 1.12))
        _panel(b, "b", "Defense on, NO attack:\ngeometric filters discard honest UAVs")
        # (c) per-round retention trajectories, no attack
        for m in ["krum", "multikrum", "trustfl"]:
            band = self.d.retention_band(f"exp_a_{m}")
            if band:
                x, mu, lo, hi = band
                _line(c, m, x, mu, lo, hi, every=10)
        _panel(c, "c", "The exclusion, live:\nper-round retention (5-round mean)")
        c.set_xlabel("round"); c.set_ylabel("retention"); c.set_ylim(-0.02, 1.05)
        c.legend(frameon=False, loc="center right")
        # (d) convergence consequence
        for m in ["krum", "trustfl"]:
            band = self.d.acc_band(f"exp_a_{m}")
            if band:
                x, mu, lo, hi = band
                _line(dd, m, x, mu, lo, hi, every=10)
        _panel(dd, "d", "The consequence:\nunstable, slower convergence")
        dd.set_xlabel("round"); dd.set_ylabel("test accuracy")
        dd.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        self._save(fig, "F1_motivation.pdf")

    # ---- Fig 2: the distinction -------------------------------------------
    def fig2(self):
        methods = ["trustfl", "sophon", "fedaa", "fltrust",
                   "fedgreed", "multikrum", "fedavg"]
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.2))
        (a, b), (c, dd) = ax
        for panel, prefix, ds in [(a, "exp_h_adsys", "MNIST"),
                                  (b, "femh_adsys", "FEMNIST")]:
            any_data = False
            for m in methods:
                rets = self.d.retention_vals(f"{prefix}_{m}")
                infls = self.d.infl_late(f"{prefix}_{m}")
                if not rets or not infls:
                    continue
                any_data = True
                n = min(len(rets), len(infls))
                ours = m == "trustfl"
                panel.scatter(infls[:n], rets[:n], s=12, color=COLORS[m],
                              alpha=0.35, edgecolors="none", zorder=2)
                panel.scatter([np.mean(infls)], [np.mean(rets)],
                              s=190 if ours else 70, color=COLORS[m],
                              marker=MARKERS[m], edgecolors="black",
                              linewidths=1.2 if ours else 0.6,
                              zorder=6 if ours else 3, label=LABELS[m])
            if any_data:
                panel.axvspan(0, 1, ymin=0.75, color="green", alpha=0.05)
                panel.text(0.985, 0.97, "keeps honest data",
                           transform=panel.transAxes, ha="right", va="top",
                           fontsize=6.5, color="#2a7a2a")
                panel.axvline(0.4, ls="--", color="#888", lw=1)
            panel.set_xlabel("malicious influence share")
            panel.set_ylabel("honest high-mob. retention")
            panel.set_ylim(0, 1.05)
            panel.set_xlim(0, 0.55)
            _panel(panel, "a" if ds == "MNIST" else "b",
                   f"{ds}: retention vs influence\n(40% adaptive system adversary)")
        # (c)/(d) per-seed stability strips
        for panel, prefix, ds, letter in [(c, "exp_h_adsys", "MNIST", "c"),
                                          (dd, "femh_adsys", "FEMNIST", "d")]:
            ms = [m for m in methods if self.d.retention_vals(f"{prefix}_{m}")]
            for i, m in enumerate(ms):
                rets = self.d.retention_vals(f"{prefix}_{m}")
                ours = m == "trustfl"
                panel.scatter([i] * len(rets), rets, s=26, color=COLORS[m],
                              marker=MARKERS[m], alpha=0.85,
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
            _panel(panel, letter,
                   f"{ds}: per-seed spread\n(adaptive methods destabilized)")
        fig.tight_layout()
        self._save(fig, "F2_distinction.pdf")

    # ---- Fig 3: paradigm blindness -----------------------------------------
    def fig3(self):
        methods = ["sophon", "fedaa", "fltrust", "trustfl"]
        grad = {m: self.d.infl_late(f"fid_signflip_{m}") for m in methods}
        syst = {m: self.d.infl_late(f"exp_h_adsys_{m}") for m in methods}
        methods = [m for m in methods if grad[m] and syst[m]]
        if not methods:
            print("  fig3: fidelity runs missing"); return
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        (a, b), (c, dd) = ax
        # (a) endpoint bars, grouped, hatched
        w = 0.36
        for i, m in enumerate(methods):
            a.bar(i - w / 2, np.mean(grad[m]), width=w, color=COLORS[m],
                  alpha=0.45, edgecolor="black", linewidth=0.5,
                  label="gradient (sign-flip, 30%)" if i == 0 else None)
            a.bar(i + w / 2, np.mean(syst[m]), width=w, color=COLORS[m],
                  alpha=1.0 if m == "trustfl" else 0.85, hatch=HATCHES[m],
                  edgecolor="black", linewidth=1.2 if m == "trustfl" else 0.5,
                  label="system (adaptive, 40%)" if i == 0 else None)
            for dx, vals in [(-w / 2, grad[m]), (w / 2, syst[m])]:
                a.scatter([i + dx] * len(vals), vals, s=8, color="black",
                          alpha=0.55, zorder=4)
        a.axhline(0.3, ls=":", color="#888", lw=1)
        a.axhline(0.4, ls="--", color="#888", lw=1)
        a.text(len(methods) - 0.5, 0.405, "pop. share 40%", fontsize=6,
               color="#777", ha="right", va="bottom")
        a.text(len(methods) - 0.5, 0.305, "pop. share 30%", fontsize=6,
               color="#777", ha="right", va="bottom")
        a.set_xticks(range(len(methods)))
        a.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)")
                           for m in methods], fontsize=6.5)
        a.set_ylabel("malicious influence share")
        a.legend(frameon=False, loc="upper left", fontsize=6.5)
        _panel(a, "a", "Influence under each paradigm's\nblind-spot attack")
        # (b) jump factors
        for i, m in enumerate(methods):
            jump = np.mean(syst[m]) / max(np.mean(grad[m]), 1e-6)
            a_ours = m == "trustfl"
            b.bar(i, jump, width=0.62, color=COLORS[m], hatch=HATCHES[m],
                  alpha=1.0 if a_ours else 0.82, edgecolor="black",
                  linewidth=1.3 if a_ours else 0.5)
            b.text(i, jump + 0.18, f"{jump:.1f}\u00d7", ha="center",
                   fontsize=7.5, fontweight="bold" if a_ours else "normal")
        b.set_xticks(range(len(methods)))
        b.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)")
                           for m in methods], fontsize=6.5)
        b.set_ylabel("influence jump (system / gradient)")
        _panel(b, "b", "The jump measures the blind spot")
        # (c)/(d) dynamics
        for panel, prefix, letter, ttl in [
                (c, "fid_signflip", "c",
                 "Dynamics under sign-flip (their turf)\n5-round rolling mean"),
                (dd, "exp_h_adsys", "d",
                 "Dynamics under system attack (our turf)\n5-round rolling mean")]:
            for m in methods:
                band = self.d.col_band_smooth(f"{prefix}_{m}",
                                              "malicious_influence_share")
                if band:
                    x, mu, lo, hi = band
                    if letter == "d":
                        _line(panel, m, x, mu, None, None, every=12)
                    else:
                        _line(panel, m, x, mu, lo, hi, every=12)
            panel.axhline(0.3 if prefix == "fid_signflip" else 0.4,
                          ls="--", color="#888", lw=1)
            panel.set_xlabel("round")
            panel.set_ylabel("malicious influence")
            panel.set_ylim(-0.02, 0.75)
            _panel(panel, letter, ttl)
        c.legend(frameon=False, fontsize=6, loc="upper right")
        fig.tight_layout()
        self._save(fig, "F3_paradigm.pdf")

    # ---- Fig 4: system attacks vs gradient defenses ------------------------
    def fig4(self):
        attacks = [("exp_b_selective_participation", "Selective participation"),
                   ("exp_b2_delay", "Strategic delay (recalibrated)"),
                   ("exp_b_resource_misreport", "Resource misreporting")]
        methods = ["trustfl", "fltrust", "fedgreed", "fedavg"]
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        panels = [ax[0][0], ax[0][1], ax[1][0]]
        for (atk, ttl), panel, letter in zip(attacks, panels, "abc"):
            vals = {m: self.d.infl_late(f"{atk}_{m}") for m in methods}
            vals = {m: v for m, v in vals.items() if v}
            if vals:
                _bars(panel, [m for m in methods if m in vals], vals,
                      dots=vals, ylabel="malicious influence", fmt="{:.3f}",
                      ylim=(0, max(max(v) for v in vals.values()) * 1.35))
            _panel(panel, letter, ttl)
        dd = ax[1][1]
        for m in methods:
            band = self.d.col_band_smooth(f"exp_b_selective_participation_{m}",
                                          "malicious_influence_share")
            if band:
                x, mu, lo, hi = band
                _line(dd, m, x, mu, None, None, every=12)
        dd.set_xlabel("round"); dd.set_ylabel("malicious influence")
        dd.legend(frameon=False, fontsize=6)
        _panel(dd, "d", "Selective participation: dynamics")
        fig.tight_layout()
        self._save(fig, "F4_system_attacks.pdf")

    # ---- Fig 5: threat-spectrum sweep ---------------------------------------
    def fig5(self, fracs=(0.1, 0.2, 0.3, 0.4)):
        methods = ["trustfl", "fltrust", "fedgreed", "multikrum", "fedavg"]
        panels_spec = [
            ("retention", "honest retention", lambda n: self.d.retention_vals(n)),
            ("delivery", "delivery efficiency", lambda n: self.d.deliv_vals(n)),
            ("influence", "malicious influence", lambda n: self.d.infl_late(n)),
            ("accuracy", "final accuracy", lambda n: self.d.acc_vals(n)),
        ]
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        axes = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
        for (key, ylab, fn), panel, letter in zip(panels_spec, axes, "abcd"):
            for m in methods:
                mus, los, his = [], [], []
                for f in fracs:
                    v = fn(f"rs_f{f}_{m}")
                    if v:
                        mus.append(np.mean(v)); los.append(min(v)); his.append(max(v))
                    else:
                        mus.append(np.nan); los.append(np.nan); his.append(np.nan)
                if not all(np.isnan(mus)):
                    _line(panel, m, list(fracs), np.array(mus),
                          np.array(los), np.array(his), every=1)
            panel.set_xlabel("malicious fraction")
            panel.set_ylabel(ylab)
            panel.set_xticks(list(fracs))
            _panel(panel, letter, ylab.capitalize() + " vs. threat level")
        axes[0].set_ylim(0, 1.05); axes[1].set_ylim(0, 1.05)
        axes[3].set_ylim(0.9, 1.0)
        self._figlegend(fig, methods, y=1.04, ncol=5)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self._save(fig, "F5_sweep.pdf")

    # ---- Fig 6: resource effectiveness --------------------------------------
    def fig6(self, prefix="exp_h_adsys"):
        methods = ["trustfl", "sophon", "fedaa", "fltrust",
                   "fedgreed", "multikrum", "fedavg"]
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        (a, b), (c, dd) = ax
        # (a) cumulative useful updates
        for m in methods:
            curves = []
            for rid in self.d.runs(f"{prefix}_{m}"):
                rd = self.d.s.rounds(rid)
                dl = rd.apply(lambda r: len(_parse_listcol(r.get("delivered"))),
                              axis=1).to_numpy()
                import pandas as pd
                curves.append(pd.Series(np.cumsum(dl), index=rd["round"].values))
            band = Data._band(curves)
            if band:
                x, mu, lo, hi = band
                _line(a, m, x, mu, lo, hi, every=14)
        a.set_xlabel("round"); a.set_ylabel("cumulative useful updates")
        _panel(a, "a", "Useful work delivered over time")
        # (b) energy depletion
        for m in methods:
            band = self.d.col_band(f"{prefix}_{m}", "mean_energy_j")
            if band:
                x, mu, lo, hi = band
                _line(b, m, x, mu, lo, hi, every=14)
        b.set_xlabel("round"); b.set_ylabel("mean remaining energy (J)")
        _panel(b, "b", "Swarm energy depletion")
        # (c) energy per USEFUL update: (consumed energy) / (delivered updates)
        vals = {}
        for m in methods:
            per = []
            for rid in self.d.runs(f"{prefix}_{m}"):
                rd = self.d.s.rounds(rid)
                if "mean_energy_j" not in rd:
                    continue
                e = rd.dropna(subset=["mean_energy_j"])["mean_energy_j"]
                nuav = None
                try:
                    row = self.d.df[self.d.df["run_id"] == rid].iloc[0]
                    nuav = float(row.get("num_uavs", np.nan))
                except Exception:
                    pass
                total_deliv = int(rd.apply(
                    lambda r: len(_parse_listcol(r.get("delivered"))),
                    axis=1).sum())
                if len(e) >= 2 and total_deliv > 0 and nuav == nuav:
                    consumed = (float(e.iloc[0]) - float(e.iloc[-1])) * nuav
                    per.append(consumed / total_deliv)
            if per:
                vals[m] = per
        if vals:
            _bars(c, [m for m in methods if m in vals], vals, dots=vals,
                  ylabel="energy per useful update (J)", fmt="{:.0f}")
            for lbl in c.get_xticklabels():
                lbl.set_rotation(14); lbl.set_ha("right")
        _panel(c, "c", "Energy cost per useful update\n(lower is better)")
        # (d) cost pareto
        cost = self.d.df.groupby("method")["agg_time_ms"].mean().to_dict()
        for m in methods:
            rets = self.d.retention_vals(f"{prefix}_{m}")
            ms = cost.get(m, np.nan)
            if rets and ms == ms:
                ours = m == "trustfl"
                dd.scatter([ms], [np.mean(rets)], s=170 if ours else 60,
                           color=COLORS[m], marker=MARKERS[m],
                           edgecolors="black", linewidths=1.2 if ours else 0.6,
                           zorder=5 if ours else 3)
                dd.annotate(LABELS[m].replace(" (ours)", ""), (ms, np.mean(rets)),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=6, fontweight="bold" if ours else "normal")
        dd.set_xscale("log")
        dd.set_xlabel("aggregation time (ms, log)")
        dd.set_ylabel("retention under attack")
        _panel(dd, "d", "Robustness per unit compute")
        self._figlegend(fig, methods, y=1.04, ncol=7)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self._save(fig, "F6_resource.pdf")

    # ---- Fig 7: mechanism + ablation ----------------------------------------
    def fig7(self):
        fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0))
        (a, b), (c, dd) = ax
        # (a) trust separation
        for col, lab, col_c in [("mean_trust_honest", "honest UAVs", "#0072B2"),
                                ("mean_trust_malicious", "malicious UAVs", "#D55E00")]:
            band = self.d.col_band("exp_h_adsys_trustfl", col)
            if band:
                x, mu, lo, hi = band
                a.plot(x, mu, color=col_c, lw=1.8, label=lab)
                a.fill_between(x, lo, hi, color=col_c, alpha=0.15, lw=0)
        a.set_xlabel("round"); a.set_ylabel("mean trust")
        a.legend(frameon=False)
        _panel(a, "a", "Trust separates the populations\n(system attack)")
        # (b)/(c) weight dynamics under the two attack families
        names = ["behavioral", "computational", "communication"]
        cs = ["#009E73", "#CC79A7", "#56B4E9"]
        lss = ["-", "--", ":"]
        for panel, run, letter, ttl in [
                (b, "exp_h_adsys_trustfl", "b",
                 "Weights under CYCLING system attack:\ntracking the moving dimension"),
                (c, "fid_signflip_trustfl", "c",
                 "Weights under sign-flip: convergence\nto pure behavioral (Thm. 2 verified)")]:
            bands = self.d.weights_band(run)
            for d_i, band in enumerate(bands):
                if band:
                    x, mu, lo, hi = band
                    panel.plot(x, mu, color=cs[d_i], ls=lss[d_i], lw=1.6,
                               label=names[d_i])
                    panel.fill_between(x, lo, hi, color=cs[d_i], alpha=0.13, lw=0)
            panel.set_ylim(-0.02, 1.05)
            panel.set_xlabel("round"); panel.set_ylabel("dimension weight")
            panel.legend(frameon=False, fontsize=6)
            _panel(panel, letter, ttl)
        # (d) gate ablation
        vals = {}
        for name, lab in [("exp_c_adaptive_trustfl", "full"),
                          ("exp_c_ablate_symewma", "sym.\nEWMA"),
                          ("exp_c_ablate_nogate", "no\ngate")]:
            v = self.d.acc_vals(name)
            if v:
                vals[lab] = v
        if vals:
            for i, (lab, v) in enumerate(vals.items()):
                col = "#D55E00" if lab == "full" else "#888888"
                dd.bar(i, np.mean(v), width=0.6, color=col,
                       alpha=1.0 if lab == "full" else 0.7,
                       hatch=None if lab == "full" else "//",
                       edgecolor="black", linewidth=1.2 if lab == "full" else 0.5)
                dd.scatter([i] * len(v), v, s=14, color="black", alpha=0.6,
                           zorder=4)
            dd.set_xticks(range(len(vals)))
            dd.set_xticklabels(list(vals.keys()), fontsize=7)
            dd.set_ylabel("final accuracy")
            dd.set_ylim(0, 1.1)
        _panel(dd, "d", "Ablation: removing the consistency\ngate collapses robustness")
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
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("cmd", choices=["fig1", "fig2", "fig3", "fig4",
                                    "fig5", "fig6", "fig7", "all"])
    args = ap.parse_args()
    f = Figs(args.store)
    getattr(f, args.cmd if args.cmd != "all" else "all")()


if __name__ == "__main__":
    main()
