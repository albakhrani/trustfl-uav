"""Publication figures from the results store (Phase 4).

Signature figure: the retention-vs-influence plane. Every method is a point;
the y-axis is honest-data retention (higher = keeps honest UAVs), the x-axis
is malicious influence share (lower = benches attackers). The paper's thesis
is that TrustFL alone occupies the top region (keeps honest data) while every
baseline sits low -- and crucially, they do NOT buy lower influence with that
sacrifice. The figure makes that one truth unmissable.

Design principle (publication, not marketing): structure encodes the claim.
Colorblind-safe palette, print-legible, no chartjunk. The only emphasis is on
the one true thing -- TrustFL's retention lead.

    python -m trustfl.figures --store ./results_store dual --attack adaptive_system
    python -m trustfl.figures --store ./results_store cost
    python -m trustfl.figures --store ./results_store all
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# headless backend for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .store import Store
from .audit import retention_for_run

# colorblind-safe (Okabe-Ito); ours gets the one saturated highlight
COLORS = {
    "trustfl":   "#D55E00",   # vermillion — ours, the highlight
    "sophon":    "#8B4789",   # dark magenta — closest competitor
    "fedaa":     "#B8860B",   # dark goldenrod — RL learnable-weights line
    "fedavg":    "#999999",   # grey
    "fltrust":   "#0072B2",   # blue
    "fedgreed":  "#009E73",   # green
    "krum":      "#CC79A7",   # purple
    "multikrum": "#56B4E9",   # light blue
    "median":    "#E69F00",   # orange
    "trimmed_mean": "#F0E442",
}
LABELS = {
    "trustfl": "TrustFL (ours)", "sophon": "Sophon", "fedaa": "FedAA", "fedavg": "FedAvg",
    "fltrust": "FLTrust", "fedgreed": "FedGreed", "krum": "Krum",
    "multikrum": "Multi-Krum", "median": "Median", "trimmed_mean": "Trimmed-Mean",
}

# --- publication style system (Rougier et al. 2014; Okabe-Ito CUD) ---------
# One message per figure; ours gets the single saturated highlight; baselines
# are muted. Distinct MARKER SHAPES so every figure survives grayscale print
# ("get it right in black and white"). Sizes target IEEE print widths so
# fonts land at legible size without rescaling.
IEEE_COL = 3.5        # single-column width (in)
IEEE_FULL = 7.16      # double-column width (in)
MARKERS = {
    "trustfl": "*", "sophon": "D", "fedaa": "P", "fltrust": "s",
    "fedgreed": "^", "multikrum": "v", "fedavg": "o", "krum": "X",
    "median": "d", "trimmed_mean": "h",
}
BASE_ALPHA = 0.75     # baselines muted; ours at 1.0

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300,   # 300 DPI print masters
    "lines.linewidth": 1.4, "grid.alpha": 0.25,
})


def _style_for(method):
    """(color, marker, size, alpha, lw, zorder) — ours highlighted, rest muted."""
    ours = method == "trustfl"
    return dict(color=COLORS.get(method, "#333333"),
                marker=MARKERS.get(method, "o"),
                ms=10 if ours else 5.5,
                alpha=1.0 if ours else BASE_ALPHA,
                lw=2.3 if ours else 1.2,
                zorder=5 if ours else 2)


def _canon(store, where=""):
    try:
        base = "paper_tag IS NOT NULL AND paper_tag != 'exploratory'"
        w = f"({where}) AND {base}" if where else base
        return store.query(w)
    except Exception:
        return store.query(where)


class Figures:
    def __init__(self, store_root, out="./paper_assets/figures"):
        self.s = Store(store_root)
        self.out = out
        os.makedirs(out, exist_ok=True)

    # ---- signature: retention vs influence ----
    def dual(self, attack="adaptive_system", name_prefix="exp_h_adsys"):
        """One point per method: (mean malicious influence, mean hm retention)
        with per-seed scatter to show spread. TrustFL highlighted."""
        df = _canon(self.s, f"attack_type='{attack}'")
        df = df[df["name"].str.startswith(name_prefix)] if not df.empty else df
        if df.empty:
            print(f"no runs for {attack} / {name_prefix}"); return

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        methods = sorted(df["method"].unique(),
                         key=lambda m: 0 if m == "trustfl" else 1)

        # gather means first so we can place labels without collisions
        pts = {}  # method -> (mean_infl, mean_ret, seed_infls, seed_rets)
        for method in methods:
            runs = df[df["method"] == method]["run_id"].tolist()
            rets, infls = [], []
            for rid in runs:
                _, hm = retention_for_run(self.s, rid)
                rd = self.s.rounds(rid)
                mi = [v for v in rd.get("malicious_influence_share", []) if v == v]
                if hm == hm and mi:
                    rets.append(hm); infls.append(float(np.mean(mi)))
            if rets:
                pts[method] = (float(np.mean(infls)), float(np.mean(rets)), infls, rets)

        mal_frac = df["malicious_frac"].dropna().iloc[0] if "malicious_frac" in df else 0.4

        # tight data-driven axis limits (retention never below data, with pad)
        all_ret = [r for v in pts.values() for r in v[3]]
        all_inf = [i for v in pts.values() for i in v[2]]
        ymin = max(0.0, min(all_ret) - 0.06)
        ymax = min(1.02, max(all_ret) + 0.06)
        xmin = max(0.0, min(all_inf) - 0.03)
        xmax = max(max(all_inf) + 0.03, mal_frac + 0.02)

        # winning-corner cue (low influence, high retention = top-left)
        ax.axhspan(0.75, ymax, color="#D55E00", alpha=0.06, zorder=0)
        ax.text(0.015, 0.985, "keeps honest data", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#B0530A", style="italic")

        # draw each method: seed dots tethered to the bold mean
        for method, (mi, mr, sinf, sret) in pts.items():
            c = COLORS.get(method, "#333333")
            is_ours = method == "trustfl"
            for si, sr in zip(sinf, sret):          # tether lines
                ax.plot([mi, si], [mr, sr], color=c, alpha=0.25, lw=0.8, zorder=1)
            ax.scatter(sinf, sret, s=32, color=c, alpha=0.4,
                       edgecolors="none", zorder=2)
            ax.scatter([mi], [mr], s=340 if is_ours else 180, color=c,
                       edgecolors="black", linewidths=1.8 if is_ours else 0.8,
                       marker="*" if is_ours else "o", zorder=5,
                       label=LABELS.get(method, method))

        # collision-aware labels: offset direction chosen per point
        # (ours: up-right and bold; others: away from the crowd centre)
        cx = np.mean([v[0] for v in pts.values()])
        cy = np.mean([v[1] for v in pts.values()])
        for method, (mi, mr, _, _) in pts.items():
            is_ours = method == "trustfl"
            if is_ours:
                dx, dy, ha = 10, 10, "left"
            else:
                dx = 9 if mi >= cx else -9
                dy = 9 if mr >= cy else -11
                ha = "left" if dx > 0 else "right"
            ax.annotate(LABELS.get(method, method), (mi, mr),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=9.5 if is_ours else 8.5,
                        fontweight="bold" if is_ours else "normal",
                        color=COLORS.get(method) if is_ours else "#333333",
                        ha=ha, zorder=6,
                        path_effects=None)

        # population-share reference (attackers' fair share) — label at top
        ax.axvline(mal_frac, ls="--", color="#999999", lw=1, zorder=1)
        ax.text(mal_frac - 0.004, ymax - 0.02, f"attacker pop. share ({mal_frac:.0%})",
                fontsize=7.5, color="#999999", rotation=90, va="top", ha="right")

        ax.set_xlabel("Malicious influence share  ($\\leftarrow$ better: benches attackers)",
                      fontsize=11)
        ax.set_ylabel("Honest high-mobility retention  (better $\\rightarrow$: keeps UAVs)",
                      fontsize=11)
        ds = df["dataset"].dropna().iloc[0].upper() if "dataset" in df and not df["dataset"].dropna().empty else ""
        ax.set_title(f"{ds}: honest-data retention vs. attacker influence\n"
                     f"adaptive system-dimension adversary ({mal_frac:.0%} malicious)",
                     fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(self.out, f"dual_{name_prefix}.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- six-facet effectiveness heatmap ----
    def heatmap(self, prefix="exp_h_adsys"):
        """Methods x facets, each facet normalized so brighter = better.
        TrustFL's row should glow; each baseline shows its weak (dark) cell."""
        from .metrics_matrix import matrix, matrix_per_seed, column_is_significant
        df = matrix(self.s, prefix)
        if df.empty:
            print(f"no runs for {prefix}"); return
        per_seed = matrix_per_seed(self.s, prefix)
        facets = ["final_acc", "rounds_to_target", "hm_retention", "mal_infl",
                  "deliv_eff", "energy_per_update", "agg_ms"]
        fac_labels = ["Accuracy", "Converg.\nspeed", "Honest\nretention",
                      "Security\n(low infl)", "Delivery\nefficiency",
                      "Energy\nefficiency", "Low\ncost"]
        higher_better = {"final_acc", "hm_retention", "deliv_eff"}
        log_scale = {"agg_ms"}
        # Accuracy on these tasks is near-saturated and is NOT where methods
        # differ -- that is the paper's thesis. We therefore show it as a
        # neutral annotated column ("comparable"), never coloured, so the
        # figure does not over-dramatise a 0.01 gap. Colour is reserved for the
        # facets that actually differentiate the methods.
        never_colour = {"final_acc"}
        M = df[facets].to_numpy(float)
        norm = np.zeros_like(M)
        tied_cols = set()
        for j, f in enumerate(facets):
            if f in never_colour or not column_is_significant(per_seed, f):
                norm[:, j] = 0.5
                tied_cols.add(j)
                continue
            col = M[:, j].astype(float)
            if f in log_scale:
                col = np.log10(np.clip(col, 1e-6, None))
            lo, hi = col.min(), col.max()
            rng = hi - lo
            if rng < 1e-12:
                norm[:, j] = 0.5
                tied_cols.add(j)
            elif f in higher_better:
                norm[:, j] = (col - lo) / rng
            else:
                norm[:, j] = (hi - col) / rng
        methods = df["method"].tolist()

        fig, ax = plt.subplots(figsize=(7.6, 0.7 * len(methods) + 1.6))
        im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(facets)))
        # accuracy is shown as "comparable" (neutral, not scored); other tied
        # columns get "(n.s.)". This states plainly that accuracy is not a
        # differentiating facet -- the paper's thesis.
        xt = []
        for j, lbl in enumerate(fac_labels):
            if facets[j] == "final_acc":
                xt.append(lbl + "\n(comparable)")
            elif j in tied_cols:
                xt.append(lbl + "\n(n.s.)")
            else:
                xt.append(lbl)
        ax.set_xticklabels(xt, fontsize=9)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([f"{m} (ours)" if m == "TrustFL" else m for m in methods],
                           fontsize=10,
                           fontweight="bold")
        # annotate each cell with the RAW value; contrast-aware text colour
        for i in range(len(methods)):
            for j, f in enumerate(facets):
                raw = df.iloc[i][f]
                txt = (f"{raw:.0f}" if f in ("agg_ms", "energy_per_update", "rounds_to_target")
                       else f"{raw:.2f}" if f != "final_acc" else f"{raw:.3f}")
                v = norm[i, j]
                # RdYlGn is dark at both ends, light in the middle
                tc = "white" if (v < 0.22 or v > 0.88) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=tc)
        # highlight TrustFL row
        if "TrustFL" in methods:
            ti = methods.index("TrustFL")
            ax.add_patch(plt.Rectangle((-0.5, ti - 0.5), len(facets), 1,
                         fill=False, edgecolor="#0033AA", lw=2.5, zorder=5))
        ds = ""
        try:
            d0 = self.s.query()
            d0 = d0[d0["name"].str.startswith(prefix)]
            ds = d0["dataset"].dropna().iloc[0].upper()
        except Exception:
            pass
        ax.set_title(f"{ds}: system effectiveness across facets\n"
                     f"colour shown only where the between-method gap exceeds "
                     f"2$\\times$ seed noise",
                     fontsize=10.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("normalized (per facet)", fontsize=8)
        cbar.set_ticks([0, 1]); cbar.set_ticklabels(["worst", "best"])
        fig.tight_layout()
        p = os.path.join(self.out, f"heatmap_{prefix}.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- robustness curves: do the SUPPORTED advantages scale with threat? ----
    def robustness(self, prefix="rs", fractions=(0.1, 0.2, 0.3, 0.4)):
        """One panel per facet; x = malicious fraction; one line per method.

        Only the facets that survived the variance check are plotted as claims
        (retention, delivery efficiency, energy per useful update). Accuracy and
        malicious-influence are plotted too, but labelled as context, because
        the snapshot analysis found them comparable across methods.
        """
        from .metrics_matrix import facets_for_run
        panels = [("hm_retention", "Honest retention", True),
                  ("deliv_eff", "Delivery efficiency", True),
                  ("energy_per_update", "Energy per useful update (J)", False),
                  ("mal_infl", "Malicious influence (context)", False)]
        all_runs = self.s.query()
        methods = ["trustfl", "fltrust", "fedgreed", "multikrum", "fedavg"]

        fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.5))
        for ax, (facet, title, higher_better) in zip(axes, panels):
            for m in methods:
                xs, ys, es = [], [], []
                for f in fractions:
                    name = f"rs_f{f}_{m}"
                    runs = all_runs[all_runs["name"] == name]["run_id"].tolist()
                    if not runs:
                        continue
                    vals = []
                    for rid in runs:
                        v = facets_for_run(self.s, rid).get(facet)
                        if v == v:
                            vals.append(v)
                    if vals:
                        xs.append(f); ys.append(np.mean(vals))
                        es.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                if not xs:
                    continue
                c = COLORS.get(m, "#333")
                is_ours = m == "trustfl"
                ax.errorbar(xs, ys, yerr=es, color=c, marker="*" if is_ours else "o",
                            markersize=11 if is_ours else 5,
                            lw=2.4 if is_ours else 1.3,
                            capsize=2.5, alpha=1.0 if is_ours else 0.75,
                            label=LABELS.get(m, m), zorder=5 if is_ours else 2)
            ax.set_xlabel("malicious fraction", fontsize=10)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(list(fractions))
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("value", fontsize=10)
        axes[0].legend(fontsize=8, frameon=False, loc="best")
        fig.suptitle("Does the advantage scale with threat severity? "
                     "(MNIST, adaptive system-dimension adversary)", fontsize=11)
        fig.tight_layout()
        p = os.path.join(self.out, "robustness_curves.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- sustained effectiveness: does the system keep working as batteries deplete? ----
    # ---- sustained effectiveness: does the system keep working as batteries deplete? ----
    def sustained(self, prefix="exp_h_adsys", seeds=(0, 1, 2, 3, 4)):
        """Two panels over rounds: (a) cumulative useful (delivered) updates,
        (b) mean remaining UAV energy. Shows whether TrustFL sustains useful
        participation as the swarm's batteries deplete, rather than just a
        single end-of-run number."""
        import json
        methods = ["trustfl", "sophon", "fltrust", "fedgreed", "multikrum", "fedavg"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
        for m in methods:
            cum_curves, energy_curves = [], []
            for seed in seeds:
                rid = f"{prefix}_{m}/seed{seed}"
                try:
                    rd = self.s.rounds(rid)
                except Exception:
                    continue
                if rd.empty:
                    continue
                delivered = rd.apply(
                    lambda r: len(json.loads(r["delivered"]) if isinstance(r["delivered"], str)
                                  else r["delivered"]), axis=1).to_numpy()
                cum_curves.append(np.cumsum(delivered))
                if "mean_energy_j" in rd:
                    e = rd["mean_energy_j"].to_numpy(dtype=float)
                    energy_curves.append(e)
            if not cum_curves:
                continue
            L = min(len(c) for c in cum_curves)
            cum = np.mean([c[:L] for c in cum_curves], axis=0)
            c = COLORS.get(m, "#333")
            ours = m == "trustfl"
            rounds = np.arange(L)
            ax1.plot(rounds, cum, color=c, lw=2.6 if ours else 1.4,
                     alpha=1.0 if ours else 0.7, label=LABELS.get(m, m),
                     zorder=5 if ours else 2)
            if energy_curves:
                Le = min(len(e) for e in energy_curves)
                en = np.mean([e[:Le] for e in energy_curves], axis=0)
                ax2.plot(np.arange(Le), en, color=c, lw=2.6 if ours else 1.4,
                         alpha=1.0 if ours else 0.7, label=LABELS.get(m, m),
                         zorder=5 if ours else 2)
        ax1.set_xlabel("round", fontsize=10)
        ax1.set_ylabel("cumulative useful updates", fontsize=10)
        ax1.set_title("Useful work delivered over time", fontsize=10)
        ax1.legend(fontsize=8, frameon=False, loc="upper left")
        ax1.grid(True, alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)
        ax2.set_xlabel("round", fontsize=10)
        ax2.set_ylabel("mean remaining energy (J)", fontsize=10)
        ax2.set_title("Swarm energy depletion", fontsize=10)
        ax2.grid(True, alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Sustained effectiveness as UAV batteries deplete "
                     "(MNIST, adaptive system adversary)", fontsize=11)
        fig.tight_layout()
        p = os.path.join(self.out, f"sustained_{prefix}.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- Fig 1: the two failures that motivate the paper ----
    def twofail(self):
        """(a) FedAvg admits everything: a gradient attack collapses accuracy.
        (b) Krum, the geometric archetype, discards ~90% of honest
        high-mobility UAVs WITH NO ATTACK PRESENT (its home turf).
        TrustFL shown on both panels. One figure, two failure modes, sets up
        the reliable-vs-malicious distinction.
        SRC: mnist_attack_fedavg / mnist_trustfl_debug (grad_scale);
             exp_a_krum / exp_a_trustfl (no attack, retention)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_FULL, 2.6))

        # (a) accuracy trajectories under grad_scale
        for name, method in [("mnist_attack_fedavg", "fedavg"),
                             ("mnist_trustfl_debug", "trustfl")]:
            df = self.s.query()
            runs = df[df["name"] == name]["run_id"].tolist()
            curves = []
            for rid in runs:
                rd = self.s.rounds(rid).dropna(subset=["test_acc"])
                if len(rd):
                    curves.append(rd.set_index("round")["test_acc"])
            if not curves:
                continue
            L = min(len(c) for c in curves)
            arr = np.stack([c.values[:L] for c in curves])
            st = _style_for(method)
            x = curves[0].index.values[:L]
            ax1.plot(x, arr.mean(0), color=st["color"], lw=st["lw"],
                     alpha=st["alpha"], zorder=st["zorder"],
                     label=LABELS.get(method, method))
            ax1.fill_between(x, arr.min(0), arr.max(0), color=st["color"],
                             alpha=0.15, lw=0)
        ax1.set_xlabel("round"); ax1.set_ylabel("test accuracy")
        ax1.set_title("(a) No defense: gradient attack\ncollapses FedAvg")
        ax1.set_ylim(0, 1.02); ax1.grid(True)
        ax1.legend(frameon=False, loc="center right")

        # (b) honest high-mobility retention, NO attack
        vals = {}
        for name, method in [("exp_a_krum", "krum"),
                             ("exp_a_multikrum", "multikrum"),
                             ("exp_a_fltrust", "fltrust"),
                             ("exp_a_trustfl", "trustfl")]:
            df = self.s.query()
            runs = df[df["name"] == name]["run_id"].tolist()
            rets = []
            for rid in runs:
                _, hm = retention_for_run(self.s, rid)
                if hm == hm:
                    rets.append(hm)
            if rets:
                vals[method] = rets
        methods = [m for m in ["krum", "multikrum", "fltrust", "trustfl"] if m in vals]
        for i, m in enumerate(methods):
            st = _style_for(m)
            mean = np.mean(vals[m])
            ax2.bar(i, mean, width=0.62, color=st["color"], alpha=st["alpha"],
                    edgecolor="black" if m == "trustfl" else "none",
                    linewidth=1.4 if m == "trustfl" else 0)
            ax2.scatter([i] * len(vals[m]), vals[m], s=12, color="black",
                        alpha=0.55, zorder=4)
            ax2.text(i, mean + 0.03, f"{mean:.2f}", ha="center", fontsize=8,
                     fontweight="bold" if m == "trustfl" else "normal")
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([LABELS.get(m, m).replace(" (ours)", "\n(ours)")
                             for m in methods])
        ax2.set_ylabel("honest high-mobility retention")
        ax2.set_title("(b) Defense on, no attack: geometric\nfilters discard honest mobile UAVs")
        ax2.set_ylim(0, 1.1); ax2.grid(True, axis="y")

        fig.tight_layout()
        p = os.path.join(self.out, "twofail_motivation.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- Fig 4: paradigm fidelity-vs-blindness (the novelty visual) ----
    def paradigm(self, grad_prefix="fid_signflip", sys_prefix="exp_h_adsys",
                 methods=("sophon", "fedaa", "fltrust", "trustfl")):
        """For each method, two bars: malicious influence under a GRADIENT
        attack (the attacks these defenses were built for -- near 0 proves our
        reimplementations are faithful and strong) vs under the SYSTEM attack
        (blind: influence stays near population share). TrustFL low on both.
        This is the anti-straw-man protocol made visual.
        Requires the fidelity runs: {grad_prefix}_<method> (sign_flip, 30%).
        SRC: fid_signflip_* + exp_h_adsys_* / femh_adsys_*."""
        df = self.s.query()

        def infl_for(prefix, m):
            runs = df[df["name"] == f"{prefix}_{m}"]["run_id"].tolist()
            out = []
            for rid in runs:
                rd = self.s.rounds(rid)
                mi = [v for v in rd.get("malicious_influence_share", []) if v == v]
                if mi:
                    out.append(float(np.mean(mi[len(mi) // 2:])))  # late half
            return out

        have = [(m, infl_for(grad_prefix, m), infl_for(sys_prefix, m))
                for m in methods]
        have = [(m, g, s) for m, g, s in have if g and s]
        if not have:
            print(f"  paradigm: no fidelity runs found under {grad_prefix}_* — "
                  f"run the fidelity commands first"); return

        fig, ax = plt.subplots(figsize=(IEEE_COL, 2.7))
        x = np.arange(len(have))
        w = 0.36
        for i, (m, g, s) in enumerate(have):
            st = _style_for(m)
            ax.bar(i - w / 2, np.mean(g), width=w, color=st["color"],
                   alpha=0.45, edgecolor="none",
                   label="gradient attack" if i == 0 else None)
            ax.bar(i + w / 2, np.mean(s), width=w, color=st["color"],
                   alpha=1.0 if m == "trustfl" else BASE_ALPHA,
                   edgecolor="black" if m == "trustfl" else "none",
                   linewidth=1.2,
                   label="system attack" if i == 0 else None)
            for dx, vals in [(-w / 2, g), (w / 2, s)]:
                ax.scatter([i + dx] * len(vals), vals, s=9, color="black",
                           alpha=0.55, zorder=4)
        ax.axhline(0.3, ls=":", color="#888", lw=1)
        ax.text(len(have) - 0.45, 0.305, "grad-attack pop. share (30%)",
                fontsize=6.5, color="#888", ha="right", va="bottom")
        ax.axhline(0.4, ls="--", color="#888", lw=1)
        ax.text(len(have) - 0.45, 0.405, "system-attack pop. share (40%)",
                fontsize=6.5, color="#888", ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(m, m).replace(" (ours)", "\n(ours)")
                            for m, _, _ in have])
        ax.set_ylabel("malicious influence share")
        ax.set_title("Faithful on their turf, blind on ours:\n"
                     "influence under gradient vs. system attacks")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(True, axis="y")
        fig.tight_layout()
        p = os.path.join(self.out, "paradigm_fidelity.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- Fig: mechanism — trust separation + learnable-weight dynamics ----
    def mechanism(self, prefix="exp_h_adsys", seeds=(0, 1, 2, 3, 4)):
        """(a) mean trust of honest vs malicious UAVs over rounds (TrustFL,
        adaptive-system adversary): the separation IS the mechanism.
        (b) learnable dimension weights over rounds: mass shifts toward the
        currently-discriminative dimension as the attacker cycles strategies
        (ties to the EG-regret Remark in the theory).
        SRC: exp_h_adsys_trustfl rounds: mean_trust_honest/_malicious,
             trust_weights."""
        import json as _json
        rows = []
        for seed in seeds:
            try:
                rd = self.s.rounds(f"{prefix}_trustfl/seed{seed}")
                if not rd.empty:
                    rows.append(rd)
            except Exception:
                pass
        if not rows:
            print("  mechanism: no trustfl rounds found"); return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_FULL, 2.6))

        # (a) honest vs malicious mean trust, seed-averaged band
        for col, lab, c in [("mean_trust_honest", "honest UAVs", "#0072B2"),
                            ("mean_trust_malicious", "malicious UAVs", "#D55E00")]:
            curves = [r.dropna(subset=[col]).set_index("round")[col]
                      for r in rows if col in r]
            curves = [c_ for c_ in curves if len(c_)]
            if not curves:
                continue
            L = min(len(c_) for c_ in curves)
            arr = np.stack([c_.values[:L] for c_ in curves])
            x = curves[0].index.values[:L]
            ax1.plot(x, arr.mean(0), color=c, lw=1.8, label=lab)
            ax1.fill_between(x, arr.min(0), arr.max(0), color=c, alpha=0.15, lw=0)
        ax1.set_xlabel("round"); ax1.set_ylabel("mean trust score")
        ax1.set_title("(a) Trust separates the populations")
        ax1.legend(frameon=False); ax1.grid(True)

        # (b) learnable dimension weights over rounds (seed 0 as exemplar)
        rd0 = rows[0]
        if "trust_weights" in rd0:
            tw = rd0.dropna(subset=["trust_weights"])
            if len(tw):
                W = np.array([(_json.loads(v) if isinstance(v, str) else v)
                              for v in tw["trust_weights"]])
                x = tw["round"].values
                names = ["behavioral", "computational", "communication"]
                cs = ["#009E73", "#CC79A7", "#56B4E9"]
                for d in range(min(3, W.shape[1])):
                    ax2.plot(x, W[:, d], color=cs[d], lw=1.6, label=names[d])
                ax2.set_ylim(0, 1)
        ax2.set_xlabel("round"); ax2.set_ylabel("dimension weight")
        ax2.set_title("(b) Learnable weights track the\ndiscriminative dimension (seed 0)")
        ax2.legend(frameon=False); ax2.grid(True)

        fig.tight_layout()
        p = os.path.join(self.out, f"mechanism_{prefix}.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- cost vs robustness Pareto ----
    def cost(self, prefix="exp_h_adsys"):
        """x = per-round aggregation cost (log ms, all canonical runs);
        y = honest-data retention under the hardened adaptive-system adversary
        ({prefix}_*). One point per method: what robustness does the compute
        buy? TrustFL: near-FedAvg cost, highest retention.
        SRC: agg_time_ms (all runs) + retention_for_run on {prefix}_*."""
        df = _canon(self.s)
        if df.empty:
            print("store empty"); return
        cost_ms = df.groupby("method")["agg_time_ms"].mean().to_dict()
        full = self.s.query()
        pts = {}
        for m, ms in cost_ms.items():
            runs = full[full["name"] == f"{prefix}_{m}"]["run_id"].tolist()
            rets = []
            for rid in runs:
                _, hm = retention_for_run(self.s, rid)
                if hm == hm:
                    rets.append(hm)
            if rets and ms == ms:
                pts[m] = (float(ms), float(np.mean(rets)), rets)
        if not pts:
            print("  cost: no hardened retention available"); return
        fig, ax = plt.subplots(figsize=(IEEE_COL, 2.9))
        for m, (ms, ret, rets) in pts.items():
            st = _style_for(m)
            ax.scatter([ms] * len(rets), rets, s=10, color=st["color"],
                       alpha=0.3, edgecolors="none", zorder=2)
            ax.scatter([ms], [ret], s=170 if m == "trustfl" else 70,
                       color=st["color"], marker=st["marker"],
                       alpha=st["alpha"], edgecolors="black",
                       linewidths=1.2 if m == "trustfl" else 0.6,
                       zorder=st["zorder"])
            ax.annotate(LABELS.get(m, m), (ms, ret),
                        textcoords="offset points", xytext=(6, 5),
                        fontsize=7,
                        fontweight="bold" if m == "trustfl" else "normal")
        ax.set_xscale("log")
        ax.set_xlabel("per-round aggregation time (ms, log)")
        ax.set_ylabel("honest retention under attack")
        ax.set_title("Robustness per unit compute")
        ax.grid(True, which="both")
        fig.tight_layout()
        p = os.path.join(self.out, "cost_pareto.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    # ---- convergence bands (Claim 1: TrustFL stable, Krum unstable) ----
    def convergence(self, dataset="mnist", attack="none", prefix="exp_a"):
        """Mean +/- std accuracy trajectory per method. Makes Krum's
        instability visible (wide band, low mean) vs TrustFL (tight, high)."""
        df = _canon(self.s, f"dataset='{dataset}' AND attack_type='{attack}'")
        df = df[df["name"].str.startswith(prefix)] if not df.empty else df
        if df.empty:
            print(f"no {dataset}/{attack}/{prefix} runs"); return
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for method in sorted(df["method"].unique(),
                             key=lambda m: 0 if m == "trustfl" else 1):
            runs = df[df["method"] == method]["run_id"].tolist()
            # collect per-round test_acc across seeds
            byround = {}
            for rid in runs:
                rd = self.s.rounds(rid)
                if "test_acc" not in rd.columns:
                    continue
                sub = rd.dropna(subset=["test_acc"])
                for _, r in sub.iterrows():
                    byround.setdefault(int(r["round"]), []).append(float(r["test_acc"]))
            if not byround:
                continue
            rounds = sorted(byround)
            mean = np.array([np.mean(byround[r]) for r in rounds])
            std = np.array([np.std(byround[r]) for r in rounds])
            c = COLORS.get(method, "#333")
            is_ours = method == "trustfl"
            ax.plot(rounds, mean, color=c, lw=2.2 if is_ours else 1.5,
                    zorder=4 if is_ours else 3, label=LABELS.get(method, method))
            ax.fill_between(rounds, mean - std, mean + std, color=c,
                            alpha=0.18, zorder=2)
        ax.set_xlabel("Communication round", fontsize=11)
        ax.set_ylabel("Test accuracy", fontsize=11)
        ax.set_title(f"Convergence on {dataset.upper()} "
                     f"({'high-mobility, no attack' if attack=='none' else attack})\n"
                     f"band = ±1 std over seeds (wide = unstable)", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(self.out, f"convergence_{dataset}_{prefix}.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p} (+ .png)")

    def all(self):
        self.twofail()
        self.dual("adaptive_system", "exp_h_adsys")
        self.dual("adaptive_system", "femh_adsys")
        self.heatmap("exp_h_adsys")
        self.heatmap("femh_adsys")
        self.paradigm()
        self.mechanism("exp_h_adsys")
        self.robustness()
        self.sustained("exp_h_adsys")
        self.cost()
        self.convergence("mnist", "none", "exp_a")
        self.convergence("femnist", "none", "fem_a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="./results_store")
    ap.add_argument("--attack", default="adaptive_system")
    ap.add_argument("--prefix", default="exp_h_adsys")
    ap.add_argument("cmd", choices=["dual", "heatmap", "robustness", "sustained",
                                    "cost", "convergence", "twofail", "paradigm",
                                    "mechanism", "all"])
    ap.add_argument("--dataset", default="mnist")
    args = ap.parse_args()
    f = Figures(args.store)
    if args.cmd == "dual":
        f.dual(args.attack, args.prefix)
    elif args.cmd == "heatmap":
        f.heatmap(args.prefix)
    elif args.cmd == "robustness":
        f.robustness()
    elif args.cmd == "sustained":
        f.sustained(args.prefix)
    elif args.cmd == "cost":
        f.cost()
    elif args.cmd == "twofail":
        f.twofail()
    elif args.cmd == "paradigm":
        f.paradigm()
    elif args.cmd == "mechanism":
        f.mechanism(args.prefix)
    elif args.cmd == "convergence":
        pfx = "fem_a" if args.dataset == "femnist" else "exp_a"
        f.convergence(args.dataset, "none", pfx)
    else:
        f.all()


if __name__ == "__main__":
    main()
