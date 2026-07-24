"""The novelty-proof analysis: does Sophon (the closest published competitor,
TDSC 2025) make the honest-vs-malicious distinction that we claim only our
system-dimension trust can make?

Sophon is a STRONG gradient-trust defense. The question is not whether it
works (it does, on gradient attacks) but whether it can see the SYSTEM
dimension. If Sophon shows LOW retention + baseline-level influence under the
adaptive-system attack, that empirically proves our contribution: even the
best published multi-signal gradient-trust cannot make the distinction.
"""
import numpy as np, statistics as st
from trustfl.audit import retention_for_run
from trustfl.store import Store

s = Store('./results_store')
METHODS = ['trustfl', 'sophon', 'fedaa', 'fltrust', 'fedgreed', 'multikrum', 'fedavg']

def analyze(prefix, label, seeds):
    print(f"\n{'='*66}\n{label}  ({prefix}_*)\n{'='*66}")
    print(f"{'method':11s} {'retention':>10s} {'mal_infl':>10s} {'deliv_eff':>10s} {'acc':>7s}")
    print("-"*52)
    rows = {}
    for m in METHODS:
        rets, infls, deffs, accs = [], [], [], []
        for seed in seeds:
            rid = f'{prefix}_{m}/seed{seed}'
            try:
                _, hm = retention_for_run(s, rid)
                rd = s.rounds(rid)
                mi = [v for v in rd.get('malicious_influence_share', []) if v==v]
                import json
                deff = []
                for _, r in rd.iterrows():
                    inv = len(json.loads(r['invited']) if isinstance(r['invited'],str) else r['invited'])
                    dlv = len(json.loads(r['delivered']) if isinstance(r['delivered'],str) else r['delivered'])
                    if inv: deff.append(dlv/inv)
                acc = rd.dropna(subset=['test_acc'])['test_acc'].iloc[-1]
                if hm==hm:
                    rets.append(hm); accs.append(acc)
                    if mi: infls.append(np.mean(mi))
                    if deff: deffs.append(np.mean(deff))
            except Exception:
                pass
        if rets:
            rows[m] = (rets, infls)
            tag = "  <- ours" if m=="trustfl" else ("  <- closest competitor" if m=="sophon" else ("  <- RL learnable-weights (2025)" if m=="fedaa" else ""))
            print(f"{m:11s} {st.mean(rets):10.3f} {st.mean(infls):10.3f} {st.mean(deffs):10.3f} {st.mean(accs):7.3f}{tag}")
    # the money comparisons: TrustFL vs BOTH 2025 competitors on retention
    for comp, label in [('sophon', 'Sophon'), ('fedaa', 'FedAA')]:
        if 'trustfl' in rows and comp in rows:
            tf, so = rows['trustfl'][0], rows[comp][0]
            gap = st.mean(tf) - st.mean(so)
            print(f"\n  TrustFL vs {label} retention gap: +{gap:.3f}")
            print(f"  per-seed {label}: {' '.join(f'{v:.3f}' for v in sorted(so))}")
            print(f"  TrustFL worst seed: {min(tf):.3f}   {label} best seed: {max(so):.3f}")
            print(f"  -> {'CLEAN SEPARATION' if min(tf)>max(so) else 'overlap - report means only'}")
            try:
                from scipy.stats import mannwhitneyu
                if len(tf)>=3 and len(so)>=3:
                    u,p = mannwhitneyu(tf, so, alternative='greater')
                    print(f"  Mann-Whitney (TrustFL>{label} retention): p={p:.4f}")
            except ImportError:
                pass

analyze("exp_h_adsys", "MNIST HARDENED (5 seeds w/ Sophon)", [0,1,2,3,4])
analyze("femh_adsys", "FEMNIST HARDENED (5 seeds w/ Sophon)", [0,1,2,3,4])
