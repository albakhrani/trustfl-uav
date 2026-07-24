"""Dual-metric check (retention vs malicious influence), any experiment prefix.

    python check_dual_v2.py                    # MNIST hardened (exp_h_adsys)
    python check_dual_v2.py femh_adsys         # FEMNIST hardened
    python check_dual_v2.py femh_adsys 60      # explicit round count (unused, informational)

Prints per-method means AND per-seed values, plus the Mann-Whitney p-value of
the retention gap vs each baseline (the reproducibility checklist's required
significance test).
"""
import sys
import statistics as st
from trustfl.audit import retention_for_run
from trustfl.store import Store

try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "exp_h_adsys"
SEEDS = [0, 1, 2]
METHODS = ["trustfl", "multikrum", "fltrust", "fedgreed", "fedavg"]

s = Store("./results_store")


def stats_for(method):
    rets, infls, accs = [], [], []
    for seed in SEEDS:
        rid = f"{PREFIX}_{method}/seed{seed}"
        try:
            _, hm = retention_for_run(s, rid)
            rd = s.rounds(rid)
            mi = [v for v in rd.get("malicious_influence_share", []) if v == v]
            acc = rd.dropna(subset=["test_acc"])["test_acc"].iloc[-1] if "test_acc" in rd else float("nan")
            if hm == hm:
                rets.append(hm)
            if mi:
                infls.append(sum(mi) / len(mi))
            accs.append(float(acc))
        except Exception:
            pass
    return rets, infls, accs


print(f"=== DUAL METRIC: {PREFIX} ===\n")
print(f"{'method':11s} {'retention':>10s} {'mal_infl':>10s} {'final_acc':>10s}   per-seed retention")
print("-" * 78)
data = {}
for m in METHODS:
    rets, infls, accs = stats_for(m)
    if not rets:
        print(f"  {m:9s} (no runs)")
        continue
    data[m] = rets
    per = " ".join(f"{r:.2f}" for r in rets)
    print(f"{m:11s} {st.mean(rets):10.3f} "
          f"{(st.mean(infls) if infls else float('nan')):10.3f} "
          f"{(st.mean(accs) if accs else float('nan')):10.3f}   [{per}]")

if "trustfl" in data and HAVE_SCIPY:
    print("\nRetention gap significance (one-sided Mann-Whitney, TrustFL > baseline):")
    tf = data["trustfl"]
    for m in METHODS[1:]:
        if m in data and len(data[m]) == len(tf) == 3:
            gap = st.mean(tf) - st.mean(data[m])
            _, p = mannwhitneyu(tf, data[m], alternative="greater")
            sep = "PERFECT separation" if min(tf) > max(data[m]) else "overlap"
            print(f"  vs {m:10s} gap=+{gap:.3f}  p={p:.3f}  ({sep})")

print("\nHEADLINE TEST: TrustFL retention should exceed every baseline in EVERY seed")
print("(perfect separation). Influence should be COMPARABLE across methods —")
print("baselines gain nothing from discarding honest data.")