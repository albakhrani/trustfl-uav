"""Fidelity-run analysis: per-method late-half malicious influence under
sign_flip (30%), plus TrustFL's per-dimension diagnosis (trust separation
and learnable-weight trajectory) to ground the complementarity framing."""
import json, statistics as st
import numpy as np
from trustfl.store import Store

s = Store('./results_store')
df = s.query()
print(f"{'method':10s} {'late_infl':>10s} {'acc':>8s}   per-seed influence")
print("-" * 60)
for m in ["sophon", "fedaa", "fltrust", "trustfl"]:
    runs = df[df["name"] == f"fid_signflip_{m}"]["run_id"].tolist()
    infls, accs = [], []
    for rid in sorted(runs):
        rd = s.rounds(rid)
        mi = [v for v in rd.get("malicious_influence_share", []) if v == v]
        if mi:
            infls.append(float(np.mean(mi[len(mi)//2:])))
        acc = rd.dropna(subset=["test_acc"])["test_acc"].iloc[-1]
        accs.append(float(acc))
    if infls:
        tag = "  <- ours" if m == "trustfl" else ""
        print(f"{m:10s} {st.mean(infls):10.3f} {st.mean(accs):8.3f}   "
              f"{' '.join(f'{v:.3f}' for v in infls)}{tag}")
print()
print("Interpretation guide (pop share = 0.30):")
print("  near 0.00 -> method suppresses sign-flippers (root-anchored defenses)")
print("  ~0.15-0.30 -> flippers retain weight (median-anchored consistency)")
print()
# TrustFL diagnosis: did the learnable weights find the behavioral dimension?
for seed in [0, 1, 2]:
    try:
        rd = s.rounds(f"fid_signflip_trustfl/seed{seed}")
        tw = rd.dropna(subset=["trust_weights"])
        if len(tw):
            W = np.array([(json.loads(v) if isinstance(v, str) else v)
                          for v in tw["trust_weights"]])
            print(f"trustfl seed{seed} final weights "
                  f"(beh/comp/comm): {W[-1][0]:.2f}/{W[-1][1]:.2f}/{W[-1][2]:.2f}"
                  f"   beh mean last-half: {W[len(W)//2:,0].mean():.2f}")
    except Exception as e:
        print(f"seed{seed}: {e}")
