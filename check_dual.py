"""Compute the dual metric (retention vs malicious influence) for the hardened
adaptive-system runs. This settles whether TrustFL wins on the COMBINATION
even when MultiKrum wins on influence alone."""
import json, sys
from trustfl.audit import retention_for_run
from trustfl.store import Store

s = Store('./results_store')
print(f"{'method':10s} {'hm_retention':>13s} {'overall_ret':>12s} {'mal_infl':>10s}  {'final_acc':>10s}")
print("-"*62)
for m in ['trustfl','multikrum','fltrust','fedgreed','fedavg']:
    ovs, hms, mis, accs = [], [], [], []
    for seed in [0,1,2]:
        rid = f'exp_h_adsys_{m}/seed{seed}'
        try:
            ov, hm = retention_for_run(s, rid)
            rows_df = s.rounds(rid)
            mi = [v for v in rows_df.get('malicious_influence_share', []) if v==v]  # drop NaN
            import pandas as pd
            acc = rows_df.dropna(subset=['test_acc'])['test_acc'].iloc[-1] if 'test_acc' in rows_df else float('nan')
            ovs.append(ov); hms.append(hm)
            if len(mi): mis.append(sum(mi)/len(mi))
            accs.append(acc)
        except Exception as e:
            pass
    if hms:
        import statistics as st
        print(f"{m:10s} {st.mean(hms):13.3f} {st.mean(ovs):12.3f} {st.mean(mis):10.3f}  {st.mean(accs):10.3f}")
print("\nINTERPRETATION:")
print("  If TrustFL has HIGH retention + LOW influence, and MultiKrum has")
print("  LOW influence but ALSO LOW retention (discards honest) -> TrustFL wins")
print("  the DUAL metric (Hypothesis 1: the joint tradeoff is the real story).")
print("  If MultiKrum matches TrustFL on BOTH -> Claim 4 is genuinely weak")
print("  (Hypothesis 2: report adaptive modestly, lead with Claims 1 & 3).")