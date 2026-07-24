"""Per-seed dual metric — verify the retention-influence win holds across ALL
seeds, not just the mean. A headline claim must not rest on seed averaging."""
from trustfl.audit import retention_for_run
from trustfl.store import Store

s = Store('./results_store')
print("Per-seed hm_retention / mal_infl (headline robustness check):\n")
for m in ['trustfl','multikrum','fltrust','fedgreed','fedavg']:
    line = f"  {m:10s} "
    for seed in [0,1,2]:
        rid = f'exp_h_adsys_{m}/seed{seed}'
        try:
            ov, hm = retention_for_run(s, rid)
            rd = s.rounds(rid)
            mi = [v for v in rd.get('malicious_influence_share', []) if v==v]
            mim = sum(mi)/len(mi) if mi else float('nan')
            line += f"s{seed}:[ret={hm:.2f} infl={mim:.2f}] "
        except Exception as e:
            line += f"s{seed}:[err] "
    print(line)
print("\nTrustFL should show ret>0.80 AND infl<0.28 in ALL three seeds;")
print("baselines should show ret<0.65 in all seeds (honest-data discard).")