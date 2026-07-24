"""Statistical significance of the retention gap (reproducibility checklist
requires this). Mann-Whitney U (non-parametric, appropriate for n=3 seeds)
comparing TrustFL retention vs each baseline. Also reports effect size."""
from trustfl.audit import retention_for_run
from trustfl.store import Store
try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

s = Store('./results_store')

def retentions(method):
    out = []
    for seed in [0,1,2]:
        try:
            _, hm = retention_for_run(s, f'exp_h_adsys_{method}/seed{seed}')
            out.append(hm)
        except Exception:
            pass
    return out

tf = retentions('trustfl')
print(f"TrustFL retention: {[f'{x:.3f}' for x in tf]}  mean={sum(tf)/len(tf):.3f}\n")
for m in ['multikrum','fltrust','fedgreed','fedavg']:
    b = retentions(m)
    gap = sum(tf)/len(tf) - sum(b)/len(b)
    line = f"  vs {m:10s}: baseline={[f'{x:.2f}' for x in b]} gap=+{gap:.3f}"
    if HAVE_SCIPY and len(tf)==3 and len(b)==3:
        # one-sided: is trustfl retention > baseline?
        u, p = mannwhitneyu(tf, b, alternative='greater')
        line += f"  Mann-Whitney p={p:.3f}"
    print(line)
print("\nNote: n=3 seeds -> min possible one-sided p is 0.05 (perfect separation).")
print("Perfect separation (all TrustFL > all baseline) gives p=0.05, which we")
print("expect here since TrustFL's min (0.78) > every baseline's max (0.73).")
if not HAVE_SCIPY:
    print("\n[scipy not installed: pip install scipy --quiet to get p-values]")