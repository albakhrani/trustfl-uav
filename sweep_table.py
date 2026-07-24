"""Print the robustness sweep as a table: each supported facet at each
malicious fraction, per method. Tells us whether TrustFL's advantage SCALES."""
import numpy as np
from trustfl.store import Store
from trustfl.metrics_matrix import facets_for_run

s = Store('./results_store')
fractions = [0.1, 0.2, 0.3, 0.4]
methods = ['trustfl', 'fltrust', 'fedgreed', 'multikrum', 'fedavg']
facets = [('hm_retention', 'RETENTION (higher=better)'),
          ('deliv_eff', 'DELIVERY EFF (higher=better)'),
          ('energy_per_update', 'ENERGY/UPDATE (lower=better)'),
          ('mal_infl', 'MAL INFLUENCE (context)')]

for facet, label in facets:
    print(f"\n=== {label} vs malicious fraction ===")
    print(f"{'method':11s} " + "  ".join(f"f={f}" for f in fractions))
    print("-"*44)
    rows = {}
    for m in methods:
        vals = []
        for f in fractions:
            runs = s.query()
            runs = runs[runs['name'] == f'rs_f{f}_{m}']['run_id'].tolist()
            seedvals = [facets_for_run(s, r).get(facet) for r in runs]
            seedvals = [v for v in seedvals if v == v]
            vals.append(np.mean(seedvals) if seedvals else float('nan'))
        rows[m] = vals
        star = " *" if m == "trustfl" else ""
        fmt = "  ".join(f"{v:5.1f}" if facet=='energy_per_update' else f"{v:.3f}" for v in vals)
        print(f"{m:11s} {fmt}{star}")
    # does TrustFL's gap over the best baseline grow with fraction?
    if facet in ('hm_retention','deliv_eff'):
        tf = rows['trustfl']
        gaps = [tf[i] - max(rows[m][i] for m in methods if m!='trustfl') for i in range(len(fractions))]
        print(f"  TrustFL gap over best baseline: " + "  ".join(f"{g:+.3f}" for g in gaps))
        print(f"  -> {'WIDENS with threat' if gaps[-1]>gaps[0] else 'stable/narrows'}")
