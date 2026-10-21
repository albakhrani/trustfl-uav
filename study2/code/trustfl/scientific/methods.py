"""Five prospective identities; legacy aggregation functions remain unchanged.

Effective alpha multiplies post-attack, pre-aggregation client deltas. Internal
q is kept separately. Neither is poisoning damage. No optimizer lives here.
"""
from __future__ import annotations
import math

METHODS = ('trustfl', 'fedavg', 'fltrust_declared_root_fallback',
           'fedgreed_algorithm1', 'fd_distance_validation_local')
RECONSTRUCTION_ATOL = 1e-6
RECONSTRUCTION_RTOL = 1e-5


def krum_count_guard(n, f, m=1):
    """Pure validation only: not a registered scientific comparator."""
    if any(type(x) is not int for x in (n, f, m)) or f < 0:
        return False
    return n >= 2*f+3 and 1 <= m <= n-f-2


def fedgreed_algorithm1(deltas, loss_eval):
    """Reference Algorithm1: stop at FIRST nonimprovement, never best-all-prefix.

    Individual candidate ties have a declared UID tie break. Singleton prefix
    loss is reused. Losses must be finite. Returns (delta, q, query_details).
    """
    import torch
    if not deltas:
        raise ValueError('FedGreed requires a nonempty usable candidate set')
    losses = {j: float(loss_eval(deltas[j])) for j in sorted(deltas)}
    if not all(math.isfinite(v) for v in losses.values()):
        raise FloatingPointError('nonfinite FedGreed individual loss')
    order = sorted(deltas, key=lambda j: (losses[j], j))
    selected = [order[0]]
    best_loss = losses[order[0]]
    result = deltas[order[0]].clone()
    tested = []
    for j in order[1:]:
        trial_ids = selected + [j]
        trial = torch.stack([deltas[u] for u in trial_ids]).mean(0)
        loss = float(loss_eval(trial))
        if not math.isfinite(loss):
            raise FloatingPointError('nonfinite FedGreed prefix loss')
        tested.append({'ids': trial_ids, 'loss': loss})
        if loss >= best_loss:
            break
        selected, best_loss, result = trial_ids, loss, trial
    # A float32 uniform weight is the algebraic coefficient of torch.mean.
    weight = float(torch.tensor(1.0 / len(selected), dtype=result.dtype))
    q = {str(j): weight if j in selected else 0.0 for j in sorted(deltas)}
    return result, q, {'individual_order': order, 'individual_losses': losses,
                       'prefixes_tested': tested, 'selected_ids': selected,
                       'full_root_queries': len(losses) + len(tested)}


def reconcile(out, deltas, alpha, root_delta=None, beta=0.0):
    import torch
    expected = torch.zeros_like(out, dtype=torch.float64, device='cpu')
    scale = 0.0
    for j, delta in sorted(deltas.items()):
        a = float(alpha[str(j)])
        d = delta.detach().cpu().double()
        expected += a * d
        scale += abs(a) * float(d.norm())
    if root_delta is not None and beta:
        root = root_delta.detach().cpu().double()
        expected += beta * root
        scale += abs(beta) * float(root.norm())
    residual = float((out.detach().cpu().double() - expected).norm())
    bound = RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * scale
    return {'residual_l2': residual, 'scale': scale, 'bound': bound,
            'passed': math.isfinite(residual) and residual <= bound}


def aggregate(method, deltas, n_map, *, cfg, trust_scores=None, grad_cons=None,
              root_delta=None, loss_eval=None, state=None):
    import torch
    from ..fl import aggregation as legacy
    from ..trust.selection import aggregation_weights
    if method not in METHODS or not deltas:
        raise ValueError('Unsupported method or empty aggregation invocation')
    if any(not bool(torch.isfinite(d).all()) for d in deltas.values()):
        raise FloatingPointError('nonfinite candidate must be quarantined before aggregation')
    ids = sorted(deltas)
    info, details, beta = {}, {}, 0.0
    if method == 'trustfl':
        omega = aggregation_weights(ids, trust_scores, n_map,
                    t_min=cfg.trust.t_min, gate=cfg.trust.aggregation_gate)
        if cfg.trust.round_consistency_gate and grad_cons:
            omega = {j: w * max(grad_cons.get(j, 1.0), 0.0) for j,w in omega.items()}
        out = legacy.trust_weighted(deltas, omega=omega, info=info)
        q = dict(info['coefficients'])
        details['precursor'] = {str(j): float(v) for j,v in omega.items()}
        alpha = dict(q)
    elif method == 'fedavg':
        out = legacy.fedavg(deltas, n=n_map, info=info)
        q = dict(info['coefficients']); alpha = dict(q)
    elif method == 'fltrust_declared_root_fallback':
        if root_delta is None or not bool(torch.isfinite(root_delta).all()):
            raise FloatingPointError('missing/nonfinite FLTrust root delta')
        out = legacy.fltrust(deltas, root_delta, info=info)
        q = {str(j): float(info['influence'].get(j, 0.0)) for j in ids}
        beta = 1.0 if not any(q.values()) else 0.0
        root_norm = root_delta.norm().clamp_min(1e-12)
        alpha = {str(j): float(torch.tensor(q[str(j)], dtype=torch.float32) *
                     (root_norm / deltas[j].norm().clamp_min(1e-12))) for j in ids}
        details.update(norm_clamp=1e-12, fallback_threshold=1e-12,
                       root_norm_clamped=float(root_norm))
    elif method == 'fedgreed_algorithm1':
        out, q, details = fedgreed_algorithm1(deltas, loss_eval)
        alpha = dict(q)
    else:
        if state is None:
            raise ValueError('FD variant requires persistent per-UID state')
        out = legacy.fedaa(deltas, loss_eval=loss_eval, state=state, info=info)
        q = {str(j): float(info['influence'].get(j, 0.0)) for j in ids}
        alpha = dict(q)
        details.update(select_fraction=.6, steps=8, lr=.8, epsilon=.15,
                       objective='negative_root_loss', published_DDPG=False)
    meta = {'method': method, 'alpha': alpha, 'q': q,
            'root_term': {'coefficient': beta,
                          'norm': float(root_delta.norm()) if root_delta is not None else 0.0,
                          'fallback': bool(beta), 'reason': 'declared_root_only' if beta else None},
            'dtype': str(out.dtype), 'actual_tensor_device': str(out.device),
            'delta_norms': {str(j): float(deltas[j].norm()) for j in ids},
            'details': details}
    if not all(math.isfinite(a) and a >= 0 for a in alpha.values()):
        raise FloatingPointError('invalid effective coefficients')
    meta['aggregate_finite'] = bool(torch.isfinite(out).all())
    meta['reconstruction'] = reconcile(out, deltas, alpha, root_delta, beta)
    return out, meta
