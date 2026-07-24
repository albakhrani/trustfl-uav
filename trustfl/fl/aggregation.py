"""Aggregation rules. Every rule maps {uid: delta} (+ per-uid weights or
server context) -> one flat delta applied to the global model.

Classical baselines:
  fedavg        weighted mean by n_j (McMahan et al. 2017)
  krum          Blanchard et al. 2017 — O(M^2) pairwise distances
  multikrum     Krum's m-best average
  trimmed_mean  Yin et al. 2018 — coordinate-wise beta-trimmed mean
  median        coordinate-wise median (Yin et al. 2018)
Modern baselines:
  fltrust       Cao et al., NDSS 2021 — server root dataset; cosine-ReLU
                trust scores + magnitude normalization
  fedgreed      Kritharakis et al. 2025 (arXiv:2508.18060) — evaluate each
                client update's loss on the server's trusted set; greedily
                keep the k lowest-loss updates, average them
Ours:
  trust_weighted  Eq. (32)-(33): omega_j from the TrustManager
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def _stack(deltas: Dict[int, torch.Tensor]):
    uids = sorted(deltas)
    return uids, torch.stack([deltas[u] for u in uids])


def fedavg(deltas: Dict[int, torch.Tensor], n: Dict[int, int], **_) -> torch.Tensor:
    uids, mat = _stack(deltas)
    w = torch.tensor([n[u] for u in uids], dtype=torch.float32)
    w = w / w.sum()
    return (mat * w[:, None]).sum(0)


def trust_weighted(deltas: Dict[int, torch.Tensor], omega: Dict[int, float], **_) -> torch.Tensor:
    """Eq. (32): w^{t+1} = sum_j omega_j w_j^{t+1}, expressed on deltas."""
    uids, mat = _stack(deltas)
    w = torch.tensor([omega[u] for u in uids], dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-12)
    return (mat * w[:, None]).sum(0)


def krum(deltas: Dict[int, torch.Tensor], f: int = 0, multi_m: int = 1,
         info: Optional[dict] = None, **_) -> torch.Tensor:
    """(Multi-)Krum. Score_i = sum of distances to its M-f-2 nearest peers;
    lowest scores win. Deliberately O(M^2) — that cost is measured for
    flagship claim #3."""
    uids, mat = _stack(deltas)
    m_cnt = len(uids)
    if m_cnt <= 2:
        return mat.mean(0)
    dists = torch.cdist(mat, mat) ** 2
    k = max(m_cnt - f - 2, 1)
    scores = []
    for i in range(m_cnt):
        d = dists[i].clone()
        d[i] = float("inf")
        scores.append(d.topk(k, largest=False).values.sum())
    scores = torch.stack(scores)
    sel = scores.topk(min(multi_m, m_cnt), largest=False).indices
    if info is not None:
        info["influence"] = {uids[i]: 1.0 / len(sel) for i in sel.tolist()}
        nn = {}
        for i in sel.tolist():
            d = dists[i].clone()
            d[i] = float("inf")
            nn[uids[i]] = uids[int(d.argmin())]
        info["nearest_neighbor"] = nn
    return mat[sel].mean(0)


def trimmed_mean(deltas: Dict[int, torch.Tensor], beta: float = 0.1, **_) -> torch.Tensor:
    _, mat = _stack(deltas)
    m_cnt = mat.shape[0]
    k = int(beta * m_cnt)
    if 2 * k >= m_cnt:
        k = max((m_cnt - 1) // 2, 0)
    sorted_mat, _ = mat.sort(dim=0)
    return sorted_mat[k: m_cnt - k].mean(0) if m_cnt - 2 * k > 0 else mat.median(0).values


def median(deltas: Dict[int, torch.Tensor], **_) -> torch.Tensor:
    _, mat = _stack(deltas)
    return mat.median(0).values


def fltrust(deltas: Dict[int, torch.Tensor], server_delta: torch.Tensor,
            info: Optional[dict] = None, **_) -> torch.Tensor:
    """FLTrust (Cao et al. 2021): TS_j = ReLU(cos(delta_j, delta_0));
    each delta rescaled to ||delta_0||; weighted average by TS."""
    uids, mat = _stack(deltas)
    g0 = server_delta
    g0n = g0.norm().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(mat, g0[None, :], dim=1)
    ts = torch.relu(cos)
    if ts.sum() <= 1e-12:
        if info is not None:
            info["influence"] = {u: 0.0 for u in uids}
        return g0.clone()  # nothing trusted: fall back to server update
    if info is not None:
        w = (ts / ts.sum()).tolist()
        info["influence"] = {u: w[i] for i, u in enumerate(uids)}
    scaled = mat * (g0n / mat.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return (scaled * ts[:, None]).sum(0) / ts.sum()


def fedgreed(deltas: Dict[int, torch.Tensor], loss_eval: Callable[[torch.Tensor], float],
             k: int = 5, info: Optional[dict] = None, **_) -> torch.Tensor:
    """FedGreed (2025): rank client updates by loss on the server's trusted
    dataset; average the k lowest-loss updates. `loss_eval(delta)` applies
    the candidate delta to the global model and returns validation loss."""
    uids, mat = _stack(deltas)
    losses = torch.tensor([loss_eval(mat[i]) for i in range(len(uids))])
    sel = losses.topk(min(k, len(uids)), largest=False).indices
    if info is not None:
        info["influence"] = {uids[i]: 1.0 / len(sel) for i in sel.tolist()}
    return mat[sel].mean(0)


def sophon(deltas: Dict[int, torch.Tensor], server_delta: torch.Tensor,
           beta: float = 1.0, info: Optional[dict] = None, **_) -> torch.Tensor:
    """Sophon (Gui et al., IEEE TDSC 2025) — EXACT implementation of
    Algorithm 1 / Eqs. (1)-(7), audited line-by-line against the paper
    (fidelity audit 2026-07-14; supersedes the earlier structural version).

      Eq.1  normalize each gradient to the reference magnitude.
      Eq.2  CTS_i = cos(g_hat_i, g_r)  (raw cosine — NOT ReLU; the paper
            explicitly rejects FLTrust's ReLU as discarding information).
      Eq.3  min-max normalize CTS over clients (only the worst -> 0).
      Eq.4  DTS_i = 1 - cos(g_hat_i, nearest neighbour) — penalizes clones,
            REWARDS isolated gradients.
      Eq.5  min-max normalize DTS.
      Eq.6  w_i = (1+beta^2) * CTS_i*DTS_i / (beta^2*CTS_i + DTS_i), beta=1.
      Eq.7  weighted average of the normalized gradients.

    KEY LIMITATION (unchanged by the audit): every signal is a function of
    the submitted gradient alone; system-dimension attackers submitting
    valid gradients pass both scores.
    """
    uids, mat = _stack(deltas)
    n = len(uids)
    g0 = server_delta
    g0n = g0.norm().clamp_min(1e-12)
    scaled = mat * (g0n / mat.norm(dim=1, keepdim=True).clamp_min(1e-12))  # Eq.1
    cos0 = torch.nn.functional.cosine_similarity(scaled, g0[None, :], dim=1)  # Eq.2

    def _minmax(x):
        lo, hi = x.min(), x.max()
        rng = (hi - lo).clamp_min(1e-12)
        return (x - lo) / rng

    cts = _minmax(cos0) if n > 1 else torch.ones(n)                        # Eq.3
    if n > 1:
        nm = scaled / scaled.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = nm @ nm.t()
        C.fill_diagonal_(-2.0)                    # exclude self from the max
        nn_sim = C.max(dim=1).values              # nearest-neighbour cosine
        dts = _minmax(1.0 - nn_sim)                                        # Eq.4-5
    else:
        dts = torch.ones(n)
    denom = ((beta ** 2) * cts + dts).clamp_min(1e-12)
    weight = (1.0 + beta ** 2) * (cts * dts) / denom                       # Eq.6
    if weight.sum() <= 1e-12:
        if info is not None:
            info["influence"] = {u: 0.0 for u in uids}
        return g0.clone()
    w = weight / weight.sum()
    if info is not None:
        info["influence"] = {u: float(w[i]) for i, u in enumerate(uids)}
    return (scaled * w[:, None]).sum(0)                                    # Eq.7


def fedaa(deltas: Dict[int, torch.Tensor], loss_eval: Callable[[torch.Tensor], float],
          select_frac: float = 0.6, weight_lr: float = 0.8, n_weight_steps: int = 8,
          state: Optional[dict] = None, info: Optional[dict] = None, **_) -> torch.Tensor:
    """FedAA (He, Chen & Zhang, AAAI 2025): RL-adaptive aggregation.

    Structural reimplementation of the published pipeline:
      1. CLIENT SELECTION by model-parameter proximity: keep the top
         select_frac of clients closest to the coordinate-wise median of
         updates (their geometric filter against Byzantine outliers).
      2. ADAPTIVE AGGREGATION WEIGHTS over the selected clients, optimized
         against the paper's reward signal: validation-set performance of the
         aggregated model. The original uses a DDPG agent whose continuous
         action is the weight vector; at our round budget (~100 rounds) a DDPG
         agent has too few samples to converge, so we optimize the SAME
         objective directly -- finite-difference reward-gradient steps on the
         weight simplex, warm-started from the previous round's weights
         (persistent `state`), which converges FASTER than DDPG at this scale.
         This simplification is therefore favorable to FedAA.

    KEY LIMITATION (documented for our comparison): both signals -- parameter
    proximity and validation reward -- are functions of the SUBMITTED UPDATE
    alone. A client that submits a proximate, validation-improving gradient
    while misreporting compute, delaying strategically, or participating
    selectively passes both filters. FedAA, like all update-space methods, is
    structurally blind to the system dimension.
    """
    uids, mat = _stack(deltas)
    n = len(uids)
    if n == 1:
        if info is not None:
            info["influence"] = {uids[0]: 1.0}
        return mat[0].clone()
    # (1) Algorithm 1 (exact): pairwise Euclidean distance matrix; keep the
    #     top select_frac clients with MINIMUM ROW-SUM of distances to all
    #     others (medoid-proximity; FABA-inspired). [Fidelity audit
    #     2026-07-14: replaces the earlier distance-to-coordinate-median
    #     criterion, which was behaviourally close but not Algorithm 1.]
    D = torch.cdist(mat, mat, p=2)
    dists = D.sum(dim=1)
    k = max(1, int(round(select_frac * n)))
    keep = torch.argsort(dists)[:k]
    sel_uids = [uids[i] for i in keep.tolist()]
    sel = mat[keep]
    # (2) adaptive weights vs validation reward, warm-started PER CLIENT.
    # Participants change round to round, so positional warm-starts are
    # useless; we persist a logit per uid (mirrors DDPG's per-client action).
    logit_map = state.setdefault("logit_map", {}) if state is not None else {}
    logits = torch.tensor([float(logit_map.get(u, 0.0)) for u in sel_uids])
    def reward(w):
        return -loss_eval((sel * w[:, None]).sum(0))  # reward = -val loss
    for _ in range(n_weight_steps):
        w = torch.softmax(logits, 0)
        base = reward(w)
        grad = torch.zeros(k)
        eps = 0.15
        for i in range(k):
            lp = logits.clone(); lp[i] += eps
            grad[i] = (reward(torch.softmax(lp, 0)) - base) / eps
        logits = logits + weight_lr * grad
    w = torch.softmax(logits, 0)
    if state is not None:
        for i, u in enumerate(sel_uids):
            logit_map[u] = float(logits[i])
    if info is not None:
        infl = {u: 0.0 for u in uids}
        for i, u in enumerate(sel_uids):
            infl[u] = float(w[i])
        info["influence"] = infl
    return (sel * w[:, None]).sum(0)


AGGREGATORS = {
    "fedavg": fedavg,
    "trust_weighted": trust_weighted,
    "krum": krum,
    "multikrum": krum,          # dispatched with multi_m > 1
    "trimmed_mean": trimmed_mean,
    "median": median,
    "fltrust": fltrust,
    "fedgreed": fedgreed,
    "sophon": sophon,
    "fedaa": fedaa,
}
