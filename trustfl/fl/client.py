"""Local training. One Client per UAV; SGD with optional FedProx proximal
term (mu > 0). Returns the model DELTA (update = local - global), which is
what aggregators and gradient attacks operate on.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def flatten_state(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().flatten().float() for v in sd.values()])


def unflatten_like(vec: torch.Tensor, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out, i = {}, 0
    for k, v in sd.items():
        n = v.numel()
        out[k] = vec[i:i + n].view_as(v).to(v.dtype)
        i += n
    return out


class Client:
    def __init__(self, uid: int, dataset, indices: np.ndarray, train_cfg, device, seed: int):
        self.uid = uid
        self.cfg = train_cfg
        self.device = device
        self.subset = Subset(dataset, indices.tolist())
        self.n = len(indices)
        self._gen = torch.Generator().manual_seed(seed * 100003 + uid)

    def local_update(self, model: torch.nn.Module, malicious_hook=None) -> Dict:
        """Runs local epochs; returns dict with flat delta and stats.
        `malicious_hook(kind='label', y)->y` lets label-space attacks act
        during training (label flipping); delta-space attacks act after."""
        cfg = self.cfg
        global_flat = flatten_state(model.state_dict())
        local = type(model)() if False else None  # placeholder, replaced below
        import copy
        local = copy.deepcopy(model).to(self.device)
        local.train()
        opt = torch.optim.SGD(local.parameters(), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        loader = DataLoader(self.subset, batch_size=cfg.batch_size,
                            shuffle=True, generator=self._gen, drop_last=False)
        global_params = [p.detach().clone() for p in local.parameters()] if cfg.fedprox_mu > 0 else None

        n_batches = 0
        loss_sum = 0.0
        for _ep in range(cfg.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if malicious_hook is not None:
                    y = malicious_hook("label", y)
                opt.zero_grad()
                out = local(x)
                loss = F.cross_entropy(out, y)
                if cfg.fedprox_mu > 0:  # FedProx (Li et al. 2020)
                    prox = sum(((p - g) ** 2).sum() for p, g in zip(local.parameters(), global_params))
                    loss = loss + cfg.fedprox_mu / 2 * prox
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach())
                n_batches += 1

        local_flat = flatten_state(local.state_dict()).cpu()
        delta = local_flat - global_flat.cpu()
        return {
            "uid": self.uid,
            "delta": delta,
            "n": self.n,
            "train_loss": loss_sum / max(n_batches, 1),
        }
