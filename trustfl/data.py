"""Datasets and non-IID partitioning.

Datasets: MNIST (debug), CIFAR-100, FEMNIST (LEAF-style, natural writer
partition approximated via Dirichlet on EMNIST-byclass if raw LEAF data is
absent), and a synthetic Gaussian-blob dataset for offline CI/smoke tests.

Partitioning: Dirichlet(alpha) label-skew (Hsu et al. 2019), the standard
non-IID protocol in Byzantine-FL papers. A server-side validation split is
carved out FIRST (trusted root data for FLTrust / FedGreed / learnable trust
weights), then the remainder is partitioned across UAV clients.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

NUM_CLASSES = {"mnist": 10, "cifar100": 100, "femnist": 62, "synthetic": 5}


class _SyntheticBlobs(Dataset):
    """5-class Gaussian blobs, 32-dim. Lets the full pipeline run with no
    downloads (container smoke tests) while exercising every code path."""

    def __init__(self, n: int, sample_seed: int, centers_seed: int):
        k, d = 5, 32
        centers = np.random.default_rng(centers_seed).normal(0, 3.0, size=(k, d))
        g = np.random.default_rng(sample_seed)
        y = g.integers(0, k, size=n)
        x = centers[y] + g.normal(0, 2.5, size=(n, d))
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_dataset(cfg, seed: int) -> Tuple[Dataset, Dataset]:
    """Returns (train, test)."""
    name = cfg.dataset.lower()
    if name == "synthetic":
        return (_SyntheticBlobs(4000, seed, centers_seed=seed),
                _SyntheticBlobs(1000, seed + 1, centers_seed=seed))

    import torchvision
    import torchvision.transforms as T

    if name == "mnist":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        tr = torchvision.datasets.MNIST(cfg.data_dir, train=True, download=True, transform=tf)
        te = torchvision.datasets.MNIST(cfg.data_dir, train=False, download=True, transform=tf)
    elif name == "cifar100":
        norm = T.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
        te_tf = T.Compose([T.ToTensor(), norm])
        tr = torchvision.datasets.CIFAR100(cfg.data_dir, train=True, download=True, transform=tr_tf)
        te = torchvision.datasets.CIFAR100(cfg.data_dir, train=False, download=True, transform=te_tf)
    elif name == "femnist":
        # EMNIST-byclass (62 classes) as the FEMNIST base; Dirichlet partition
        # then approximates writer heterogeneity. Swap in raw LEAF splits later
        # if reviewer-grade fidelity is needed.
        tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        tr = torchvision.datasets.EMNIST(cfg.data_dir, split="byclass", train=True, download=True, transform=tf)
        te = torchvision.datasets.EMNIST(cfg.data_dir, split="byclass", train=False, download=True, transform=tf)
    else:
        raise ValueError(f"unknown dataset {name}")
    return tr, te


def _labels_of(ds: Dataset) -> np.ndarray:
    if hasattr(ds, "targets"):
        t = ds.targets
        return t.numpy() if torch.is_tensor(t) else np.asarray(t)
    if hasattr(ds, "y"):
        return ds.y.numpy()
    return np.array([ds[i][1] for i in range(len(ds))])


def split_server_val(train: Dataset, fraction: float, seed: int) -> Tuple[Dataset, Subset]:
    """Carve the trusted server root/validation set (class-balanced)."""
    labels = _labels_of(train)
    g = np.random.default_rng(seed)
    n_val = max(int(len(labels) * fraction), 50)
    per_class = max(n_val // len(np.unique(labels)), 1)
    val_idx = []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        val_idx.extend(g.choice(idx_c, size=min(per_class, len(idx_c)), replace=False))
    val_idx = np.array(sorted(val_idx))
    rest = np.setdiff1d(np.arange(len(labels)), val_idx)
    return Subset(train, rest.tolist()), Subset(train, val_idx.tolist())


def dirichlet_partition(train: Dataset, num_clients: int, alpha: float,
                        min_samples: int, seed: int) -> List[np.ndarray]:
    """Label-skew Dirichlet partition. Resamples until every client has at
    least `min_samples` (standard practice; bounded retries)."""
    base_idx = np.array(train.indices) if isinstance(train, Subset) else np.arange(len(train))
    inner = train.dataset if isinstance(train, Subset) else train
    labels = _labels_of(inner)[base_idx]
    classes = np.unique(labels)
    g = np.random.default_rng(seed)

    for _attempt in range(100):
        parts = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = base_idx[labels == c]
            g.shuffle(idx_c)
            props = g.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(idx_c, cuts)):
                parts[cid].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_samples:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError("Dirichlet partition failed to satisfy min_samples; "
                       "lower min_samples_per_client or raise alpha.")


def iid_partition(train: Dataset, num_clients: int, seed: int) -> List[np.ndarray]:
    base_idx = np.array(train.indices) if isinstance(train, Subset) else np.arange(len(train))
    g = np.random.default_rng(seed)
    g.shuffle(base_idx)
    return [np.array(sorted(p)) for p in np.array_split(base_idx, num_clients)]


def label_variance(train: Dataset, part: np.ndarray, num_classes: int) -> float:
    """sigma_j^2 proxy in the quality factor Q_j, Eq. (31): variance of the
    client's label distribution vs uniform (higher = more skewed)."""
    inner = train.dataset if isinstance(train, Subset) else train
    labels = _labels_of(inner)[part]
    hist = np.bincount(labels, minlength=num_classes) / max(len(labels), 1)
    return float(np.var(hist) * num_classes**2)  # normalized so uniform ~ 0
