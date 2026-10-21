"""Explicit base identities and scoped sample access, without altering native data."""
from __future__ import annotations


def base_identity(dataset, index, namespace):
    """Compose every Subset mapping, retaining a declared leaf split namespace."""
    i = int(index)
    if isinstance(index, bool) or i != index or not 0 <= i < len(dataset):
        raise ValueError("Invalid dataset index")
    seen = set()
    while hasattr(dataset, "dataset"):
        if id(dataset) in seen:
            raise ValueError("Cyclic dataset view")
        seen.add(id(dataset))
        if hasattr(dataset, "indices"):
            i = int(dataset.indices[i])
        elif not isinstance(dataset, EventDataset):
            raise ValueError("Unsupported dataset wrapper without explicit index mapping")
        dataset = dataset.dataset
        if not 0 <= i < len(dataset):
            raise ValueError("Nested index out of bounds")
    declared = getattr(dataset, "cp_namespace", namespace)
    if declared != namespace:
        raise ValueError("Wrong dataset split namespace")
    ids = getattr(dataset, "cp_base_ids", None)
    return {"namespace": namespace, "index": int(ids[i]) if ids is not None else i}


def expand_indices(dataset, indices, namespace):
    return [base_identity(dataset, i, namespace) for i in indices]


class SavedTensorDataset:
    def __init__(self, x, y, namespace, base_ids=None):
        if len(x) != len(y):
            raise ValueError("Feature/label length mismatch")
        self.x, self.y = x, y
        self.cp_namespace = namespace
        self.cp_base_ids = list(range(len(y))) if base_ids is None else list(base_ids)
        if len(self.cp_base_ids) != len(y) or len(set(self.cp_base_ids)) != len(y):
            raise ValueError("Invalid base IDs")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class EventDataset:
    """A DataLoader-compatible wrapper with deterministic, namespaced sample access.

    This wrapper retains the original transform chain. It neither downloads data
    nor converts a stochastic transform to a deterministic production transform.
    """
    def __init__(self, dataset, context, purpose, uid=-1, namespace=None):
        self.dataset, self.context, self.purpose, self.uid = dataset, context, purpose, uid
        leaf = dataset
        while hasattr(leaf, "dataset"):
            leaf = leaf.dataset
        self.namespace = namespace or getattr(leaf, "cp_namespace", purpose)
        self.accessed_ids = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ident = base_identity(self.dataset, index, self.namespace)
        ctx = self.context
        epoch = int(getattr(ctx, "local_epoch", 0)) if self.uid >= 0 else 0
        with ctx.rng.scope("per_sample_transform", ctx.round_idx, self.uid,
                           epoch=epoch, sample=ident["index"], occurrence=0,
                           operation=self.purpose + "|" + self.namespace, device="cpu"):
            value = self.dataset[index]
        self.accessed_ids.append(ident)
        if hasattr(ctx, "record_payload"):
            ctx.record_payload("per_sample_transform", self.uid, value[0],
                               epoch=epoch, sample=ident["index"], occurrence=0,
                               operation=self.purpose + "|" + self.namespace)
        return value


def build_views(train_base, test_base, indices, context=None):
    """Production runner's view builder; tests check actual access through it."""
    import numpy as np
    from torch.utils.data import Subset
    train_ns, test_ns = indices["train_namespace"], indices["test_namespace"]
    clients, root, test = indices["client_local"], indices["root_local"], indices["test_local"]
    all_client = [i for part in clients for i in part]
    for values, limit in [(all_client + root, len(train_base)), (test, len(test_base))]:
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= limit for i in values):
            raise ValueError("Noninteger or out-of-range index")
        if len(values) != len(set(values)):
            raise ValueError("Repeated or leaking sample index")
    if set(all_client + root) != set(range(len(train_base))) or set(test) != set(range(len(test_base))):
        raise ValueError("Missing coverage in immutable views")
    computed = {
        "client_base": [expand_indices(train_base, part, train_ns) for part in clients],
        "root_base": expand_indices(train_base, root, train_ns),
        "test_base": expand_indices(test_base, test, test_ns),
    }
    for key, value in computed.items():
        if indices[key] != value:
            raise ValueError("Saved base IDs do not match actual view: " + key)
    root_ids = {(x["namespace"], x["index"]) for x in computed["root_base"]}
    client_ids = {(x["namespace"], x["index"]) for p in computed["client_base"] for x in p}
    test_ids = {(x["namespace"], x["index"]) for x in computed["test_base"]}
    if root_ids & client_ids or test_ids & (root_ids | client_ids):
        raise ValueError("Dataset split/root leakage")
    return ([(train_base, np.asarray(part, dtype=np.int64)) for part in clients],
            Subset(train_base, root), Subset(test_base, test))
