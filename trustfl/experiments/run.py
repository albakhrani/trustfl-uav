"""Experiment entry point.

    python -m trustfl.experiments.run --config configs/mnist_fedavg_debug.yaml
    python -m trustfl.experiments.run --config ... --override method.name=krum train.rounds=30

Multi-seed: set `seeds: [0,1,2,3,4]` in YAML; each seed gets its own
subdirectory with metrics.jsonl, resolved config, and run metadata —
exactly what the reproducibility checklist asks to report.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

from ..attacks.attacks import AttackController
from ..config import ExperimentConfig
from ..data import (dirichlet_partition, iid_partition, label_variance,
                    load_dataset, split_server_val, NUM_CLASSES)
from ..fl.client import Client
from ..fl.server import Server
from ..models import build_model
from ..seeding import RunMeta, set_all_seeds
from ..trust.trust import TrustManager
from ..uav.network import UAVNetwork


def run_single(cfg: ExperimentConfig) -> dict:
    device = cfg.train.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = os.path.join(cfg.out_dir, cfg.name, f"seed{cfg.seed}")
    os.makedirs(out_dir, exist_ok=True)
    set_all_seeds(cfg.seed)
    RunMeta(seed=cfg.seed, config_dump=cfg.to_dict()).save(out_dir)
    cfg.dump(os.path.join(out_dir, "config_resolved.yaml"))

    # data
    train, test = load_dataset(cfg.data, cfg.seed)
    train, server_val = split_server_val(train, cfg.data.server_val_fraction, cfg.seed)
    n_clients = cfg.uav.num_uavs
    if cfg.data.partition == "dirichlet":
        parts = dirichlet_partition(train, n_clients, cfg.data.dirichlet_alpha,
                                    cfg.data.min_samples_per_client, cfg.seed)
    else:
        parts = iid_partition(train, n_clients, cfg.seed)
    n_cls = cfg.model.num_classes or NUM_CLASSES[cfg.data.dataset]
    base = train.dataset if hasattr(train, "dataset") else train
    sigma2 = {j: label_variance(train, parts[j], n_cls) for j in range(n_clients)}

    # components
    model = build_model(cfg.model, cfg.data.dataset)
    clients = [Client(j, base, parts[j], cfg.train, device, cfg.seed) for j in range(n_clients)]
    network = UAVNetwork(cfg.uav, cfg.seed)
    trust_mgr = TrustManager(cfg.trust, n_clients)
    attack = AttackController(cfg.attack, n_clients, cfg.seed)
    attack.apply_static_effects(network)

    server = Server(cfg, model, clients, network, trust_mgr, attack,
                    server_val, test, sigma2, device, out_dir)

    print(f"[{cfg.name} seed={cfg.seed}] device={device} method={cfg.method.name} "
          f"attack={cfg.attack.attack_type}({cfg.attack.malicious_fraction}) "
          f"uavs={n_clients} rounds={cfg.train.rounds}", flush=True)

    t0 = time.time()
    last = {}
    for t in range(cfg.train.rounds):
        last = server.run_round(t)
        if "test_acc" in last:
            print(f"  round {t:3d}  test_acc={last['test_acc']:.4f} "
                  f"test_loss={last['test_loss']:.4f} "
                  f"delivered={len(last['delivered'])}", flush=True)
    dur = time.time() - t0
    print(f"[{cfg.name} seed={cfg.seed}] done in {dur/60:.1f} min "
          f"final_acc={last.get('test_acc')}", flush=True)
    return last


def apply_overrides(cfg: ExperimentConfig, overrides) -> ExperimentConfig:
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        obj = cfg
        *path, leaf = key.split(".")
        for p in path:
            obj = getattr(obj, p)
        cur = getattr(obj, leaf)
        if isinstance(cur, list) or (cur is None and val.startswith("[")):
            items = [x for x in val.strip("[]").split(",") if x.strip()]
            val = [int(float(x)) if x.strip().replace(".", "", 1).lstrip("-").isdigit()
                   else x.strip() for x in items]
        elif isinstance(cur, bool):
            val = val.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            val = int(val)
        elif isinstance(cur, float):
            val = float(val)
        setattr(obj, leaf, val)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="dot-path overrides, e.g. method.name=krum train.rounds=30")
    args = ap.parse_args()

    cfg = apply_overrides(ExperimentConfig.load(args.config), args.override)
    seeds = cfg.seeds or [cfg.seed]
    results = []
    for s in seeds:
        results.append(run_single(cfg.with_seed(int(s))))
    accs = [r.get("test_acc") for r in results if r.get("test_acc") is not None]
    if len(accs) > 1:
        print(f"\n=== {cfg.name}: final acc over {len(accs)} seeds: "
              f"mean={np.mean(accs):.4f} std={np.std(accs):.4f} ===")


if __name__ == "__main__":
    sys.exit(main())