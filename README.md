# TrustFL-UAV Simulation Environment

Trust-aware federated learning for UAV-assisted edge computing — simulation
code for the AAAI-2027 submission. Implements the draft's Algorithms 1–4
(trust assessment, trust-constrained selection, dual-decomposition resource
allocation, main loop), the UAV physics of Section III (Gauss-Markov
mobility Eqs. 4–6, energy Eqs. 7–10, Rayleigh air-to-ground channel
Eqs. 11–13), all baselines, and all three attack families.

First-time setup: see `SETUP_WINDOWS.md`.

## Run anything

    python -m trustfl.experiments.run --config configs/<file>.yaml
    python -m trustfl.experiments.run --config configs/<file>.yaml \
        --override method.name=krum train.rounds=30 attack.malicious_fraction=0.3

Multi-seed: put `seeds: [0,1,2,3,4]` in the YAML — each seed writes its own
`results/<name>/seed<k>/` with `metrics.jsonl` (per-round), the fully
resolved `config_resolved.yaml`, and `run_meta.json` (hardware, torch
version, git commit) — the reproducibility checklist fills itself from these.

## Methods (`method.name`)

| name | what it is |
|---|---|
| `trustfl` | ours: Alg. 2 selection + trust-gated weighted aggregation |
| `fedavg` | McMahan et al. 2017 (add `train.fedprox_mu>0` for FedProx) |
| `krum` / `multikrum` | Blanchard et al. 2017 — deliberately O(M²) |
| `trimmed_mean` / `median` | Yin et al. 2018 |
| `fltrust` | Cao et al., NDSS 2021 — server root dataset, cosine trust |
| `fedgreed` | Kritharakis et al. 2025 — greedy lowest-val-loss selection |

## Attacks (`attack.attack_type`)

Gradient-space: `label_flip`, `sign_flip`, `grad_scale`, `random_update`.
System-dimension (invisible to gradient-only defenses): `resource_misreport`,
`strategic_delay`, `selective_participation`.
Adaptive: `adaptive_trust` — on-off adversary that builds trust to a target,
attacks, goes quiet, rebuilds.

## Trust mechanism knobs (`trust.*`)

- `weight_mode: fixed|learnable` — learnable = exponentiated-gradient online
  update of the dimension weights (modernization #1)
- `aggregation_gate` — ω ∝ ReLU(T−T_min)·n (ours) vs draft's linear Eq. 33
- `round_consistency_gate` — multiply ω by current-round gradient-consistency
  (closes the EWMA lag an adaptive adversary exploits)
- `asym_down_factor` — asymmetric EWMA: trust falls fast, recovers slowly
  (1.0 restores the draft's symmetric update)

The last three are deliberate, documented deviations from the draft (the
draft's mean-referenced consistency and linear weighting fail under strong
attacks — verified in smoke tests). Each is an ablation in the paper.

## Flagship experiment mapping

| Claim | Config seed file | Sweep |
|---|---|---|
| (a) honest-outlier retention | `exp_a_honest_outlier.yaml` | `method.name` ∈ {trustfl, krum, multikrum, fedavg, fltrust} |
| (b) system-dimension attacks | smoke/mnist config | `attack.attack_type` ∈ system attacks × `method.name` ∈ {trustfl, fltrust, fedgreed} |
| (c) cost vs robustness | any | `uav.num_uavs` ∈ {20..100}; `agg_time_s` logged per round |
| (d) adaptive adversaries | any | `attack.attack_type=adaptive_trust` × methods; ablate `round_consistency_gate`, `asym_down_factor` |
| (e) scalability + accuracy | mnist/femnist/cifar configs | standard tables, `seeds: [...]` |

## Layout

    trustfl/
      config.py           dataclass config + YAML + CLI overrides
      seeding.py          seeds, per-stream RNGs, run metadata
      data.py             MNIST/CIFAR-100/FEMNIST/synthetic + Dirichlet partition
      models.py           MLP, CNN (MNIST/FEMNIST), ResNet-18-GN (CIFAR-100)
      uav/network.py      Sec. III physics (Eqs. 3-13)
      trust/trust.py      Algorithm 1 + learnable weights (Eqs. 17-25)
      trust/selection.py  Algorithms 2-3 (Eqs. 26-39)
      fl/client.py        local SGD (+FedProx), delta extraction
      fl/aggregation.py   all aggregation rules
      fl/server.py        Algorithm 4 round loop, attacks applied server-blind
      attacks/attacks.py  gradient / system / adaptive adversaries
      experiments/run.py  entry point, multi-seed
    configs/              one YAML per experiment
