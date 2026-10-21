# TrustFL-UAV

**Revision R1 (October 2026).** The code, regenerated data store, Study 2 protocol, traces and
analysis, and the claims-audit record for the revised manuscript (IEEE IoT-J, IoT-71387-2026.R1)
are on the [`revision-r1`](https://github.com/albakhrani/trustfl-uav/tree/revision-r1) branch.
This `main` branch is kept unchanged as the version submitted with the original manuscript.

**Distinguishing Malicious from Unreliable Participants in Federated Learning
over UAV Networks**

This repository contains the simulation code, experiment configurations, and
result aggregates for the manuscript *"TrustFL-UAV: Distinguishing Malicious
from Unreliable Participants in Federated Learning over UAV Networks"*,
**submitted to the IEEE Internet of Things Journal (under review)**. It
implements the trust-assessment mechanism,
trust-constrained client selection, dual-decomposition resource allocation,
and the UAV physics model (Gauss–Markov mobility, energy, and Rayleigh
air-to-ground channel), together with all baselines and the three attack
families (gradient-space, system-dimension, and adaptive adversaries).

Every quoted number in the paper is machine-checked by `claims_audit.py`
against the exported CSVs in `paper_assets/data/`.

## Install

Python 3.12 is recommended (3.11 also works; avoid 3.13+ until PyTorch
support stabilizes).

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux/macOS:  source .venv/bin/activate
python -m pip install --upgrade pip
```

For a **GPU build**, install PyTorch from its own index first, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

A **CPU-only** environment needs just `pip install -r requirements.txt`
(the default PyPI `torch` wheel is CPU-only). Detailed Windows/CUDA setup is
in [`SETUP_WINDOWS.md`](SETUP_WINDOWS.md). Tested with Python 3.12.10,
torch 2.12 (cu126), numpy 2.4, pandas 3.0, scipy 1.18, matplotlib 3.11.

## Reproduce the paper numbers

The exported aggregates are committed under `paper_assets/data/*.csv`, so the
audit and figures run with no experiments required.

**1. Re-export the CSVs from the results store** (optional — regenerates the
committed CSVs identically from `results_store/`):

```bash
python -m trustfl.export --store ./results_store --out ./paper_assets/data
```

**2. Machine-check every quoted number** (exits 0 when all Group-A claims
pass; Group-B prints the retention-derived values):

```bash
python claims_audit.py                 # add --matrix for the Table III grid
```

**3. Regenerate the EuroSAT figure (F9) from the CSVs:**

```bash
python paper_assets/make_fig_eurosat.py
```

This writes `paper_assets/figures/F9_eurosat.{pdf,png}` deterministically
(byte-identical on re-run). The remaining figures (F1–F7 and the
dual/heatmap/mechanism panels) are provided pre-rendered in
`paper_assets/figures/`; regenerating them with
`python -m trustfl.figures --store ./results_store <panel>` requires a
paper-tagged results store.

## Run an experiment

General form:

```bash
python -m trustfl.experiments.run --config configs/<file>.yaml \
    --override method.name=krum train.rounds=30 attack.malicious_fraction=0.3
```

The flagship hardened comparison — EuroSAT under a 40% adaptive
system+model adversary (30 UAVs, 60 rounds, 3 seeds) — is one command per
method (this is exactly what produced the `eur_adsys_*` runs behind F9):

```bash
python -m trustfl.experiments.run --config configs/eurosat_hardened.yaml \
    --override name=eur_adsys_trustfl method.name=trustfl
```

Each seed writes `results/<name>/seed<k>/` with `metrics.jsonl` (per round),
the fully resolved `config_resolved.yaml`, and `run_meta.json` (hardware,
torch version, git commit). `results/` is not tracked; ingest runs into the
store to aggregate them.

### Methods (`method.name`)

| name | what it is |
|---|---|
| `trustfl` | ours: trust-constrained selection + trust-gated weighted aggregation |
| `fedavg` | McMahan et al. 2017 (add `train.fedprox_mu>0` for FedProx) |
| `krum` / `multikrum` | Blanchard et al. 2017 |
| `trimmed_mean` / `median` | Yin et al. 2018 |
| `fltrust` | Cao et al., NDSS 2021 — server root dataset, cosine trust |
| `fedgreed` | greedy lowest-val-loss selection |
| `sophon`, `fedaa` | additional robust-aggregation baselines |

### Attacks (`attack.attack_type`)

Gradient-space: `label_flip`, `sign_flip`, `grad_scale`, `random_update`.
System-dimension (invisible to gradient-only defenses): `resource_misreport`,
`strategic_delay`, `selective_participation`. Adaptive: `adaptive_trust` and
`adaptive_system` — an on-off adversary that builds trust to a target,
attacks, goes quiet, and rebuilds.

## Datasets

All datasets auto-download via `torchvision` into `data/` (gitignored) on
first use:

- **MNIST** — quick baseline / debug.
- **FEMNIST** — EMNIST-`byclass` (62 classes) as the base, partitioned across
  clients with a Dirichlet split.
- **EuroSAT** — 27,000 64×64 RGB aerial land-use images, 10 classes
  (~90 MB). Deterministic stratified 80/20 split (fixed split seed 42,
  independent of the run seed).
- **CIFAR-100** — also supported.
- **synthetic** — Gaussian-blob data for offline smoke tests (no download).

## Layout

```
trustfl/                 the package
  config.py              dataclass config + YAML + CLI overrides
  seeding.py             seeds, per-stream RNGs, run metadata
  data.py                dataset loading + Dirichlet partition
  models.py              MLP, CNNs (MNIST/FEMNIST/EuroSAT), ResNet-18-GN
  uav/network.py         UAV mobility, energy, and channel physics
  trust/trust.py         trust assessment + learnable dimension weights
  trust/selection.py     trust-constrained selection + resource allocation
  fl/client.py           local SGD (+FedProx), delta extraction
  fl/aggregation.py      all aggregation rules
  fl/server.py           round loop; attacks applied server-blind
  attacks/attacks.py     gradient / system / adaptive adversaries
  store.py, export.py    results store (SQLite + parquet) and CSV export
  metrics_matrix.py      Table III facet grid
  experiments/run.py     entry point, multi-seed
configs/                 one YAML per experiment
tests/                   unit tests (pytest)
paper_assets/            exported CSVs, tables, and figures
  data/*.csv             the aggregates every claim is audited against
  make_fig_eurosat.py    standalone EuroSAT figure (F9) regenerator
results_store/           canonical run store (SQLite + parquet)
repro_check/             a sample verified reproduction run
claims_audit.py          machine-checks every quoted number (exit 0 = pass)
```

## Hardware note

Experiments were run on a single NVIDIA RTX 4080 Laptop GPU (12 GB). No
multi-GPU or cluster is required. Every experiment also runs on CPU for
deterministic replay — add `--override train.device=cpu`; the synthetic
smoke test (`configs/smoke_synthetic.yaml`) exercises the full pipeline on
CPU in seconds and downloads nothing.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite the manuscript:

```bibtex
@article{albakhrani2026trustfl,
  title   = {TrustFL-UAV: Distinguishing Malicious from Unreliable Participants
             in Federated Learning over UAV Networks},
  author  = {AL-Bakhrani, Ali A. and Xu, Zichuan and Obaidat, Mohammad S. and
             Li, Mingchu and Xia, Qiufen and Li, Zhongwei},
  journal = {IEEE Internet of Things Journal},
  note    = {under review},
  year    = {2026}
}
```
