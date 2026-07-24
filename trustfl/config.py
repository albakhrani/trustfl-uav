"""Experiment configuration. One YAML = one experiment; code never changes.

Every field has a default so smoke configs stay short. `ExperimentConfig.load`
reads YAML; `dump` writes the fully-resolved config next to the results
(reproducibility checklist: final hyper-parameters are logged by construction).
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml


def _coerce(default, value):
    """Coerce YAML value to the default's type (PyYAML reads '2.0e5' as str)."""
    if value is None or default is None:
        return value
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
    if isinstance(default, float) and not isinstance(value, float):
        return float(value)
    if isinstance(default, int) and not isinstance(default, bool) and isinstance(value, str):
        return int(float(value))
    return value


@dataclass
class DataConfig:
    dataset: str = "mnist"            # mnist | cifar100 | femnist | synthetic
    data_dir: str = "./data"
    partition: str = "dirichlet"      # dirichlet | iid
    dirichlet_alpha: float = 0.5      # lower = more non-IID
    server_val_fraction: float = 0.01 # trusted root/validation set on server
                                      # (used by FLTrust, FedGreed, learnable trust weights)
    min_samples_per_client: int = 10


@dataclass
class ModelConfig:
    name: str = "auto"                # auto | mlp | cnn_mnist | cnn_femnist | resnet18
    num_classes: int = 0              # 0 = infer from dataset


@dataclass
class TrainConfig:
    rounds: int = 100
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    fedprox_mu: float = 0.0           # >0 turns clients into FedProx clients
    device: str = "auto"              # auto | cuda | cpu
    eval_every: int = 1


@dataclass
class UAVConfig:
    """Section III of the draft: mobility (Eqs. 4-6), energy (Eqs. 7-10),
    channel (Eqs. 11-13). SI units unless noted."""
    enabled: bool = True
    num_uavs: int = 20
    area_m: float = 1000.0            # square operating area side
    alt_min_m: float = 50.0
    alt_max_m: float = 150.0
    v_max: float = 20.0               # Eq. (5) max speed [m/s]
    a_max: float = 5.0                # Eq. (6) max acceleration [m/s^2]
    gm_alpha: float = 0.85            # Gauss-Markov memory (0=random walk, 1=constant velocity)
    round_duration_s: float = 10.0    # wall-clock advance per FL round
    # heterogeneous mobility: fraction of "high-mobility" UAVs and their speed multiplier
    high_mobility_fraction: float = 0.0
    high_mobility_speed_factor: float = 2.5

    # Energy model, Eqs. (7)-(10)
    battery_j: float = 5.0e5          # initial energy (varied per-UAV +-20%)
    kappa: float = 1e-27              # effective switching capacitance, Eq. (8)
    p0_hover_w: float = 80.0          # Eq. (10)
    p1_speed_w: float = 0.15          # Eq. (10) velocity-dependent coefficient
    # Compute model
    f_min_hz: float = 5.0e8
    f_max_hz: float = 2.0e9
    cycles_per_sample: float = 2.0e6  # tau_j: CPU cycles to process one sample

    # Channel model, Eqs. (11)-(13)
    server_pos: tuple = (500.0, 500.0, 0.0)
    bandwidth_hz: float = 5.0e6       # B in Eq. (12); calibrated so the
                                      # MNIST CNN (~53 Mbit) delivers ~85-95%
                                      # under Rayleigh fading at the 5 s
                                      # deadline. Set 1.0e6 for the harsh-
                                      # comm experimental condition.
    beta0_db: float = -50.0           # channel gain at 1 m
    pathloss_alpha: float = 2.2       # alpha in Eq. (11)
    noise_dbm_per_hz: float = -174.0  # N0
    p_tx_min_w: float = 0.05
    p_tx_max_w: float = 1.0
    bits_per_param: float = 32.0      # Eq. (13)
    comm_deadline_s: float = 5.0      # T^{comm,max}: absolute deadline (the
                                      # validated default for MNIST/FEMNIST)
    comm_slack: float = 0.0           # if > 0, comm deadline = comm_slack x the
                                      # transmission time of THIS model at a
                                      # median channel (scales with model size).
                                      # DEFAULT 0 = use the fixed comm_deadline_s
                                      # (preserves all banked MNIST/FEMNIST
                                      # results). Large models (ResNet-18, 358
                                      # Mbit) set comm_slack>0 so the fixed 5s
                                      # does not drop 100% of updates.
    e_min_j: float = 1.0e4            # Eq. (29) energy floor for participation
    idle_energy_drain: bool = True    # non-participants still burn P_mob*dt
                                      # (set false to reproduce pre-fix runs)


@dataclass
class TrustConfig:
    """Section III-E / Algorithm 1. Weight mechanism supports fixed and
    learnable modes from day one (mandatory modernization #1)."""
    alpha_beh: float = 0.7            # EWMA memories, Eqs. (17)(21)(23)
    alpha_comp: float = 0.7
    alpha_comm: float = 0.7
    # phi^beh sub-weights w1,w2,w3 in Algorithm 1 line 6
    w_participation: float = 0.2
    w_commitment: float = 0.2
    w_gradient: float = 0.6   # security-critical term dominates by default
    # Overall combination, Eq. (25)
    weight_mode: str = "fixed"        # fixed | learnable
    w_beh: float = 0.34
    w_comp: float = 0.33
    w_comm: float = 0.33
    learnable_lr: float = 0.5         # exponentiated-gradient step for learnable mode
    asymmetry: float = 3.0            # bad news incorporated this-many-times
                                      # faster than good; 1.0 = draft's
                                      # symmetric EWMA (adaptive-attack ablation)
                                      # faster than good news (trust is slow to
                                      # earn, fast to lose); 1.0 = symmetric draft
    init_trust: float = 0.5
    t_min: float = 0.3                # Eq. (28) trust threshold
    aggregation_gate: bool = True     # omega ~ ReLU(T - t_min)*n (ours);
                                      # False = draft's linear Eq. (33)
    round_consistency_gate: bool = True  # multiply omega by current-round
                                         # consistency score (anti-adaptive)
    k_max: int = 10                   # Eq. (27) max participants per round
    staleness_beta: float = 0.0       # Eq. (31); draft's 0.1 PENALIZES
                                      # benched UAVs and causes selection
                                      # lock-in under adversarial
                                      # non-delivery -- kept as ablation
    explore_slots: int = 1            # per round, reserve this many of the
                                      # K slots for the longest-benched
                                      # eligible UAVs (guaranteed re-probe)


@dataclass
class ResourceConfig:
    """Section IV-C / Algorithm 3 (dual decomposition)."""
    enabled: bool = True
    lambda1: float = 1.0              # compute-time weight, Eq. (34)
    lambda2: float = 1.0              # comm-time weight
    lambda3: float = 1.0e-4           # energy weight
    tau_max_s: float = 8.0            # Eq. (35) compute deadline
    t_max_s: float = 5.0              # Eq. (36) comm deadline
    dual_step: float = 0.1
    max_iters: int = 50
    tol: float = 1e-3
    compute_deadline: bool = False    # if True, updates whose ACTUAL compute
                                      # time exceeds the round compute budget
                                      # are dropped (converts resource-
                                      # misreport into real data loss)
    compute_slack: float = 1.5        # deadline = compute_slack x the compute
                                      # time an AVERAGE-speed UAV needs for its
                                      # workload. Dataset-independent (scales
                                      # with data volume); ~78% of honest UAVs
                                      # make it at 1.5. Replaces the old fixed
                                      # tau_max_s wall-clock (which broke across
                                      # datasets with different data volumes).


@dataclass
class AttackConfig:
    """Gradient-space, system-dimension, and adaptive attacks."""
    attack_type: str = "none"
    # gradient-space: label_flip | sign_flip | grad_scale | random_update
    # system-dimension: resource_misreport | strategic_delay | selective_participation
    # adaptive: adaptive_trust
    malicious_fraction: float = 0.0
    # gradient attack knobs
    label_flip_prob: float = 1.0
    grad_scale_factor: float = 5.0
    random_update_std: float = 1.0
    # system attack knobs
    misreport_factor: float = 2.0     # claim f is misreport_factor x actual
    delay_slack_s: float = 0.5        # deliver just inside/outside deadline
    selective_prob: float = 0.4       # probability of showing up when invited
    # adaptive attack knobs
    adaptive_trust_target: float = 0.6  # build trust to here, then attack
    adaptive_probe_rounds: int = 5
    # adaptive_evasive knobs (the gradient-evasive, trust-farming adversary)
    evasive_align: float = 0.9        # cosine alignment kept with benign dir
                                      # (1.0 = perfectly benign-looking)
    evasive_poison_scale: float = 1.0 # magnitude of the orthogonal poison
                                      # component, relative to benign norm


@dataclass
class MethodConfig:
    """Which server-side method runs. 'trustfl' is ours; the rest are baselines."""
    name: str = "fedavg"
    # fedavg | trustfl | krum | multikrum | trimmed_mean | median | fltrust | fedgreed
    krum_f: int = 0                   # assumed Byzantine count for (Multi-)Krum
    multikrum_m: int = 5
    trimmed_mean_beta: float = 0.1    # fraction trimmed each side
    fedgreed_k: int = 5               # subset size selected by FedGreed


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    seed: int = 42
    seeds: Optional[list] = None      # multi-seed sweep: overrides `seed`
    out_dir: str = "./results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    method: MethodConfig = field(default_factory=MethodConfig)

    # ---------- IO ----------
    @staticmethod
    def load(path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return ExperimentConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict) -> "ExperimentConfig":
        cfg = ExperimentConfig()
        for key, val in raw.items():
            if not hasattr(cfg, key):
                raise KeyError(f"Unknown config section/field: {key}")
            cur = getattr(cfg, key)
            if dataclasses.is_dataclass(cur) and isinstance(val, dict):
                for k, v in val.items():
                    if not hasattr(cur, k):
                        raise KeyError(f"Unknown field {key}.{k}")
                    setattr(cur, k, _coerce(getattr(cur, k), v))
            else:
                setattr(cfg, key, _coerce(cur, val))
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def with_seed(self, seed: int) -> "ExperimentConfig":
        c = copy.deepcopy(self)
        c.seed = seed
        return c

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
