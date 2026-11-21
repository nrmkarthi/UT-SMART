"""Simulation parameters (Table III of the paper) and method variants."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Mission:
    name: str
    share: float          # fraction of generated tasks
    deadline: float       # D^max (s)
    voi: float            # value of information
    a_min: float          # minimum inference accuracy
    size_lo: float        # raw size range (bits)
    size_hi: float
    sens: float           # accuracy sensitivity s_m to compression


MISSIONS = (
    Mission("disaster", 0.25, 90.0, 3.0, 0.80, 150e3, 300e3, 0.8),
    Mission("inspection", 0.40, 240.0, 2.0, 0.88, 300e3, 600e3, 1.3),
    Mission("pollution", 0.35, 480.0, 1.0, 0.85, 200e3, 500e3, 1.0),
)


@dataclass
class SimConfig:
    # --- geometry / mobility -------------------------------------------
    area: float = 2000.0                    # L (m), square side
    sensor_depth: tuple = (250.0, 600.0)
    auv_depth: tuple = (150.0, 300.0)
    n_sensors: int = 100                    # M
    n_auvs: int = 5                         # N
    auv_speed: float = 1.5                  # m/s
    current_max: float = 0.3                # m/s
    slot: float = 2.0                       # Delta (s)
    n_slots: int = 500                      # episode length (slots)

    # --- acoustic / optical channel ------------------------------------
    freq_khz: float = 12.0
    bandwidth: float = 4000.0
    eta: float = 0.4
    spreading_k: float = 1.5
    shipping: float = 0.5
    wind: float = 5.0
    sl_sensor: float = 135.0                # dB re uPa
    sl_auv: float = 150.0
    sound_speed: float = 1500.0
    shadow_sigma: float = 4.0               # dB
    shadow_rho: float = 0.97                # per-slot correlation
    p_off: float = 0.004
    p_on: float = 0.05
    min_snr: float = 3.0                    # dB, goodput is zero below
    optical_range: float = 60.0             # m
    optical_rate: float = 1e6               # b/s

    # --- tasks ---------------------------------------------------------
    task_rate: float = 1.0 / 600.0          # per sensor (1/s)

    # --- semantic encoder ----------------------------------------------
    kappa: tuple = (1, 4, 16, 64)
    enc_cycles_per_bit: tuple = (0.0, 100.0, 150.0, 200.0)
    acc_drop: tuple = (0.0, 0.010, 0.035, 0.090)
    acc_auv: float = 0.90
    acc_gw: float = 0.95

    # --- computing -----------------------------------------------------
    f_sensor: float = 1e8
    f_auv: float = 4e9
    f_gw: float = 2e10
    cycles_raw: float = 2e9
    cycles_feat: float = 0.8e9
    gw_cycle_factor: float = 1.5
    result_bits: float = 2000.0

    # --- energy --------------------------------------------------------
    p_tx_sensor: float = 2.0
    p_tx_auv: float = 10.0
    e_cycle_sensor: float = 1e-9
    e_cycle_auv: float = 5e-10
    e_cycle_gw: float = 5e-10

    # --- threat model --------------------------------------------------
    malicious_frac: float = 0.2
    corrupt_prob: float = 0.7
    drop_prob: float = 0.1

    # --- operating regime ----------------------------------------------
    gateway_compute: bool = True            # False = AUV-only edge regime

    # --- digital twin --------------------------------------------------
    meas_noise_var: float = 1.0             # R_m (dB^2)
    sync_threshold: float = 2.5             # tau_s (dB)
    sync_period: float = 60.0               # T_s (s) for periodic sync
    probe_bits: float = 512.0
    lcb_z: float = 1.0

    # --- trust ---------------------------------------------------------
    spot_check_prob: float = 0.2
    spot_check_max_bits: float = 20e3
    trust_threshold: float = 0.55
    trust_weights: tuple = (0.3, 0.3, 0.4)  # reliability, consistency, result
    trust_prior: tuple = (2.0, 1.0)         # Beta prior (alpha, beta)
    consistency_slack: float = 30.0

    # --- state sharing -------------------------------------------------
    state_bits: float = 512.0
    share_delta: float = 15.0
    share_heartbeat: float = 120.0
    share_period: float = 20.0

    # --- reward / constraints ------------------------------------------
    w1: float = 0.3
    w2: float = 0.05
    e0: float = 20.0
    e_cap: float = 5.0
    w3: float = 0.03
    b0: float = 300e3
    w5: float = 0.3
    v_max: float = 3.0
    eps_d: float = 0.08
    eps_a: float = 0.02

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class Variant:
    """A method = a policy plus the mechanisms it is allowed to use."""
    name: str
    policy: str = "learned"         # learned | raw_local | debr | greedy_sem
    uncertainty: bool = True        # LCB predictions + uncertainty features
    sync: str = "ondemand"          # ondemand | periodic
    share: str = "event"            # event | periodic
    trust: bool = True              # Beta trust + spot checks + exclusion
    semantic: bool = True           # compression levels > 1x allowed


_HEUR = dict(uncertainty=False, sync="periodic", share="periodic", trust=False)

VARIANTS = {
    "utsmart": Variant("utsmart"),
    "mappo": Variant("mappo", uncertainty=False, sync="periodic", share="periodic", trust=False),
    "wo_uncertainty": Variant("wo_uncertainty", uncertainty=False, sync="periodic"),
    "wo_ondemand": Variant("wo_ondemand", sync="periodic"),
    "wo_trust": Variant("wo_trust", trust=False),
    "wo_event": Variant("wo_event", share="periodic"),
    "wo_semantic": Variant("wo_semantic", semantic=False),
    "raw_local": Variant("raw_local", policy="raw_local", semantic=False, **_HEUR),
    "debr": Variant("debr", policy="debr", **_HEUR),
    "greedy_sem": Variant("greedy_sem", policy="greedy_sem", **_HEUR),
}

LEARNED = [k for k, v in VARIANTS.items() if v.policy == "learned"]
HEURISTIC = [k for k, v in VARIANTS.items() if v.policy != "learned"]

LABELS = {
    "utsmart": "UT-SMART",
    "mappo": "MAPPO",
    "wo_uncertainty": "w/o uncertainty",
    "wo_ondemand": "w/o on-demand sync",
    "wo_trust": "w/o trust",
    "wo_event": "w/o event-triggered sharing",
    "wo_semantic": "w/o semantic computing",
    "raw_local": "Raw-Local",
    "debr": "DE-BR",
    "greedy_sem": "Greedy-Sem",
}
