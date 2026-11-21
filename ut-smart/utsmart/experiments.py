"""Evaluation sweeps of Sec. IV (Figs. 3-7, Table IV)."""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np

from .config import VARIANTS, SimConfig
from .env import UnderwaterEnv
from .policies import make_policy

EVAL_SEED_BASE = 10_000      # disjoint from training seeds
N_TRAIN_SEEDS = 3

MAIN = ["utsmart", "mappo", "greedy_sem", "raw_local", "debr"]
ABLATION = ["utsmart", "wo_uncertainty", "wo_ondemand", "wo_trust", "wo_event", "wo_semantic", "mappo"]


def sweeps() -> dict[str, list[dict]]:
    """Each sweep is a list of grid points: {'methods', 'param', 'value', 'cfg' overrides}."""
    gw = dict(gateway_compute=True)
    edge = dict(gateway_compute=False)
    s = {}
    s["sensors"] = [dict(param="n_sensors", value=m, cfg=dict(n_sensors=m, **gw), methods=MAIN)
                    for m in (50, 100, 150, 200)]
    s["auvs"] = [dict(param="n_auvs", value=n, cfg=dict(n_auvs=n, **gw), methods=MAIN)
                 for n in (3, 5, 7, 10)]
    s["channel"] = [dict(param="shadow_sigma", value=x, cfg=dict(shadow_sigma=x, **gw), methods=MAIN)
                    for x in (2.0, 4.0, 6.0, 8.0)]
    s["attack"] = []
    for regime, ov in (("gateway", gw), ("auv_only", edge)):
        for f in (0.0, 0.1, 0.2, 0.3):
            s["attack"].append(dict(param="malicious_frac", value=f, regime=regime,
                                    cfg=dict(n_auvs=10, malicious_frac=f, **ov),
                                    methods=MAIN + ["wo_trust"]))
    s["ablation"] = [dict(param="regime", value=r, regime=r, cfg=ov, methods=ABLATION)
                     for r, ov in (("gateway", gw), ("auv_only", edge))]
    s["sync"] = ([dict(param="sync_period", value=p, mode="periodic",
                       cfg=dict(sync_period=float(p), **gw), methods=["utsmart"],
                       variant_override=dict(sync="periodic")) for p in (20, 40, 60, 120, 240)]
                 + [dict(param="sync_threshold", value=x, mode="ondemand",
                         cfg=dict(sync_threshold=x, **gw), methods=["utsmart"])
                    for x in (1.5, 2.0, 2.5, 3.0, 4.0)])
    return s


def _model_path(models_dir: str, variant: str, eval_seed: int) -> str:
    return os.path.join(models_dir, f"{variant}_s{eval_seed % N_TRAIN_SEEDS}.npz")


def run_point(args) -> dict:
    method, cfg_over, seed, models_dir, var_over = args
    from dataclasses import replace

    from .mappo import load_actor
    v = VARIANTS[method]
    if var_over:
        v = replace(v, **var_over)
    actor = None
    if v.policy == "learned":
        path = _model_path(models_dir, method, seed)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run scripts/train.py first")
        actor, _ = load_actor(path)
    policy = make_policy(v, np.random.default_rng(seed), actor)
    cfg = SimConfig().with_(**cfg_over)
    return UnderwaterEnv(cfg, v, policy, EVAL_SEED_BASE + seed).run()


def run_sweep(name: str, seeds: int, models_dir: str, workers: int) -> list[dict]:
    points = sweeps()[name]
    jobs, meta = [], []
    for pt in points:
        for method in pt["methods"]:
            for s in range(seeds):
                jobs.append((method, pt["cfg"], s, models_dir, pt.get("variant_override")))
                meta.append(dict(sweep=name, method=method, param=pt["param"], value=pt["value"],
                                 regime=pt.get("regime", "gateway" if pt["cfg"].get("gateway_compute", True)
                                               else "auv_only"),
                                 mode=pt.get("mode", ""), seed=s))
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(run_point, jobs, chunksize=4)
    else:
        res = [run_point(j) for j in jobs]
    rows = []
    for m, r in zip(meta, res):
        rows.append({**m, **{k: float(v) for k, v in r.items()}})
    return rows
