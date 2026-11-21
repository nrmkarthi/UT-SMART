#!/usr/bin/env python3
"""Train learned variants (UT-SMART, MAPPO, ablations) with constrained MAPPO.

Examples
--------
python scripts/train.py --variants utsmart --seeds 0               # one run
python scripts/train.py --variants all --seeds 0 1 2 --workers 4   # paper setup
python scripts/train.py --variants utsmart mappo --episodes 100    # quick
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utsmart.config import LEARNED  # noqa: E402
from utsmart.mappo import ConstrainedMAPPO, PPOConfig  # noqa: E402


def _job(args):
    variant, seed, episodes, out_models, out_logs = args
    trainer = ConstrainedMAPPO(variant, seed, PPOConfig(episodes=episodes))
    trainer.train(log_path=os.path.join(out_logs, f"train_{variant}_s{seed}.csv"), verbose=True)
    path = os.path.join(out_models, f"{variant}_s{seed}.npz")
    trainer.save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["all"], help=f"any of {LEARNED} or 'all'")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--models", default="models")
    ap.add_argument("--logs", default="results/train")
    a = ap.parse_args()
    variants = LEARNED if a.variants == ["all"] else a.variants
    for v in variants:
        if v not in LEARNED:
            ap.error(f"unknown learned variant {v!r}; choose from {LEARNED}")
    os.makedirs(a.models, exist_ok=True)
    os.makedirs(a.logs, exist_ok=True)
    jobs = [(v, s, a.episodes, a.models, a.logs) for v in variants for s in a.seeds]
    if a.workers <= 1 or len(jobs) == 1:
        paths = [_job(j) for j in jobs]
    else:
        with Pool(min(a.workers, len(jobs))) as pool:
            paths = pool.map(_job, jobs, chunksize=1)
    print("saved:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
