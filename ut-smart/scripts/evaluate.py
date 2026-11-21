#!/usr/bin/env python3
"""Run the evaluation sweeps (10 seeds each) and write per-seed CSVs to results/.

python scripts/evaluate.py                       # all sweeps
python scripts/evaluate.py --sweeps ablation attack --seeds 10
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utsmart.experiments import run_sweep, sweeps  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="+", default=list(sweeps()))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in a.sweeps:
        t = time.time()
        rows = run_sweep(name, a.seeds, a.models, a.workers)
        path = os.path.join(a.out, f"{name}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{name:9s} {len(rows):4d} runs  {time.time() - t:6.1f}s  -> {path}", flush=True)


if __name__ == "__main__":
    main()
