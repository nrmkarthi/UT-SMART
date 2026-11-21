#!/usr/bin/env bash
# Full reproduction: train 7 learned variants x 3 seeds, run all sweeps x 10 seeds, make figures.
# ~30-60 min on a 4-core laptop. Pass QUICK=1 for a ~3-minute smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."
EPISODES=600; SEEDS="0 1 2"; EVAL_SEEDS=10
if [[ "${QUICK:-0}" == "1" ]]; then EPISODES=40; SEEDS="0"; EVAL_SEEDS=3; fi
WORKERS="${WORKERS:-$(python -c 'import os; print(os.cpu_count())')}"

python scripts/train.py --variants all --seeds $SEEDS --episodes "$EPISODES" --workers "$WORKERS"
if [[ "${QUICK:-0}" == "1" ]]; then          # reuse seed-0 models for every eval seed
  for f in models/*_s0.npz; do cp "$f" "${f/_s0/_s1}"; cp "$f" "${f/_s0/_s2}"; done
fi
python scripts/evaluate.py --seeds "$EVAL_SEEDS" --workers "$WORKERS"
python scripts/make_figures.py
