"""Decision policies: the learned action scorer and the three heuristic baselines."""
from __future__ import annotations

import numpy as np

from .env import G_DIM, N_ACTIONS, PHI_DIM
from .nn import MLP


class RawLocal:
    """Raw data to the cluster AUV, full inference there."""

    def act(self, obs):
        return 0, 0.0


class DelayEnergyBR:
    """DE-BR: best response to current congestion, min delay + energy (no accuracy/VoI)."""

    def __init__(self, energy_weight: float = 1.0):
        self.w = energy_weight

    def act(self, obs):
        p = obs["pred"]
        cost = np.where(p["mask"], p["D"] + self.w * p["E"], np.inf)
        return int(np.argmin(cost)), 0.0


class GreedySemantic:
    """Cheapest (energy) action whose predicted mean delay and accuracy meet the mission."""

    def act(self, obs):
        p, t = obs["pred"], obs["task"]
        m = p["mask"]
        acc_ok = p["A"] >= t.a_min - 1e-9
        feas = m & acc_ok & (p["D"] <= t.dmax)
        if feas.any():
            return int(np.argmin(np.where(feas, p["E"], np.inf))), 0.0
        if (m & acc_ok).any():
            return int(np.argmin(np.where(m & acc_ok, p["D"], np.inf))), 0.0
        return int(np.argmin(np.where(m, p["D"], np.inf))), 0.0


class ActionScorer:
    """Parameter-shared actor f_theta(g, phi_a) -> score; softmax over valid actions."""

    def __init__(self, rng: np.random.Generator, hidden=(64, 64), sample: bool = True):
        self.net = MLP([G_DIM + PHI_DIM, *hidden, 1], rng)
        self.rng = rng
        self.sample = sample

    @staticmethod
    def inputs(g, phi):
        """g: (B, 15), phi: (B, 12, 12) -> (B*12, 27)."""
        B = g.shape[0]
        gg = np.broadcast_to(g[:, None, :], (B, N_ACTIONS, G_DIM))
        return np.concatenate([gg, phi], axis=2).reshape(B * N_ACTIONS, G_DIM + PHI_DIM)

    def logits(self, g, phi, mask):
        out, hs = self.net.forward(self.inputs(g, phi))
        s = out.reshape(g.shape[0], N_ACTIONS)
        return np.where(mask, s, -1e9), hs

    def act(self, obs):
        s, _ = self.logits(obs["g"][None], obs["phi"][None], obs["mask"][None])
        s = s[0]
        z = s - s.max()
        p = np.exp(z)
        p /= p.sum()
        a = int(self.rng.choice(N_ACTIONS, p=p)) if self.sample else int(np.argmax(s))
        return a, float(np.log(p[a] + 1e-12))


def make_policy(variant, rng, actor: ActionScorer | None = None):
    if variant.policy == "raw_local":
        return RawLocal()
    if variant.policy == "debr":
        return DelayEnergyBR()
    if variant.policy == "greedy_sem":
        return GreedySemantic()
    if actor is None:
        raise ValueError(f"variant {variant.name} needs a trained actor")
    return actor
