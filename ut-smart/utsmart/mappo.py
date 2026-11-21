"""Constrained MAPPO: shared actor, centralized critic, Lagrangian dual ascent."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from .config import VARIANTS, SimConfig
from .env import CRITIC_DIM, N_ACTIONS, UnderwaterEnv
from .nn import MLP, Adam
from .policies import ActionScorer


@dataclass
class PPOConfig:
    episodes: int = 600
    episodes_per_update: int = 2
    epochs: int = 6
    minibatch: int = 128
    lr: float = 1e-3
    clip: float = 0.2
    entropy: float = 0.005
    dual_lr: float = 0.5
    max_grad_norm: float = 1.0
    lambda_max: float = 100.0


# domain randomization (Sec. III-D)
RAND_M = (50, 100, 150, 200)
RAND_N = (3, 5, 7, 10)
RAND_MAL = (0.0, 0.1, 0.2, 0.3)
RAND_SIGMA = (2.0, 4.0, 6.0, 8.0)


def random_scenario(base: SimConfig, rng: np.random.Generator) -> SimConfig:
    return base.with_(n_sensors=int(rng.choice(RAND_M)), n_auvs=int(rng.choice(RAND_N)),
                      malicious_frac=float(rng.choice(RAND_MAL)),
                      shadow_sigma=float(rng.choice(RAND_SIGMA)),
                      gateway_compute=bool(rng.random() < 0.5))


class ConstrainedMAPPO:
    def __init__(self, variant_name: str, seed: int, ppo: PPOConfig | None = None,
                 base_cfg: SimConfig | None = None):
        self.variant = VARIANTS[variant_name]
        assert self.variant.policy == "learned"
        self.seed = seed
        self.ppo = ppo or PPOConfig()
        self.base = base_cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.actor = ActionScorer(np.random.default_rng(seed + 1), sample=True)
        self.critic = MLP([CRITIC_DIM, 64, 64, 1], np.random.default_rng(seed + 2), out_gain=1.0)
        self.opt_a = Adam(self.actor.net, self.ppo.lr, max_grad_norm=self.ppo.max_grad_norm)
        self.opt_c = Adam(self.critic, self.ppo.lr, max_grad_norm=self.ppo.max_grad_norm)
        self.lam_d, self.lam_a = 0.0, 0.0

    # ----------------------------------------------------------- rollout
    def rollout(self, episode: int):
        cfg = random_scenario(self.base, self.rng)
        env_seed = 1_000_000 * (self.seed + 1) + episode
        env = UnderwaterEnv(cfg, self.variant, self.actor, env_seed, record=True)
        summary = env.run()
        recs = [r for r in env.decisions if r["r"] is not None]
        return summary, recs

    # ------------------------------------------------------------ update
    def update(self, recs: list[dict]) -> dict:
        p = self.ppo
        if len(recs) < 8:
            return {}
        G = np.stack([r["g"] for r in recs])
        PHI = np.stack([r["phi"] for r in recs])
        MASK = np.stack([r["mask"] for r in recs])
        A = np.array([r["a"] for r in recs])
        LOGP = np.array([r["logp"] for r in recs])
        CIN = np.stack([r["critic"] for r in recs])
        R = np.array([r["r"] for r in recs])
        MISS = np.array([r["miss"] for r in recs])
        ACC = np.array([r["accfail"] for r in recs])
        R_shaped = R - self.lam_d * MISS - self.lam_a * ACC

        V_old = self.critic.forward(CIN)[0][:, 0]
        adv = R_shaped - V_old
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(recs)
        idx = np.arange(n)
        clip_frac, ent_sum, nb = 0.0, 0.0, 0
        for _ in range(p.epochs):
            self.rng.shuffle(idx)
            for s in range(0, n, p.minibatch):
                b = idx[s:s + p.minibatch]
                B = len(b)
                # ---- actor
                logits, hs = self.actor.logits(G[b], PHI[b], MASK[b])
                z = logits - logits.max(1, keepdims=True)
                ez = np.exp(z) * MASK[b]
                prob = ez / ez.sum(1, keepdims=True)
                logp_all = np.log(prob + 1e-12)
                logp = logp_all[np.arange(B), A[b]]
                ratio = np.exp(logp - LOGP[b])
                adv_b = adv[b]
                unclipped = ratio * adv_b
                clipped = np.clip(ratio, 1 - p.clip, 1 + p.clip) * adv_b
                active = unclipped <= clipped          # gradient flows through unclipped term
                clip_frac += float((~active).mean())
                H = -(prob * logp_all * MASK[b]).sum(1)
                ent_sum += float(H.mean())
                onehot = np.zeros_like(prob)
                onehot[np.arange(B), A[b]] = 1.0
                # loss = -mean(min(...)) - ent * mean(H)
                dlogp = -(ratio * adv_b * active) / B                     # dL/dlogp_a
                dlogits = dlogp[:, None] * (onehot - prob)
                dH = -prob * (logp_all + H[:, None])                      # dH/dlogits
                dlogits += -(p.entropy / B) * dH
                dlogits *= MASK[b]
                grads = self.actor.net.backward(hs, dlogits.reshape(-1, 1))
                self.opt_a.step(grads)
                # ---- critic
                v, hc = self.critic.forward(CIN[b])
                dv = (v[:, 0] - R_shaped[b])[:, None] / B
                self.opt_c.step(self.critic.backward(hc, dv))
                nb += 1

        # ---- projected dual ascent
        p_miss, p_acc = float(MISS.mean()), float(ACC.mean())
        self.lam_d = float(np.clip(self.lam_d + p.dual_lr * (p_miss - self.base.eps_d), 0, p.lambda_max))
        self.lam_a = float(np.clip(self.lam_a + p.dual_lr * (p_acc - self.base.eps_a), 0, p.lambda_max))
        return dict(clip_frac=clip_frac / max(nb, 1), entropy=ent_sum / max(nb, 1),
                    p_miss=p_miss, p_acc=p_acc)

    # ------------------------------------------------------------- train
    def train(self, log_path: str | None = None, verbose: bool = True) -> list[dict]:
        p = self.ppo
        history, buf = [], []
        writer, fh = None, None
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            fh = open(log_path, "w", newline="")
        t0 = time.time()
        for ep in range(p.episodes):
            summary, recs = self.rollout(ep)
            buf.extend(recs)
            info = {}
            if (ep + 1) % p.episodes_per_update == 0:
                info = self.update(buf)
                buf = []
            row = dict(episode=ep, success_rate=summary["success_rate"], miss_rate=summary["miss_rate"],
                       acc_fail_rate=summary["acc_fail_rate"], corrupt_rate=summary["corrupt_rate"],
                       reward_per_task=summary["reward_per_task"], lambda_d=self.lam_d,
                       lambda_a=self.lam_a, tasks=summary["tasks"])
            history.append(row)
            if fh:
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
            if verbose and (ep + 1) % 10 == 0:
                recent = np.mean([h["success_rate"] for h in history[-20:]])
                print(f"[{self.variant.name} s{self.seed}] ep {ep + 1:4d}/{p.episodes} "
                      f"succ(ma20)={recent:.3f} lamD={self.lam_d:.2f} lamA={self.lam_a:.2f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        if fh:
            fh.close()
        return history

    # ------------------------------------------------------------ io
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = dict(variant=self.variant.name, seed=self.seed, lam_d=self.lam_d, lam_a=self.lam_a,
                    ppo=asdict(self.ppo))
        np.savez(path, meta=json.dumps(meta), **self.actor.net.state("actor_"),
                 **self.critic.state("critic_"))


def load_actor(path: str, rng_seed: int = 0) -> tuple[ActionScorer, dict]:
    d = np.load(path, allow_pickle=False)
    actor = ActionScorer(np.random.default_rng(rng_seed), sample=False)
    actor.net.load(d, "actor_")
    return actor, json.loads(str(d["meta"]))
