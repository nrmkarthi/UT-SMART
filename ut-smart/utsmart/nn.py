"""Minimal NumPy MLP with manual backprop and Adam (no deep-learning framework needed)."""
from __future__ import annotations

import numpy as np


class MLP:
    """tanh hidden layers, linear output."""

    def __init__(self, sizes, rng: np.random.Generator, out_gain: float = 0.01):
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2.0)
            q, _ = np.linalg.qr(rng.normal(size=(max(a, b), min(a, b))))
            W = q[:a, :b] if a >= b else q[:b, :a].T
            self.W.append(gain * W)
            self.b.append(np.zeros(b))

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x):
        hs = [x]
        h = x
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < last else z
            hs.append(h)
        return h, hs

    def backward(self, hs, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - hs[i] ** 2)
        return gW + gb

    def n_params(self) -> int:
        return sum(p.size for p in self.params)

    def state(self, prefix: str) -> dict:
        return {f"{prefix}{i}": p for i, p in enumerate(self.params)}

    def load(self, d: dict, prefix: str) -> None:
        n = len(self.W)
        for i in range(n):
            self.W[i] = np.array(d[f"{prefix}{i}"])
            self.b[i] = np.array(d[f"{prefix}{i + n}"])


class Adam:
    def __init__(self, model: MLP, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 max_grad_norm: float = 1.0):
        self.model, self.lr, self.b1, self.b2, self.eps = model, lr, betas[0], betas[1], eps
        self.max_norm = max_grad_norm
        self.m = [np.zeros_like(p) for p in model.params]
        self.v = [np.zeros_like(p) for p in model.params]
        self.t = 0

    def step(self, grads) -> float:
        norm = float(np.sqrt(sum((g ** 2).sum() for g in grads)))
        scale = min(1.0, self.max_norm / (norm + 1e-12))
        self.t += 1
        params = self.model.params
        for i, g in enumerate(grads):
            g = g * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            params[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        n = len(self.model.W)
        self.model.W, self.model.b = params[:n], params[n:]
        return norm
