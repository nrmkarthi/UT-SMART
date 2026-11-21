"""Beta-reputation trust kept at the gateway (Josang & Ismail, 2002)."""
from __future__ import annotations

import numpy as np

RELIABILITY, CONSISTENCY, RESULT = 0, 1, 2


class TrustManager:
    """Three Beta counters per AUV; T_i = w_R R_i + w_B B_i + w_S S_i."""

    def __init__(self, n_auvs: int, prior=(2.0, 1.0), weights=(0.3, 0.3, 0.4)):
        self.alpha = np.full((3, n_auvs), float(prior[0]))
        self.beta = np.full((3, n_auvs), float(prior[1]))
        self.w = np.asarray(weights, dtype=float)

    def observe(self, kind: int, auv: int, good: bool) -> None:
        if good:
            self.alpha[kind, auv] += 1
        else:
            self.beta[kind, auv] += 1

    def components(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def scores(self) -> np.ndarray:
        return self.w @ self.components()
