"""Uncertainty-aware digital twin: per-AUV Kalman posteriors over link shadowing."""
from __future__ import annotations

import numpy as np


class LinkTwin:
    """Each AUV c keeps N(m[c, l], P[c, l]) over the shadowing chi_l of every link l.

    The deterministic path loss follows from known positions, so only the
    shadowing term is uncertain. The posterior is propagated with the same AR(1)
    model as the channel and fused with SNR measurements by a scalar Kalman update.
    """

    def __init__(self, n_auvs: int, n_links: int, sigma: float, rho: float, meas_var: float):
        self.sigma2 = sigma ** 2
        self.rho = rho
        self.meas_var = meas_var
        self.m = np.zeros((n_auvs, n_links))
        self.P = np.full((n_auvs, n_links), self.sigma2)

    def propagate(self) -> None:
        self.m *= self.rho
        self.P = self.rho ** 2 * self.P + (1 - self.rho ** 2) * self.sigma2

    def update(self, c: int, link: int, z: float) -> None:
        P = self.P[c, link]
        K = P / (P + self.meas_var)
        self.m[c, link] += K * (z - self.m[c, link])
        self.P[c, link] = (1 - K) * P

    def std(self, c: int, link: int) -> float:
        return float(np.sqrt(self.P[c, link]))
