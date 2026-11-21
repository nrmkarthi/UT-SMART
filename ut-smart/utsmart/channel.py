"""Underwater acoustic channel: Thorp absorption, Urick noise, shadowing, outages."""
from __future__ import annotations

import math

import numpy as np


def thorp_absorption(f_khz: float) -> float:
    """Thorp's absorption coefficient alpha(f) in dB/km (f in kHz)."""
    f2 = f_khz ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def path_loss_db(d, f_khz: float, k: float = 1.5):
    """A(d, f) = 10 k log10(d) + d/1000 * alpha(f); d in metres."""
    d = np.maximum(np.asarray(d, dtype=float), 1.0)
    return 10 * k * np.log10(d) + d / 1000.0 * thorp_absorption(f_khz)


def urick_noise_db(f_khz: float, shipping: float = 0.5, wind: float = 5.0) -> float:
    """Ambient noise PSD N(f) in dB re uPa/Hz: turbulence + shipping + wind + thermal."""
    lf = math.log10(f_khz)
    nt = 17 - 30 * lf
    ns = 40 + 20 * (shipping - 0.5) + 26 * lf - 60 * math.log10(f_khz + 0.03)
    nw = 50 + 7.5 * math.sqrt(wind) + 20 * lf - 40 * math.log10(f_khz + 0.4)
    nth = -15 + 20 * lf
    return 10 * math.log10(sum(10 ** (x / 10) for x in (nt, ns, nw, nth)))


def per(snr_db):
    """Packet error rate PER(gamma) = 1 / (1 + exp((gamma - 8)/1.5))."""
    return 1.0 / (1.0 + np.exp((np.asarray(snr_db, dtype=float) - 8.0) / 1.5))


def raw_rate(snr_db, bandwidth: float, eta: float):
    """Physical-layer bit rate eta B log2(1 + SNR) (bits on air per second)."""
    snr_lin = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    return eta * bandwidth * np.log2(1 + snr_lin)


def goodput(snr_db, bandwidth: float, eta: float, min_snr: float = 3.0):
    """Effective ARQ goodput R = eta B log2(1+SNR)(1-PER); zero below min_snr."""
    snr_db = np.asarray(snr_db, dtype=float)
    r = raw_rate(snr_db, bandwidth, eta) * (1 - per(snr_db))
    return np.where(snr_db < min_snr, 0.0, r)


def goodput_scalar(snr_db: float, bandwidth: float, eta: float, min_snr: float = 3.0) -> float:
    if snr_db < min_snr:
        return 0.0
    snr_lin = 10 ** (snr_db / 10)
    return eta * bandwidth * math.log2(1 + snr_lin) * (1 - 1 / (1 + math.exp((snr_db - 8) / 1.5)))


class ShadowingProcess:
    """AR(1) log-normal shadowing plus two-state Markov outage, one entry per link."""

    def __init__(self, n_links: int, sigma: float, rho: float, p_off: float, p_on: float,
                 rng: np.random.Generator):
        self.sigma, self.rho, self.p_off, self.p_on = sigma, rho, p_off, p_on
        self.rng = rng
        self.chi = rng.normal(0.0, sigma, n_links)
        self.outage = np.zeros(n_links, dtype=bool)

    def step(self) -> None:
        n = self.chi.shape[0]
        self.chi = self.rho * self.chi + math.sqrt(1 - self.rho ** 2) * self.sigma * self.rng.normal(size=n)
        u = self.rng.random(n)
        go_off = (~self.outage) & (u < self.p_off)
        go_on = self.outage & (u < self.p_on)
        self.outage = (self.outage | go_off) & ~go_on
