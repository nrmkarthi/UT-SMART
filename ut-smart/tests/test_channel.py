import math

import numpy as np

from utsmart import channel as ch


def test_thorp_at_12khz():
    assert abs(ch.thorp_absorption(12.0) - 1.645) < 0.01


def test_path_loss_monotone():
    d = np.array([100.0, 500.0, 1000.0, 2000.0])
    assert np.all(np.diff(ch.path_loss_db(d, 12.0)) > 0)


def test_noise_reasonable():
    n = ch.urick_noise_db(12.0)
    assert 40 < n < 50          # wind-dominated at 12 kHz, w=5 m/s


def test_goodput_zero_below_threshold():
    assert ch.goodput(np.array([2.9]), 4000, 0.4)[0] == 0.0
    assert ch.goodput(np.array([20.0]), 4000, 0.4)[0] > 5000
    assert math.isclose(ch.goodput_scalar(15.0, 4000, 0.4), float(ch.goodput(np.array([15.0]), 4000, 0.4)[0]))


def test_shadowing_stationary_std():
    p = ch.ShadowingProcess(20000, 4.0, 0.97, 0.004, 0.05, np.random.default_rng(0))
    for _ in range(50):
        p.step()
    assert abs(p.chi.std() - 4.0) < 0.15
    assert 0.0 < p.outage.mean() < 0.2
