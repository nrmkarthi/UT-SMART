"""Finite-difference check of the manual MLP backward pass."""
import numpy as np

from utsmart.nn import MLP


def test_mlp_gradient():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 1], rng, out_gain=1.0)
    x = rng.normal(size=(4, 5))
    out, hs = net.forward(x)
    grads = net.backward(hs, np.ones_like(out))      # d(sum out)/d params
    W = net.W[0]
    i, j, eps = 2, 3, 1e-6
    W[i, j] += eps
    up = net.forward(x)[0].sum()
    W[i, j] -= 2 * eps
    dn = net.forward(x)[0].sum()
    W[i, j] += eps
    assert abs((up - dn) / (2 * eps) - grads[0][i, j]) < 1e-5
