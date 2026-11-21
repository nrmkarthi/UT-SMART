import numpy as np

from utsmart.trust import RESULT, TrustManager
from utsmart.twin import LinkTwin


def test_kalman_reduces_variance_and_tracks():
    tw = LinkTwin(1, 1, sigma=4.0, rho=0.97, meas_var=1.0)
    assert np.isclose(tw.P[0, 0], 16.0)
    tw.update(0, 0, 3.0)
    assert tw.P[0, 0] < 1.0 and 2.5 < tw.m[0, 0] < 3.0
    for _ in range(200):
        tw.propagate()
    assert abs(tw.P[0, 0] - 16.0) < 0.1 and abs(tw.m[0, 0]) < 0.1


def test_trust_drops_for_corrupt_results():
    tm = TrustManager(2)
    for _ in range(10):
        tm.observe(RESULT, 0, True)
        tm.observe(RESULT, 1, False)
    T = tm.scores()
    assert T[0] > 0.7 > 0.55 > T[1]
