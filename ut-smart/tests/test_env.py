import numpy as np
import pytest

from utsmart import VARIANTS, SimConfig, UnderwaterEnv
from utsmart.mappo import ConstrainedMAPPO, PPOConfig, load_actor
from utsmart.policies import ActionScorer, make_policy

SMALL = SimConfig(n_sensors=40, n_auvs=3, n_slots=150)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_episode_runs_and_metrics_valid(name):
    v = VARIANTS[name]
    pol = make_policy(v, np.random.default_rng(0), ActionScorer(np.random.default_rng(0), sample=False))
    s = UnderwaterEnv(SMALL, v, pol, seed=1).run()
    assert s["tasks"] > 0
    for k in ("success_rate", "miss_rate", "acc_fail_rate", "corrupt_rate"):
        assert 0.0 <= s[k] <= 1.0
    assert s["energy_per_task"] > 0 and s["traffic_kbit_per_task"] > 0


def test_deterministic_given_seed():
    v = VARIANTS["greedy_sem"]
    a = UnderwaterEnv(SMALL, v, make_policy(v, None), seed=3).run()
    b = UnderwaterEnv(SMALL, v, make_policy(v, None), seed=3).run()
    assert a == b


def test_semantic_beats_raw():
    cfg = SimConfig()
    raw = UnderwaterEnv(cfg, VARIANTS["raw_local"], make_policy(VARIANTS["raw_local"], None), seed=5).run()
    sem = UnderwaterEnv(cfg, VARIANTS["greedy_sem"], make_policy(VARIANTS["greedy_sem"], None), seed=5).run()
    assert sem["success_rate"] > raw["success_rate"] + 0.2


def test_ondemand_sync_cheaper_than_periodic():
    from dataclasses import replace
    cfg = SimConfig()
    pol = make_policy(VARIANTS["greedy_sem"], None)
    od = UnderwaterEnv(cfg, replace(VARIANTS["greedy_sem"], sync="ondemand", share="event"), pol, seed=2).run()
    pe = UnderwaterEnv(cfg, VARIANTS["greedy_sem"], pol, seed=2).run()
    assert od["control_kbit_per_task"] < pe["control_kbit_per_task"] / 3


def test_auv_only_regime_never_uses_gateway():
    cfg = SimConfig(gateway_compute=False, n_slots=200)
    v = VARIANTS["utsmart"]
    s = UnderwaterEnv(cfg, v, ActionScorer(np.random.default_rng(0)), seed=4).run()
    assert s["frac_gw"] == 0.0


def test_short_training_and_checkpoint_roundtrip(tmp_path):
    tr = ConstrainedMAPPO("utsmart", 0, PPOConfig(episodes=4), base_cfg=SMALL)
    hist = tr.train(verbose=False)
    assert len(hist) == 4
    path = tmp_path / "m.npz"
    tr.save(str(path))
    actor, meta = load_actor(str(path))
    assert meta["variant"] == "utsmart"
    assert actor.net.n_params() == tr.actor.net.n_params()
    for W1, W2 in zip(actor.net.params, tr.actor.net.params):
        np.testing.assert_allclose(W1, W2)
