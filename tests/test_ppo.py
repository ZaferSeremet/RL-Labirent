"""PPO ajanı: zamanla azaltma ve eski ayar adı."""

from __future__ import annotations

import pytest

from agents.base import EnvSpec
from agents.ppo_ajan import PPOAjan
from maze_env import EnvConfig, MazeEnv


@pytest.fixture
def spec():
    return EnvSpec.from_env(MazeEnv(EnvConfig(obs_mode="full")))


def test_decay_linear_then_constant(spec):
    ag = PPOAjan(spec, seed=0, lr=1e-3, lr_end=1e-4, ent_coef=0.05, ent_coef_end=0.01, decay_episodes=100)
    assert ag._schedule() == pytest.approx((1e-3, 0.05))
    ag.episodes_done = 50  # yarı yol
    assert ag._schedule() == pytest.approx((5.5e-4, 0.03))
    for done in (100, 5000):  # bitti -> sabit
        ag.episodes_done = done
        assert ag._schedule() == pytest.approx((1e-4, 0.01))


def test_no_decay_by_default(spec):
    ag = PPOAjan(spec, seed=0, lr=2e-3, ent_coef=0.07)
    ag.episodes_done = 10**6
    assert ag._schedule() == pytest.approx((2e-3, 0.07))


def test_old_hafiza_name_still_works(spec):
    ag = PPOAjan(spec, seed=0, hafiza=3)
    assert ag.memory == 3
    assert ag.actor.params[0].shape[0] == 3 * spec.observation_space.shape[0]
    assert "hafiza" not in ag.hyperparameters()
