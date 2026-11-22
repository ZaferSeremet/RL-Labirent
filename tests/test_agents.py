"""Faz 2: ajan arayüzü, baseline'lar, ajan bulma ve bölüm koşturucu."""

from __future__ import annotations

import pytest

from agents.base import Agent, EnvSpec
from baselines import RandomAgent, RightHandAgent
from engine import discover, env_config_for, load_shared_config, run_episode, seeds_for
from maze_env import MazeEnv


def make(cls, kind="normal", **env_overrides):
    env = MazeEnv(env_config_for(kind, cls.obs_mode, **env_overrides))
    return env, cls(EnvSpec.from_env(env), seed=0)


class SpyAgent(Agent):
    """learn / end_episode çağrılarını sayan test ajanı."""

    name = "_spy"
    obs_mode = "simple"

    def __init__(self, spec, seed=None, **hp):
        super().__init__(spec, seed, **hp)
        self.calls = {"begin": 0, "learn": 0, "end": 0, "greedy": []}

    def act(self, obs, greedy=False):
        self.calls["greedy"].append(greedy)
        return int(self.rng.integers(4))

    def begin_episode(self, obs, info):
        self.calls["begin"] += 1

    def learn(self, *args):
        self.calls["learn"] += 1

    def end_episode(self):
        self.calls["end"] += 1


def test_registry_finds_agents_without_errors():
    reg = discover()
    assert not reg.errors
    assert {"random", "sag_el", "ornek"} <= set(reg.agents)
    assert {c.name for c in reg.baselines()} == {"random", "sag_el"}
    assert "ornek" in {c.name for c in reg.learners()}
    with pytest.raises(KeyError):
        reg.get("yok_boyle_ajan")


def test_obs_mode_mismatch_raises():
    env = MazeEnv(obs_mode="simple")
    with pytest.raises(ValueError):
        RightHandAgent(EnvSpec.from_env(env))


def test_random_agent_valid_and_reproducible():
    env, a = make(RandomAgent)
    _, b = make(RandomAgent)
    obs, _ = env.reset(seed=1)
    acts_a = [a.act(obs) for _ in range(50)]
    acts_b = [b.act(obs) for _ in range(50)]
    assert acts_a == acts_b
    assert set(acts_a) <= {0, 1, 2, 3}


@pytest.mark.parametrize("kind", ["normal", "buyuk"])
def test_right_hand_solves_test_mazes_without_wall_hits(kind):
    env, agent = make(RightHandAgent, kind)
    for s in seeds_for(kind)[:20]:
        r = run_episode(env, agent, seed=s, train=False)
        assert r.success and r.wall_hits == 0
        assert r.steps >= r.shortest_path


def test_run_episode_train_vs_test_calls():
    env, agent = make(SpyAgent)
    r = run_episode(env, agent, seed=3, train=True)
    assert agent.calls["learn"] == r.steps and agent.calls["end"] == 1 and agent.calls["begin"] == 1
    assert not any(agent.calls["greedy"])

    env, agent = make(SpyAgent)
    r = run_episode(env, agent, seed=3, train=False)
    assert agent.calls["learn"] == 0 and agent.calls["end"] == 0 and agent.calls["begin"] == 1
    assert all(agent.calls["greedy"])


def test_run_episode_record_is_consistent():
    env, agent = make(RightHandAgent)
    r = run_episode(env, agent, seed=10000, train=False, record=True)
    t = r.trajectory
    assert len(t["positions"]) == r.steps + 1 == len(t["actions"]) + 1 == len(t["rewards"]) + 1
    assert t["positions"][0] == t["start"] and t["positions"][-1] == t["exit"]
    assert sum(t["rewards"]) == pytest.approx(r.total_reward)
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for (p, q), a in zip(zip(t["positions"], t["positions"][1:]), t["actions"]):
        moved = (q[0] - p[0], q[1] - p[1])
        assert moved in ((0, 0), deltas[a])


def test_same_seed_same_result():
    env, a1 = make(RightHandAgent)
    r1 = run_episode(env, a1, seed=10042, train=False)
    env, a2 = make(RightHandAgent)
    r2 = run_episode(env, a2, seed=10042, train=False)
    assert (r1.steps, r1.total_reward, r1.success) == (r2.steps, r2.total_reward, r2.success)


def test_save_load_roundtrip(tmp_path):
    env, a = make(RandomAgent)
    a.some_table = [1, 2, 3]
    a.save(tmp_path / "m.pkl")
    _, b = make(RandomAgent)
    b.load(tmp_path / "m.pkl")
    assert b.some_table == [1, 2, 3]
    obs, _ = env.reset(seed=0)
    assert [a.act(obs) for _ in range(10)] == [b.act(obs) for _ in range(10)]


def test_shared_settings():
    shared = load_shared_config()
    assert len(seeds_for("normal", shared)) == 100 and len(seeds_for("buyuk", shared)) == 100
    assert set(seeds_for("normal", shared)).isdisjoint(seeds_for("buyuk", shared))
    assert env_config_for("normal", "full", shared).maze_size == shared["test"]["normal"]["maze_size"]
    assert env_config_for("buyuk", "full", shared).maze_size == shared["test"]["buyuk"]["maze_size"]
    cfg = env_config_for("egitim", "simple", shared, reward_overrides={"step": -0.1})
    assert cfg.rewards.step == -0.1 and cfg.rewards.exit == shared["env"]["rewards"]["exit"]
    assert cfg.obs_mode == "simple"
    with pytest.raises(ValueError):
        env_config_for("bilinmeyen", "full", shared)
