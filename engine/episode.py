"""Tek bir bölümü koşturur. Eğitim, test ve arayüzdeki izleme bunu kullanır."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.base import Agent
from maze_env import MazeEnv


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    success: bool
    wall_hits: int
    visited_tiles: int
    shortest_path: int
    episode_seed: int
    trajectory: dict[str, Any] | None = None  # record=True ise izleme kaydı

    @property
    def efficiency(self) -> float | None:
        """Atılan adım / en kısa yol (yalnız başarılı bölümde; 1.0 = en kısa yol)."""
        if not self.success or self.shortest_path <= 0:
            return None
        return self.steps / self.shortest_path


def run_episode(
    env: MazeEnv,
    agent: Agent,
    *,
    seed: int | None = None,
    train: bool = True,
    record: bool = False,
) -> EpisodeResult:
    """train=True: keşifli eylem + learn + end_episode. train=False: greedy, öğrenme yok.

    record=True: arayüzde tekrar oynatmak için labirent, konumlar, eylemler ve
    ödüller kaydedilir (ziyaret bilgisi konumlardan yeniden kurulabilir).
    """
    obs, info = env.reset(seed=seed)
    agent.begin_episode(obs, info)

    traj = None
    if record:
        rs = env.get_render_state()
        traj = {
            "grid": rs["grid"],
            "start": rs["start"],
            "exit": rs["exit"],
            "view_radius": rs["view_radius"],
            "positions": [rs["agent"]],
            "actions": [],
            "rewards": [],
        }

    total = 0.0
    while True:
        action = agent.act(obs, greedy=not train)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if train:
            agent.learn(obs, action, reward, next_obs, terminated, truncated, info)
        total += reward
        if traj is not None:
            traj["positions"].append(env.get_render_state()["agent"])
            traj["actions"].append(int(action))
            traj["rewards"].append(float(reward))
        obs = next_obs
        if terminated or truncated:
            break

    if train:
        agent.end_episode()

    return EpisodeResult(
        total_reward=total,
        steps=info["steps"],
        success=info["success"],
        wall_hits=info["wall_hits"],
        visited_tiles=info["visited_tiles"],
        shortest_path=info["shortest_path"],
        episode_seed=info["episode_seed"],
        trajectory=traj,
    )
