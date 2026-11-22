"""Baseline: her adımda rastgele bir yön seçer. Öğrenmez."""

from __future__ import annotations

from agents.base import Agent


class RandomAgent(Agent):
    name = "random"
    obs_mode = "full"  # gözleme bakmadığı için fark etmez
    is_baseline = True

    def act(self, obs, greedy: bool = False) -> int:
        return int(self.rng.integers(self.spec.n_actions))
