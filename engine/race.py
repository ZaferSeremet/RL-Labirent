"""Yarış: eğitilmiş ajan ve baseline'lar AYNI labirentte koşar (SÖZLEŞME: imzalar sabit).

Test sekmesindeki izleme ve Final Test sekmesi bunu kullanır. Her oyuncu kendi
ortam kopyasında oynar (aynı seed ve boyut -> aynı labirent); birbirlerini etkilemezler.

Yarış kaydı (race) sözlüğü:
    {"grid": (H,W) uint8, "start": (y,x), "exit": (y,x), "view_radius": r,
     "maze_size": n, "seed": s, "max_steps": m,
     "racers": [  # sıra sabit: önce eğitilmiş ajan, sonra baseline'lar registry sırasıyla
        {"key": str,            # benzersiz: "agent" ya da baseline adı
         "method": str,         # ajanın name alanı ("ppo", "sag_el", "random")
         "is_baseline": bool,
         "positions": [(y,x), ...], "actions": [int ...], "rewards": [float ...],
         "success": bool, "steps": int, "wall_hits": int, "total_reward": float,
         "visited_tiles": int, "shortest_path": int},
        ...]}
Görünen ad (dil, "Sağ el" vb.) arayüzün işidir: baseline için engine.i18n.method_label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.base import Agent
from maze_env import MazeEnv

from . import settings as _settings
from . import storage
from .episode import run_episode
from .registry import discover
from .tester import _env_for, _fresh_agent, _models

# Final testte "rastgele" seed bu aralıktan seçilir: test seed'leriyle (10000-.., 20000-..) çakışmaz.
RANDOM_SEED_RANGE = (100_000, 999_999)
MIN_MAZE_SIZE = 2
MAX_MAZE_SIZE = 40


def _custom_env(cls: type[Agent], settings: dict[str, Any], maze_size: int) -> MazeEnv:
    """Çalışmanın ödül ve ortam ayarlarıyla, istenen boyutta ortam."""
    tc = settings.get("train_config", {})
    overrides = {k: v for k, v in (tc.get("env_overrides") or {}).items() if k != "maze_size"}
    overrides["maze_size"] = int(maze_size)
    return MazeEnv(
        _settings.env_config_for("normal", cls.obs_mode, settings["shared"],
                                 tc.get("reward_overrides") or {}, **overrides)
    )


def _race(run_dir: Path, repeat: int, seed: int, env_factory) -> dict[str, Any]:
    settings = storage.read_settings(run_dir)
    reg = discover()
    agent_cls = reg.get(settings["agent"]["name"])  # yoksa anlaşılır KeyError
    models = _models(run_dir, settings)
    if int(repeat) not in models:
        raise ValueError(f"Bu çalışmada tekrar {repeat} modeli yok (olanlar: {models})")

    plan = [("agent", agent_cls, int(repeat))] + [(b.name, b, None) for b in reg.baselines()]
    racers: list[dict[str, Any]] = []
    race: dict[str, Any] = {}
    for key, cls, rep in plan:
        env = env_factory(cls, settings)
        agent = _fresh_agent(cls, env, settings, run_dir, rep)
        res = run_episode(env, agent, seed=int(seed), train=False, record=True)
        traj = res.trajectory
        assert traj is not None
        if not race:
            race = {
                "grid": traj["grid"], "start": tuple(traj["start"]), "exit": tuple(traj["exit"]),
                "view_radius": traj["view_radius"], "maze_size": env.config.maze_size,
                "seed": int(seed), "max_steps": env.max_steps,
            }
        racers.append({
            "key": key, "method": cls.name, "is_baseline": bool(cls.is_baseline),
            "positions": [tuple(p) for p in traj["positions"]],
            "actions": list(traj["actions"]), "rewards": list(traj["rewards"]),
            "success": bool(res.success), "steps": int(res.steps), "wall_hits": int(res.wall_hits),
            "total_reward": float(res.total_reward), "visited_tiles": int(res.visited_tiles),
            "shortest_path": int(res.shortest_path),
        })
    race["racers"] = racers
    return race


def race_test_episode(run_dir: Path, kind: str, repeat: int, seed: int) -> dict[str, Any]:
    """Test labirentinde (kind: "normal"/"buyuk", seed o türün test seed'i) yarış."""
    if kind not in _settings.TEST_KINDS:
        raise ValueError(f"Bilinmeyen test türü: {kind}")
    return _race(Path(run_dir), repeat, seed, lambda cls, st: _env_for(kind, cls, st))


def race_custom(run_dir: Path, repeat: int, seed: int, maze_size: int) -> dict[str, Any]:
    """Final test: istenen seed ve boyutta (MIN_MAZE_SIZE..MAX_MAZE_SIZE) yarış."""
    size = int(maze_size)
    if not MIN_MAZE_SIZE <= size <= MAX_MAZE_SIZE:
        raise ValueError(f"Labirent boyutu {MIN_MAZE_SIZE}-{MAX_MAZE_SIZE} arasında olmalı")
    return _race(Path(run_dir), repeat, seed, lambda cls, st: _custom_env(cls, st, size))
