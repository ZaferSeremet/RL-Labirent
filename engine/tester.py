"""Test motoru. Ayrıntı: engine/CONTRACT.md "tester"."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Sequence

from agents.base import Agent, EnvSpec
from maze_env import MazeEnv

from . import settings as _settings
from . import storage
from .episode import EpisodeResult, run_episode
from .registry import discover

ProgressFn = Callable[[dict[str, Any]], None]


def test_shared(settings: dict[str, Any]) -> dict[str, Any]:
    """Testte kullanılacak ortak ayar: ortam ve ödüller çalışmanın kendi kaydından
    (eğitildiği koşullar), test labirentleri (seed'ler ve boyutlar) GÜNCEL ortak ayardan.
    Böylece eski çalışmalar da yeniden eğitilmeden güncel test setinde test edilir."""
    return {**settings["shared"], "test": _settings.load_shared_config()["test"]}


def _env_for(kind: str, cls: type[Agent], settings: dict[str, Any]) -> MazeEnv:
    tc = settings.get("train_config", {})
    # Boyut test türünden gelir; eğitimdeki maze_size geçersiz kılması testlere taşınmaz
    overrides = {k: v for k, v in (tc.get("env_overrides") or {}).items() if k != "maze_size"}
    return MazeEnv(
        _settings.env_config_for(kind, cls.obs_mode, test_shared(settings),
                                 tc.get("reward_overrides") or {}, **overrides)
    )


def _fresh_agent(cls: type[Agent], env: MazeEnv, settings: dict[str, Any], run_dir: Path,
                 repeat: int | None) -> Agent:
    """Her test bölümü için yeni ajan: bölümler birbirinden bağımsız ve tekrar oynatılabilir olur.
    repeat None ise baseline (seed=0, model yok)."""
    spec = EnvSpec.from_env(env)
    if repeat is None:
        return cls(spec, seed=0)
    tc = settings.get("train_config", {})
    agent = cls(spec, seed=int(tc.get("seed", 0)) + repeat, **(tc.get("hparams") or {}))
    agent.load(storage.model_path(run_dir, repeat))
    return agent


def _models(run_dir: Path, settings: dict[str, Any]) -> list[int]:
    n = int(settings.get("train_config", {}).get("repeats", 1))
    return [r for r in range(n) if storage.model_path(run_dir, r).is_file()]


def test_run(
    run_dir: Path,
    kinds: Sequence[str] = ("normal", "buyuk"),
    progress: ProgressFn | None = None,
    stop: threading.Event | None = None,
) -> dict[str, Path]:
    """Eğitilmiş çalışmayı ve baseline'ları test labirentlerinde koşturur.
    Dönüş: {kind: test_<kind>.csv yolu}. Ayrıntı: engine/CONTRACT.md "tester"."""
    run_dir = Path(run_dir)
    kinds = tuple(kinds)
    for k in kinds:
        if k not in _settings.TEST_KINDS:
            raise ValueError(f"Bilinmeyen test türü: {k} (olması gerekenler: {_settings.TEST_KINDS})")
    emit: ProgressFn = progress or (lambda _ev: None)
    settings = storage.read_settings(run_dir)
    shared = test_shared(settings)
    reg = discover()
    agent_cls = reg.get(settings["agent"]["name"])  # yoksa anlaşılır KeyError
    models = _models(run_dir, settings)
    baselines = reg.baselines()

    n_methods = len(models) + len(baselines)
    total = sum(n_methods * len(_settings.seeds_for(k, shared)) for k in kinds)
    emit({"type": "start", "run_dir": str(run_dir), "kinds": list(kinds), "total": total})

    # (sınıf, tekrar, is_baseline); ajan önce, baseline'lar registry sırasıyla
    plan: list[tuple[type[Agent], int | None, bool]] = (
        [(agent_cls, r, False) for r in models] + [(b, None, True) for b in baselines]
    )

    paths: dict[str, Path] = {}
    stopped = False
    for kind in kinds:
        seeds = _settings.seeds_for(kind, shared)
        path = storage.test_csv_path(run_dir, kind)
        path.unlink(missing_ok=True)
        paths[kind] = path
        kind_complete = True
        with storage.CsvAppender(path, storage.TEST_FIELDS) as out:
            for cls, repeat, is_base in plan:
                env = _env_for(kind, cls, settings)
                for seed in seeds:
                    if stop is not None and stop.is_set():
                        stopped = True
                        kind_complete = False
                        break
                    agent = _fresh_agent(cls, env, settings, run_dir, repeat)
                    res = run_episode(env, agent, seed=seed, train=False)
                    rep = 0 if repeat is None else repeat
                    out.write({
                        "method": cls.name, "is_baseline": is_base, "repeat": rep, "seed": seed,
                        "total_reward": res.total_reward, "steps": res.steps, "success": res.success,
                        "wall_hits": res.wall_hits, "visited_tiles": res.visited_tiles,
                        "shortest_path": res.shortest_path, "efficiency": res.efficiency,
                    })
                    out.flush()
                    emit({"type": "test_episode", "kind": kind, "method": cls.name, "repeat": rep,
                          "seed": int(seed), "success": bool(res.success), "steps": int(res.steps)})
                if stopped:
                    break
        if kind_complete:
            emit({"type": "kind_done", "kind": kind, "path": str(path)})
        if stopped:
            break

    emit({"type": "done", "run_dir": str(run_dir), "stopped": stopped})
    return paths


test_run.__test__ = False  # pytest bunu test sanmasın


def replay_test_episode(
    run_dir: Path, kind: str, method: str, repeat: int, seed: int
) -> dict[str, Any]:
    """Bir test bölümünü yeniden koşturup izleme kaydını döndürür (deterministik)."""
    run_dir = Path(run_dir)
    if kind not in _settings.TEST_KINDS:
        raise ValueError(f"Bilinmeyen test türü: {kind}")
    settings = storage.read_settings(run_dir)
    reg = discover()
    cls = reg.get(method)  # yoksa anlaşılır KeyError
    if cls.is_baseline:
        agent_repeat = None
    else:
        if method != settings["agent"]["name"]:
            raise KeyError(f"Bu çalışmanın ajanı '{settings['agent']['name']}', '{method}' değil")
        agent_repeat = int(repeat)
    env = _env_for(kind, cls, settings)
    agent = _fresh_agent(cls, env, settings, run_dir, agent_repeat)
    res: EpisodeResult = run_episode(env, agent, seed=int(seed), train=False, record=True)
    return res.trajectory
