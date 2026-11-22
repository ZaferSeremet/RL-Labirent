"""Eğitim motoru. Ayrıntı: engine/CONTRACT.md "trainer"."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.base import Agent, EnvSpec
from maze_env import MazeEnv

from . import settings as _settings  # load_shared_config'e modül üzerinden eriş (testte yamalanabilsin)
from . import storage
from .episode import run_episode
from .registry import discover

ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class TrainConfig:
    agent_name: str  # registry'deki ad
    episodes: int = 2000  # her tekrardaki bölüm sayısı
    repeats: int = 3  # aynı ayarla kaç bağımsız eğitim
    seed: int = 0  # r. tekrar seed'i = seed + r (hem ajan hem eğitim labirent dizisi)
    label: str = ""  # klasör adına ve ayarlar.json'a eklenir (ör. "adim-cezasi-yuksek")
    reward_overrides: dict[str, float] = field(default_factory=dict)  # ortak ödüllerin üstüne
    env_overrides: dict[str, Any] = field(default_factory=dict)  # ör. {"view_radius": 2}
    hparams: dict[str, Any] = field(default_factory=dict)  # ajan kurucusuna verilir
    snapshots: int = 5  # 0. tekrarda eşit aralıklı kayıt sayısı (ilk ve son bölüm dahil)
    live_every: int | None = None  # None -> max(1, episodes // 100); kayitlar/son.json
    results_root: Path | None = None  # None -> storage.RESULTS_ROOT


def snapshot_episodes(episodes: int, snapshots: int) -> list[int]:
    """0. tekrarda eşit aralıklı kaydedilecek bölüm numaraları."""
    if snapshots < 1 or episodes < 1:
        return []
    return sorted({int(round(float(x))) for x in np.linspace(1, episodes, snapshots)})


def _build(cls: type[Agent], cfg: TrainConfig, shared: dict[str, Any], r: int) -> tuple[MazeEnv, Agent]:
    env = MazeEnv(
        _settings.env_config_for("egitim", cls.obs_mode, shared, cfg.reward_overrides, **cfg.env_overrides)
    )
    agent = cls(EnvSpec.from_env(env), seed=cfg.seed + r, **cfg.hparams)
    return env, agent


def train(
    cfg: TrainConfig,
    progress: ProgressFn | None = None,
    stop: threading.Event | None = None,
) -> Path:
    """Ajanı eğitir, sonuç klasörünü döndürür. Ayrıntı: engine/CONTRACT.md "trainer"."""
    cls = discover().get(cfg.agent_name)  # yoksa anlaşılır KeyError
    if cls.is_baseline:
        raise ValueError(f"'{cfg.agent_name}' bir baseline; baseline'lar eğitilmez, yalnız test edilir")
    if cfg.episodes < 1 or cfg.repeats < 1:
        raise ValueError("episodes ve repeats en az 1 olmalı")

    emit: ProgressFn = progress or (lambda _ev: None)
    shared = _settings.load_shared_config()
    # İlk tekrarı klasör açmadan kur: hatalı hiperparametre boş klasör bırakmasın
    first = _build(cls, cfg, shared, 0)

    run_dir = storage.new_run_dir(cfg.agent_name, cfg.label, cfg.results_root)
    t0 = time.perf_counter()
    train_config = asdict(cfg)
    train_config.pop("results_root", None)
    storage.write_settings(run_dir, {
        "run_id": run_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "durum": "egitiliyor",
        "agent": {
            "name": cls.name,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "obs_mode": cls.obs_mode,
            "is_baseline": bool(cls.is_baseline),
        },
        "train_config": train_config,
        "env": first[0].config.to_dict(),
        "shared": shared,
        "hyperparameters": first[1].hyperparameters(),
    })

    episodes = cfg.episodes
    snap_eps = set(snapshot_episodes(episodes, cfg.snapshots))
    live_every = cfg.live_every if cfg.live_every else max(1, episodes // 100)
    stopped = False
    completed = 0

    emit({"type": "start", "run_dir": str(run_dir), "episodes": episodes,
          "repeats": cfg.repeats, "agent": cls.name})
    try:
        with storage.CsvAppender(storage.train_log_path(run_dir), storage.TRAIN_FIELDS) as log:
            for r in range(cfg.repeats):
                env, agent = first if r == 0 else _build(cls, cfg, shared, r)
                # Eğitim labirent dizisini sabitle; bölümler seed'siz sürer
                env.reset(seed=shared["egitim"]["seed_baslangic"] + cfg.seed + r)
                repeat_finished = True
                ran = 0
                for e in range(1, episodes + 1):
                    if stop is not None and stop.is_set():
                        stopped = True
                        repeat_finished = False
                        break
                    ran += 1
                    snap = r == 0 and e in snap_eps
                    live = e % live_every == 0
                    res = run_episode(env, agent, train=True, record=snap or live)
                    log.write({
                        "repeat": r, "episode": e, "total_reward": res.total_reward,
                        "steps": res.steps, "success": res.success, "wall_hits": res.wall_hits,
                        "visited_tiles": res.visited_tiles, "shortest_path": res.shortest_path,
                        "episode_seed": res.episode_seed,
                    })
                    log.flush()
                    emit({"type": "episode", "repeat": r, "episode": e,
                          "total_reward": float(res.total_reward), "steps": int(res.steps),
                          "success": bool(res.success)})
                    if snap:
                        p = storage.save_trajectory(storage.snapshot_path(run_dir, r, e), res.trajectory)
                        emit({"type": "snapshot", "repeat": r, "episode": e, "path": str(p)})
                    if live:
                        try:
                            p = storage.save_trajectory(storage.live_path(run_dir), res.trajectory)
                        except OSError:
                            pass  # canlı kayıt yalnız izleme içindir; yazılamazsa eğitimi düşürme
                        else:
                            emit({"type": "live", "repeat": r, "episode": e, "path": str(p)})
                if stopped and ran == 0 and r > 0:
                    break  # bu tekrar hiç başlamadı: eğitilmemiş model kaydetme
                mp = Path(storage.model_path(run_dir, r))
                agent.save(mp)
                emit({"type": "repeat_done", "repeat": r, "model_path": str(mp)})
                if repeat_finished:
                    completed += 1
                if stopped:
                    break
    except BaseException as exc:
        storage.update_settings(run_dir, durum="hata", hata=f"{type(exc).__name__}: {exc}",
                                sure_sn=round(time.perf_counter() - t0, 3), tamamlanan_tekrar=completed)
        raise

    storage.update_settings(
        run_dir,
        durum="durduruldu" if stopped else "tamamlandi",
        sure_sn=round(time.perf_counter() - t0, 3),
        tamamlanan_tekrar=completed,
    )
    emit({"type": "done", "run_dir": str(run_dir), "stopped": stopped})
    return run_dir
