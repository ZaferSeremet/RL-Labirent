"""Ortak ayarları (configs/ortak_ayarlar.json) okur ve ortam ayarı üretir."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from maze_env import EnvConfig

ROOT = Path(__file__).resolve().parent.parent
SHARED_CONFIG_PATH = ROOT / "configs" / "ortak_ayarlar.json"
TEST_KINDS = ("normal", "buyuk")


# Yalnız otomatik testler için: başka bir ortak ayar dosyası (ayrı süreçlere de geçer)
SHARED_CONFIG_ENV = "RL_LABIRENT_SHARED_CONFIG"


def load_shared_config(path: str | Path | None = None) -> dict[str, Any]:
    path = path or os.environ.get(SHARED_CONFIG_ENV) or SHARED_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def env_config_for(
    kind: str,
    obs_mode: str,
    shared: dict[str, Any] | None = None,
    reward_overrides: dict[str, float] | None = None,
    **env_overrides: Any,
) -> EnvConfig:
    """kind: "egitim", "normal" ya da "buyuk".

    Eğitim ve iki test ortamı aynı ayarları paylaşır; testlerde labirent boyutu
    ortak ayardaki test boyutuyla değiştirilir.
    """
    shared = shared or load_shared_config()
    d = {k: v for k, v in shared["env"].items() if not k.startswith("_")}
    d["rewards"] = {**d["rewards"], **(reward_overrides or {})}
    if kind in TEST_KINDS:
        d["maze_size"] = shared["test"][kind]["maze_size"]
    elif kind != "egitim":
        raise ValueError(f"Bilinmeyen ortam türü: {kind}")
    d.update(env_overrides)
    d["obs_mode"] = obs_mode
    return EnvConfig.from_dict(d)


def seeds_for(kind: str, shared: dict[str, Any] | None = None) -> list[int]:
    """Test labirentlerinin seed listesi (ortak ayardaki aralık, iki uç dahil)."""
    shared = shared or load_shared_config()
    first, last = shared["test"][kind]["seeds"]
    return list(range(first, last + 1))
