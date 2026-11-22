"""Bütün ajanların (RL algoritmaları ve baseline'lar) uyduğu ortak arayüz.

Motor (eğitim/test) ve arayüz ajanın içini bilmez; yalnız bu metodları çağırır.
Ajan ortamın kendisine erişemez: yalnız `EnvSpec` (gözlem/eylem uzayı ve
pencere boyutu) ve her adımda ortamın döndürdüğü gözlemi alır.

Bir bölümde motorun çağırma sırası:

    obs, info = env.reset(seed=...)
    agent.begin_episode(obs, info)
    döngü:
        action = agent.act(obs, greedy=False)          # eğitimde keşifli
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.learn(obs, action, reward, next_obs, terminated, truncated, info)
        obs = next_obs
        terminated ya da truncated ise çık
    agent.end_episode()

Test sırasında `act(obs, greedy=True)` çağrılır; `learn` ve `end_episode` çağrılmaz.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    """Ajanın ortam hakkında bilmesine izin verilen her şey."""

    obs_mode: str  # "full" ya da "simple"
    observation_space: gym.Space
    action_space: gym.Space
    view_radius: int
    window_size: int  # 2 * view_radius + 1
    n_actions: int

    @classmethod
    def from_env(cls, env) -> "EnvSpec":
        return cls(
            obs_mode=env.config.obs_mode,
            observation_space=env.observation_space,
            action_space=env.action_space,
            view_radius=env.config.view_radius,
            window_size=env.window_size,
            n_actions=int(env.action_space.n),
        )


class Agent:
    """Ajanların temel sınıfı.

    Alt sınıfın tanımlaması gerekenler:
    - `name`: arayüzde görünen kısa ad (benzersiz olmalı)
    - `obs_mode`: istediği gözlem ("full" ya da "simple"); motor ortamı buna göre kurar
    - `act()`; öğrenen ajanlar ayrıca `learn()` ve gerekirse `end_episode()`
    """

    name: str = "ajan"
    obs_mode: str = "full"
    is_baseline: bool = False  # True ise eğitilmez, yalnız test edilir

    def __init__(self, spec: EnvSpec, seed: int | None = None, **hparams: Any) -> None:
        if spec.obs_mode != self.obs_mode:
            raise ValueError(
                f"{self.name} '{self.obs_mode}' gözlem istiyor, ortam '{spec.obs_mode}' veriyor"
            )
        self.spec = spec
        self.seed = seed
        self.hparams = dict(hparams)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ zorunlu

    def act(self, obs: Any, greedy: bool = False) -> int:
        """Gözleme göre eylem seç (0-3).

        greedy=False: eğitim, keşif yapılabilir (ör. epsilon-greedy).
        greedy=True : test, ajan en iyi bildiği eylemi seçer.
        """
        raise NotImplementedError

    # ------------------------------------------------------------ isteğe bağlı

    def begin_episode(self, obs: Any, info: dict) -> None:
        """Bölüm başında çağrılır (ör. bölüm tamponunu sıfırlamak için)."""

    def learn(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        """Her eğitim adımından sonra çağrılır.

        DİKKAT: `terminated` (çıkışa ulaşıldı) gerçek bir bitiştir; sonraki
        durumun değeri 0 alınır. `truncated` (adım limiti doldu) gerçek bir
        bitiş DEĞİLDİR; TD yöntemlerinde next_obs üzerinden bootstrap yapılmalıdır.
        `info` içindeki `shortest_path` gibi alanlar ajan tarafından kullanılmamalıdır.
        """

    def end_episode(self) -> None:
        """Bölüm sonunda çağrılır (ör. Monte Carlo ve REINFORCE güncellemesi, epsilon azaltma)."""

    def hyperparameters(self) -> dict[str, Any]:
        """Sonuç klasöründeki ayarlar.json'a yazılır (tekrar üretilebilirlik)."""
        return dict(self.hparams)

    # ------------------------------------------------------------ kaydetme

    def save(self, path: str | Path) -> None:
        """Varsayılan: ajanın bütün alanlarını pickle ile kaydeder.
        PyTorch kullanan ajanlar bunu `torch.save` ile ezebilir."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))
