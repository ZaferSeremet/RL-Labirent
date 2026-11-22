"""Ortam ve ödül ayarları. JSON'a yazılıp okunabilir (tekrar üretilebilirlik)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

OBS_MODES = ("full", "simple")


@dataclass
class RewardConfig:
    step: float = -0.01  # her adım
    wall: float = -0.05  # duvara çarpma (adım cezasına ek)
    exit: float = 5.0  # çıkışa ulaşma
    new_tile: float = 0.0  # ilk kez girilen kare (keşif bonusu, varsayılan kapalı)
    timeout: float = 0.0  # adım limiti dolunca ek ceza (varsayılan yok)
    # Çıkışa yaklaşma: labirent içi yol mesafesi 1 azalınca +closer, 1 artınca -closer.
    # Simetrik olduğu için ileri-geri giderek ödül toplanamaz (bölüm toplamı
    # closer x (başlangıç mesafesi - son mesafe)). Varsayılan kapalı.
    closer: float = 0.0


@dataclass
class EnvConfig:
    maze_size: int = 8  # labirentin bir kenarındaki hücre sayısı (kare labirent)
    view_radius: int = 1  # görüş penceresi (2r+1)x(2r+1); 1 -> 3x3, 2 -> 5x5
    max_steps: int | None = None  # None -> 3 x boş kare sayısı
    obs_mode: str = "full"  # "full" (vektör) ya da "simple" (tek tam sayı, tablo için)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if isinstance(self.rewards, dict):
            self.rewards = RewardConfig(**self.rewards)
        if self.maze_size < 2:
            raise ValueError("maze_size en az 2 olmalı")
        if self.view_radius < 1:
            raise ValueError("view_radius en az 1 olmalı")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps pozitif olmalı")
        if self.obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode {OBS_MODES} içinden biri olmalı")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvConfig":
        return cls(**d)
