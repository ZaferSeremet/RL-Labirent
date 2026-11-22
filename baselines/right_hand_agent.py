"""Baseline: sağ el kuralı (sağ elini hep duvarda tutarak yürü). Öğrenmez.

Bakış yönü = gözlemdeki son eylem (bölüm başında yukarı kabul edilir).
Sırayla dener: sağa dön, düz git, sola dön, geri dön; ilk açık olanı seçer.
Döngüsüz (mükemmel) labirentte çıkışı her zaman bulur, ama en kısa yoldan gitmez.

Hafıza tutmaz: ihtiyacı olan tek bilgi (bakış yönü) gözlemin içinde.
"""

from __future__ import annotations

import numpy as np

from agents.base import Agent

_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # yukarı, sağ, aşağı, sol
_TURN_ORDER = (1, 0, 3, 2)  # sağ, düz, sol, geri (bakış yönüne göre)


class RightHandAgent(Agent):
    name = "sag_el"
    obs_mode = "full"
    is_baseline = True

    def act(self, obs, greedy: bool = False) -> int:
        k = self.spec.window_size
        r = self.spec.view_radius
        walls = np.asarray(obs[: k * k]).reshape(k, k)
        last = int(np.argmax(obs[-5:]))
        heading = 0 if last == 4 else last
        for turn in _TURN_ORDER:
            a = (heading + turn) % 4
            dy, dx = _DELTAS[a]
            if walls[r + dy, r + dx] < 0.5:
                return a
        return heading  # dört taraf duvar (olmaz), yerinde kal
