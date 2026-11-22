"""ÖRNEK AJAN ŞABLONU — kopyala, adını değiştir, algoritmanı yaz.

Adımlar:
1. Bu dosyayı `agents/<senin_algoritman>.py` olarak kopyala.
2. Sınıf adını, `name` ve gerekirse `obs_mode` alanını değiştir.
3. `act`, `learn`, `end_episode` içindeki TODO'ları doldur.
4. Dene:  .venv\\Scripts\\python tools\\ajan_dene.py --agent <name> --episodes 2000

Şu haliyle çalışır ama hiçbir şey öğrenmez (rastgele hareket eder).
Kurallar ve gözlemin anlamı: agents/README.md ve maze_env/SPEC.md
"""

from __future__ import annotations

from typing import Any

from agents.base import Agent, EnvSpec


class OrnekAjan(Agent):
    # Arayüzde ve komut satırında görünecek kısa, benzersiz ad
    name = "ornek"

    # "simple": tek tam sayı (0..11519), tablo yöntemleri için (Q-tablosu satırı)
    # "full"  : 0/1 değerli float32 vektör, sinir ağlı yöntemler için
    obs_mode = "simple"

    def __init__(
        self,
        spec: EnvSpec,
        seed: int | None = None,
        alpha: float = 0.1,  # öğrenme oranı (örnek)
        gamma: float = 0.99,  # indirim faktörü (örnek)
        epsilon: float = 1.0,  # keşif oranı (örnek)
        **hparams: Any,
    ) -> None:
        # Hiperparametreleri üst sınıfa da ver: sonuç klasörüne kaydedilir
        super().__init__(spec, seed, alpha=alpha, gamma=gamma, epsilon=epsilon, **hparams)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # TODO: algoritmanın veri yapılarını kur.
        # Örn. tablo yöntemi:  self.q = np.zeros((spec.observation_space.n, spec.n_actions))
        # Örn. sinir ağı:     girdi boyutu spec.observation_space.shape[0], çıktı spec.n_actions
        # Rastgelelik için her zaman self.rng kullan (seed'e bağlı, tekrar üretilebilir).

    def act(self, obs, greedy: bool = False) -> int:
        # TODO: greedy=True ise en iyi eylemi, değilse keşifli eylemi (ör. epsilon-greedy) döndür.
        return int(self.rng.integers(self.spec.n_actions))

    def learn(self, obs, action, reward, next_obs, terminated, truncated, info) -> None:
        # TODO: her adımda çağrılır. TD yöntemleri (Q-learning, SARSA, DQN) burada günceller.
        # terminated: çıkışa ulaşıldı -> sonraki durumun değeri 0.
        # truncated : adım limiti doldu -> gerçek bitiş değil, next_obs'tan bootstrap yap.
        pass

    def end_episode(self) -> None:
        # TODO: bölüm sonunda çağrılır. Monte Carlo / REINFORCE güncellemesi ya da
        # epsilon azaltma gibi işler burada yapılabilir.
        pass
