"""PPO (Proximal Policy Optimization) ajanı — yalnız NumPy ile, sıfırdan.

Hazır RL kütüphanesi kullanılmaz; sinir ağı, geri yayılım (backprop) ve Adam
optimizer da bu dosyada elle yazılmıştır.

Deneme:
    .venv\\Scripts\\python tools\\ajan_dene.py --agent ppo --episodes 2000

──────────────────────────────────────────────────────────────────────────────
PPO KISACA (dosyayı okurken bu haritayı aklında tut)

PPO bir "policy gradient" (politika gradyanı) yöntemidir. Q-learning gibi
"her eylemin değerini" öğrenip en büyüğünü seçmez; doğrudan bir POLİTİKA öğrenir:
    π(a | s) = s gözleminde a eylemini seçme OLASILIĞI
Yani ağın çıktısı 4 eylem için bir olasılık dağılımıdır (softmax).

İki ağ vardır (ikisine birden "actor-critic" denir):
  * Actor  (politika ağı): gözlem -> 4 eylemin olasılığı. Karar veren.
  * Critic (değer ağı)   : gözlem -> V(s), "buradan sonra ortalama ne kadar ödül
                           toplarım" tahmini. Karar vermez, actor'a not verir.

Döngü:
  1. TOPLA   : mevcut politikayla oyna, `rollout_steps` adımı tampona yaz.
  2. DEĞERLENDİR: her adım için AVANTAJ hesapla (GAE):
                  A = "bu eylem, o durumdaki ortalamadan ne kadar iyi çıktı?"
                  A > 0 -> eylem beklenenden iyiydi, olasılığını artır.
                  A < 0 -> kötüydü, olasılığını azalt.
  3. GÜNCELLE: aynı veriyle birkaç tur (epoch) mini-batch gradyan adımı at.
               Ama politikanın çok değişmesine izin verme (KIRPMA / clip):
                  oran r = π_yeni(a|s) / π_eski(a|s)
                  r'yi [1-ε, 1+ε] dışına itmek ödül getirmez.
               PPO'nun "proximal" (yakın) olması budur: yeni politika eskisine
               yakın kalır, bu yüzden eğitim kararlıdır.
  4. Tamponu boşalt, 1'e dön.

Önemli: PPO "on-policy"dir. Toplanan veri yalnız onu toplayan politikayla
kullanılabilir; güncellemeden sonra veri atılır (DQN'in replay buffer'ı yok).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base import Agent, EnvSpec


# ════════════════════════════════════════════════════════════════════════════
# 1) Küçük bir sinir ağı (MLP) ve elle geri yayılım
# ════════════════════════════════════════════════════════════════════════════


class MLP:
    """Tam bağlı ağ: girdi -> [Linear -> tanh] x gizli katmanlar -> Linear -> çıktı.

    PyTorch'ta autograd gradyanı kendisi hesaplar; burada `backward` içinde
    zincir kuralını kendimiz uyguluyoruz.
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator, out_gain: float) -> None:
        self.params: list[np.ndarray] = []  # [W0, b0, W1, b1, ...]
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            gain = out_gain if last else np.sqrt(2.0)
            self.params.append(_orthogonal((sizes[i], sizes[i + 1]), gain, rng))
            self.params.append(np.zeros(sizes[i + 1], dtype=np.float64))

    @property
    def n_layers(self) -> int:
        return len(self.params) // 2

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Çıktıyı ve geri yayılım için her katmanın girdisini (cache) döndürür."""
        cache = [x]
        h = x
        for i in range(self.n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ W + b
            if i < self.n_layers - 1:
                h = np.tanh(z)
                cache.append(h)
            else:
                h = z  # son katmanda aktivasyon yok (logit ya da değer)
        return h, cache

    def backward(self, cache: list[np.ndarray], d_out: np.ndarray) -> list[np.ndarray]:
        """d_out = dLoss/dÇıktı. Her parametrenin gradyanını döndürür.

        Zincir kuralı, sondan başa:
          z = h W + b    ->  dW = hᵀ·dz,  db = Σ dz,  dh = dz·Wᵀ
          h = tanh(z)    ->  dz = dh · (1 - h²)
        """
        grads: list[np.ndarray] = [None] * len(self.params)  # type: ignore[list-item]
        d = d_out
        for i in reversed(range(self.n_layers)):
            h_in = cache[i]
            W = self.params[2 * i]
            grads[2 * i] = h_in.T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = (d @ W.T) * (1.0 - h_in**2)  # h_in, bir önceki tanh'ın çıktısı
        return grads


def _orthogonal(shape: tuple[int, int], gain: float, rng: np.random.Generator) -> np.ndarray:
    """Ortogonal başlatma: PPO uygulamalarında standart, eğitimi kararlı başlatır."""
    a = rng.standard_normal(shape)
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else vt
    return gain * q


class Adam:
    """Adam optimizer: her parametre için gradyanın hareketli ortalaması (m) ve
    karesinin hareketli ortalaması (v) tutulur; adım = lr · m̂ / (√v̂ + eps)."""

    def __init__(self, params: list[np.ndarray], lr: float) -> None:
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)  # yerinde güncelleme


def _clip_grad_norm(grads: list[np.ndarray], max_norm: float) -> None:
    """Bütün gradyanların toplam uzunluğu max_norm'u geçerse küçült (patlamayı önler)."""
    total = np.sqrt(sum(float((g * g).sum()) for g in grads))
    if total > max_norm:
        for g in grads:
            g *= max_norm / (total + 1e-8)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)  # sayısal kararlılık
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════════
# 2) PPO ajanı
# ════════════════════════════════════════════════════════════════════════════


class PPOAjan(Agent):
    name = "ppo"
    # Sinir ağı kullandığımız için zengin gözlem: 3x3 pencerenin duvar, çıkış ve
    # ziyaret haritaları + son eylem = 3·9 + 5 = 32 sayılık 0/1 vektör.
    obs_mode = "full"

    def __init__(
        self,
        spec: EnvSpec,
        seed: int | None = None,
        lr: float = 1e-3,  # Adam öğrenme oranı (3e-4 bu küçük ağda yavaş kaldı)
        lr_end: float | None = None,  # zamanla azaltma hedefi; None = lr sabit
        gamma: float = 0.99,  # indirim: gelecekteki ödüle ne kadar önem verilir
        gae_lambda: float = 0.95,  # GAE: avantaj tahmininde yanlılık/varyans dengesi
        clip_eps: float = 0.2,  # PPO kırpma aralığı: oran [0.8, 1.2] dışına itilmez
        rollout_steps: int = 2048,  # bir güncellemeden önce toplanan adım sayısı
        epochs: int = 4,  # aynı veri üzerinden kaç tur geçilir
        minibatch_size: int = 256,
        # Entropi bonusu: politika erkenden tek eyleme kilitlenmesin. Bu ortamda kritik:
        # 0.01 ile entropi ~0'a çöktü (ajan duvardan kaçmayı öğrenip çıkışı hiç bulamadı);
        # 0.05 keşfi canlı tuttu. 0.2 fazla: politika hep rastgele kaldı.
        ent_coef: float = 0.05,
        ent_coef_end: float | None = None,  # zamanla azaltma hedefi; None = ent_coef sabit
        decay_episodes: int = 6000,  # azaltma kaç bölümde biter (sonra sabit kalır)
        max_grad_norm: float = 0.5,
        hidden: int = 64,  # gizli katman genişliği
        # Kısa süreli hafıza: ağa son kaç gözlemin yan yana verileceği ("frame stacking").
        # 1 = hafızasız (yalnız şu anki gözlem). Ayrıntı: aşağıdaki "Kısa süreli hafıza" bölümü.
        # 5000 bölüm, 3 seed, 8x8 test (argmax) ortalaması: 1 -> %43, 2 -> %53, 4 -> %33
        # (4'te bir seed hiç kalkamadı: girdi büyüdükçe öğrenme yavaşlıyor), 8 -> %0.
        memory: int = 2,
        **hparams: Any,
    ) -> None:
        # Eski ad: ilk sürümde bu ayarın adı "hafiza" idi; eski çalışmalar yine yüklensin.
        if "hafiza" in hparams:
            memory = hparams.pop("hafiza")
        super().__init__(
            spec, seed,
            lr=lr, lr_end=lr_end, gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            rollout_steps=rollout_steps, epochs=epochs, minibatch_size=minibatch_size,
            ent_coef=ent_coef, ent_coef_end=ent_coef_end, decay_episodes=decay_episodes,
            max_grad_norm=max_grad_norm, hidden=hidden, memory=memory,
            **hparams,
        )
        if int(memory) < 1:
            raise ValueError("memory en az 1 olmalı")
        if int(decay_episodes) < 1:
            raise ValueError("decay_episodes en az 1 olmalı")
        self.memory = int(memory)
        self.gamma = gamma
        self.lam = gae_lambda
        self.clip_eps = clip_eps
        self.rollout_steps = int(rollout_steps)
        self.epochs = int(epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = max_grad_norm

        # Zamanla azaltma (bkz. _schedule): başlangıç ve bitiş değerleri.
        self.lr_start = float(lr)
        self.lr_end = float(lr) if lr_end is None else float(lr_end)
        self.ent_start = float(ent_coef)
        self.ent_end = float(ent_coef) if ent_coef_end is None else float(ent_coef_end)
        self.decay_episodes = int(decay_episodes)
        self.episodes_done = 0  # biten eğitim bölümü sayısı (testte artmaz)
        self.ent_coef = self.ent_start

        self.obs_dim = int(spec.observation_space.shape[0])
        obs_dim = self.obs_dim * self.memory  # ağın girdisi: N gözlem art arda
        n_act = spec.n_actions
        # Actor'ın son katmanı çok küçük (0.01) başlatılır: başta bütün eylemler
        # neredeyse eşit olasılıklı olur, yani ajan rastgele keşifle başlar.
        self.actor = MLP([obs_dim, hidden, hidden, n_act], self.rng, out_gain=0.01)
        self.critic = MLP([obs_dim, hidden, hidden, 1], self.rng, out_gain=1.0)
        self.actor_opt = Adam(self.actor.params, lr)
        self.critic_opt = Adam(self.critic.params, lr)

        self._reset_buffer()
        self.n_updates = 0
        self.last_stats: dict[str, float] = {}
        self._stack: list[np.ndarray] | None = None  # son N gözlem (eskiden yeniye)
        self._last_input: np.ndarray | None = None  # act'in ağa verdiği son girdi

    # ------------------------------------------------------------ kısa süreli hafıza
    #
    # Ajan tek bir 3x3 pencereden "bu koridorda hangi yöne gidiyordum?" sorusunu
    # cevaplayamaz; aynı görüntüye hep aynı cevabı verip döngüye girebilir.
    # Çözüm: son N gözlemi art arda dizip ağa öyle vermek. Örn. N=4 ise girdi
    #     [gözlem(t-3), gözlem(t-2), gözlem(t-1), gözlem(t)]   (4 × 32 = 128 sayı)
    # Bu yine yalnız ajanın KENDİ gördükleridir, ortamdan ek bilgi alınmaz.
    #
    # Bölüm başında geçmiş yoktur; boş yerler ilk gözlemin kopyasıyla doldurulur.
    # Hafıza yalnız `act` içinde ilerler, böylece eğitimde ve testte aynı çalışır.

    def begin_episode(self, obs, info) -> None:
        self._stack = None  # yeni labirent: eski bölümün hafızası silinir
        self._last_input = None

    def _push(self, obs) -> np.ndarray:
        """Gözlemi hafızaya ekler, en eskiyi atar; ağın girdisini döndürür."""
        o = np.asarray(obs, dtype=np.float64)
        if self._stack is None:
            self._stack = [o] * self.memory
        else:
            self._stack = self._stack[1:] + [o]
        return np.concatenate(self._stack)

    def _peek_next(self, next_obs) -> np.ndarray:
        """next_obs gelseydi girdi ne olurdu (hafızayı değiştirmeden)."""
        assert self._stack is not None
        o = np.asarray(next_obs, dtype=np.float64)
        return np.concatenate(self._stack[1:] + [o])

    # ------------------------------------------------------------ karar verme

    def act(self, obs, greedy: bool = False) -> int:
        x = self._push(obs)
        self._last_input = x
        logits, _ = self.actor.forward(x[None, :])
        probs = _softmax(logits)[0]
        if greedy:
            return int(np.argmax(probs))  # test: en olası eylem
        # Eğitim: olasılığa göre örnekle. PPO'da keşif epsilon ile değil, politikanın
        # kendi rastgeleliğiyle yapılır; öğrendikçe olasılıklar keskinleşir.
        i = int(np.searchsorted(np.cumsum(probs), self.rng.random()))
        return min(i, len(probs) - 1)

    # ------------------------------------------------------------ veri toplama

    def _reset_buffer(self) -> None:
        self.buf_obs: list[np.ndarray] = []
        self.buf_next_obs: list[np.ndarray] = []
        self.buf_act: list[int] = []
        self.buf_rew: list[float] = []
        self.buf_term: list[bool] = []  # çıkışa ulaşıldı (gerçek bitiş)
        self.buf_end: list[bool] = []  # bölüm bitti (çıkış YA DA adım limiti)

    def learn(self, obs, action, reward, next_obs, terminated, truncated, info) -> None:
        # PPO her adımda öğrenmez; yalnız kaydeder. Tampon dolunca toplu güncelleme yapar.
        # Tampona ham gözlem değil, ağın gerçekten gördüğü hafızalı girdi yazılır.
        x = self._last_input if self._last_input is not None else self._push(obs)
        self.buf_obs.append(x)
        self.buf_next_obs.append(self._peek_next(next_obs))
        self.buf_act.append(int(action))
        self.buf_rew.append(float(reward))
        self.buf_term.append(bool(terminated))
        self.buf_end.append(bool(terminated or truncated))
        if len(self.buf_act) >= self.rollout_steps:
            self._update()
            self._reset_buffer()

    def end_episode(self) -> None:
        self.episodes_done += 1  # yalnız eğitimde çağrılır

    # ------------------------------------------------------------ zamanla azaltma

    def _schedule(self) -> tuple[float, float]:
        """Şu anki (lr, ent_coef) değerleri.

        İlk `decay_episodes` bölümde başlangıçtan bitiş değerine DOĞRUSAL iner,
        sonra bitiş değerinde sabit kalır:

            değer
            başlangıç ●
                       ╲
                        ╲
            bitiş        ●━━━━━━━━━━━━
                         decay_episodes      bölüm

        Ajan toplam bölüm sayısını bilmez; bu yüzden süre bölümle verilir.
        Bitiş değeri verilmezse (None) başlangıç = bitiş, yani değer sabit kalır.
        """
        frac = min(1.0, self.episodes_done / self.decay_episodes)
        lr = self.lr_start + (self.lr_end - self.lr_start) * frac
        ent = self.ent_start + (self.ent_end - self.ent_start) * frac
        return lr, ent

    # ------------------------------------------------------------ avantaj (GAE)

    def _compute_advantages(self, obs, next_obs, rew, term, end):
        """GAE (Generalized Advantage Estimation).

        Tek adımlık TD hatası:
            δ_t = r_t + γ · V(s_{t+1}) · (1 - terminated_t) - V(s_t)
        "Aldığım ödül + gelecekte beklediğim, önceden beklediğimden ne kadar fazla?"

        Avantaj, bu hataların geleceğe doğru azalan ağırlıklı toplamıdır:
            A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...   (bölüm bitince durur)
        λ=0 -> yalnız δ_t (az varyans, çok yanlılık); λ=1 -> Monte Carlo getirisi.

        terminated ile truncated farkı burada önemli:
          * terminated (çıkış): sonrası yok, V(s') = 0.
          * truncated (süre doldu): ajan aslında devam edebilirdi, V(s') kullanılır.
            Ama bir sonraki tampon satırı yeni bir bölüm olduğu için zincir yine kesilir.
        """
        v, _ = self.critic.forward(obs)
        v_next, _ = self.critic.forward(next_obs)
        v, v_next = v[:, 0], v_next[:, 0]

        deltas = rew + self.gamma * v_next * (1.0 - term) - v
        adv = np.zeros_like(rew)
        gae = 0.0
        for t in reversed(range(len(rew))):
            gae = deltas[t] + self.gamma * self.lam * (1.0 - end[t]) * gae
            adv[t] = gae
        returns = adv + v  # critic'in hedefi: V(s) ≈ A + V_eski(s)
        return adv, returns

    # ------------------------------------------------------------ güncelleme

    def _update(self) -> None:
        # Bu güncellemede kullanılacak lr ve ent_coef (zamanla azaltma)
        lr, self.ent_coef = self._schedule()
        self.actor_opt.lr = lr
        self.critic_opt.lr = lr

        obs = np.stack(self.buf_obs)
        next_obs = np.stack(self.buf_next_obs)
        act = np.asarray(self.buf_act)
        rew = np.asarray(self.buf_rew)
        term = np.asarray(self.buf_term, dtype=np.float64)
        end = np.asarray(self.buf_end, dtype=np.float64)
        n = len(act)

        adv, returns = self._compute_advantages(obs, next_obs, rew, term, end)

        # "Eski" politikanın, veriyi toplarken seçtiği eylemlere verdiği log-olasılık.
        # Toplama boyunca ağ değişmediği için şimdi toplu hesaplamak aynı sonucu verir.
        logits_old, _ = self.actor.forward(obs)
        logp_old = np.log(_softmax(logits_old)[np.arange(n), act] + 1e-12)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        n_mb = 0
        for _ in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                s = self._minibatch_step(obs[mb], act[mb], logp_old[mb], adv[mb], returns[mb])
                for k in stats:
                    stats[k] += s[k]
                n_mb += 1
        self.last_stats = {k: v / n_mb for k, v in stats.items()}
        self.last_stats.update(lr=lr, ent_coef=self.ent_coef)
        self.n_updates += 1

    def _minibatch_step(self, obs, act, logp_old, adv, returns) -> dict[str, float]:
        m = len(act)
        rows = np.arange(m)
        # Avantajı normalize et: ölçeği ödül büyüklüğünden bağımsız olur.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------- ACTOR: kırpılmış (clipped) amaç ----------
        logits, a_cache = self.actor.forward(obs)
        probs = _softmax(logits)
        logp_all = np.log(probs + 1e-12)
        logp = logp_all[rows, act]
        ratio = np.exp(logp - logp_old)  # r = π_yeni / π_eski

        # PPO amacı (büyütmek istediğimiz):  min( r·A , clip(r, 1-ε, 1+ε)·A )
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        policy_loss = -np.minimum(surr1, surr2).mean()  # en aza indirdiğimiz: eksi

        # Gradyan: min'in kırpılmış dalı seçildiyse sabittir -> gradyan 0.
        # Yani A>0 iken r zaten 1+ε'yi geçtiyse, ya da A<0 iken r 1-ε'nin altına
        # indiyse, bu örnek politikayı daha fazla itmez. Kırpmanın asıl işi bu.
        active = surr1 <= surr2  # kırpılmamış dal seçildi
        d_logp = np.where(active, -ratio * adv, 0.0) / m  # dLoss/dlog π(a|s)
        # log-softmax türevi: d log π(a) / d logit_j = 1[j=a] - π_j
        onehot = np.zeros_like(probs)
        onehot[rows, act] = 1.0
        d_logits = d_logp[:, None] * (onehot - probs)

        # Entropi bonusu: H = -Σ π log π. Büyük H = kararsız/keşifçi politika.
        # Loss'a -c·H ekliyoruz; dH/dlogit_j = -π_j (log π_j + H)
        entropy = -(probs * logp_all).sum(axis=1)
        d_ent = -probs * (logp_all + entropy[:, None])
        d_logits += -self.ent_coef * d_ent / m

        a_grads = self.actor.backward(a_cache, d_logits)
        _clip_grad_norm(a_grads, self.max_grad_norm)
        self.actor_opt.step(a_grads)

        # ---------- CRITIC: değer tahmini hatası ----------
        v, c_cache = self.critic.forward(obs)
        v = v[:, 0]
        value_loss = 0.5 * ((v - returns) ** 2).mean()
        d_v = ((v - returns) / m)[:, None]
        c_grads = self.critic.backward(c_cache, d_v)
        _clip_grad_norm(c_grads, self.max_grad_norm)
        self.critic_opt.step(c_grads)

        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy.mean()),
            "clip_frac": float((np.abs(ratio - 1) > self.clip_eps).mean()),
        }
