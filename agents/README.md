# Ajan yazma kılavuzu

## Kısaca
1. `agents/ornek_ajan.py` dosyasını `agents/<algoritman>.py` olarak kopyala.
2. `name` alanını değiştir (benzersiz olmalı, ör. `"sarsa"`), istediğin gözlemi `obs_mode` ile seç.
3. `act`, `learn`, `end_episode` metodlarını doldur.
4. Dene:
   ```bash
   .venv\Scripts\python tools\ajan_dene.py --agent <name> --episodes 2000
   ```
   Eğitim sırasında başarı oranını yazdırır, sonunda 100 test labirentinde random ve sağ el ile karşılaştırır.

`agents/` klasörüne koyduğun ve `Agent`'tan türeyen her sınıf otomatik bulunur. Arayüz geldiğinde aynı dosya değişmeden orada da çalışacak.

## Metodlar
| Metod | Ne zaman çağrılır | Ne yapmalı |
|---|---|---|
| `act(obs, greedy)` | Her adımda | Eylem döndür (0 yukarı, 1 sağ, 2 aşağı, 3 sol). `greedy=True` testte, keşif yapma. |
| `learn(obs, action, reward, next_obs, terminated, truncated, info)` | Her eğitim adımından sonra | TD güncellemesi (Q-learning, SARSA, DQN...) |
| `begin_episode(obs, info)` | Bölüm başında | İstersen bölüm tamponunu sıfırla |
| `end_episode()` | Bölüm sonunda | Monte Carlo / REINFORCE güncellemesi, epsilon azaltma |
| `save(path)` / `load(path)` | Eğitim sonunda / testten önce | Varsayılanı pickle ile her şeyi kaydeder; PyTorch kullanıyorsan `torch.save` ile ez |

## Gözlem
Ayrıntı: [maze_env/SPEC.md](../maze_env/SPEC.md).
- `obs_mode = "simple"`: 0 ile 11519 arasında tek tam sayı. Q-tablosu için doğrudan satır indisi: `np.zeros((11520, 4))`.
- `obs_mode = "full"`: `3k² + 5` uzunluğunda 0/1 vektör (duvar haritası, çıkış haritası, ziyaret haritası, son eylem). Sinir ağına doğrudan girdi.

`self.spec` içinde gözlem/eylem uzayı, `view_radius` ve `window_size` var.

## Kurallar
- **Yalnız gözlemi kullan.** Ortama (`env`) erişimin yok; `info` içindeki `shortest_path` gibi alanları kararlarında kullanma.
- **Hazır RL kütüphanesi yasak** (Stable-Baselines vb.). NumPy ve PyTorch serbest.
- **Rastgelelik için `self.rng` kullan** (seed'e bağlı). `random` ya da `np.random` doğrudan kullanılırsa sonuçlar tekrar üretilemez.
- **`terminated` ile `truncated` farklıdır:** çıkışa ulaşmak gerçek bitiştir (sonraki değer 0); adım limiti dolması değildir (bootstrap yap).
- Hiperparametreleri `super().__init__(spec, seed, alpha=..., ...)` ile ver; sonuç klasörüne otomatik yazılır.
