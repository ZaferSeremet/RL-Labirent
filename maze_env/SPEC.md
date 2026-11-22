# MazeEnv belirtimi

Ortamın davranışını tanımlayan sözleşme. Ajan yazan herkes buna göre yazar; testler buna göre yazılır.

## Kullanım
```python
from maze_env import MazeEnv, EnvConfig, RewardConfig

env = MazeEnv(maze_size=8, view_radius=1, obs_mode="simple")
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(action)
```
`MazeEnv(config=EnvConfig(...))` ya da anahtar kelimelerle (`MazeEnv(maze_size=12)`) oluşturulur. İkisi birlikte verilirse anahtar kelimeler config'in üstüne yazar.

## Labirent
- `maze_size = n` hücrelik kare labirent, `(2n+1) x (2n+1)` karelik ızgarada. Hücreler tek indisli karelerdir: hücre `(cy, cx)` → kare `(2cy+1, 2cx+1)`.
- Dış çerçeve hep duvardır. DFS ile üretilir, **döngüsüzdür**: her iki boş kare arasında tek yol vardır. Boş kare sayısı `2n² - 1`.
- Ajan ve çıkış iki farklı hücrenin karesinde başlar. Aralarındaki en kısa yol mümkünse en az `2n` karedir.
- **Aynı seed her bilgisayarda ve her Python sürümünde aynı labirenti, başlangıcı ve çıkışı üretir** (`random.Random.random()` dışında bir rastgelelik kaynağı kullanılmaz).

## Seed davranışı
- `reset(seed=s)`: bölüm `s` seed'iyle üretilir, ayrıca sonraki seed'siz `reset()` çağrılarının kaynağı `s` ile yeniden tohumlanır.
- `reset()`: bir önceki tohumdan türetilen yeni bir seed kullanır. Yani `reset(seed=s)` ve ardından gelen seed'siz `reset()` dizisi her seferinde aynı labirent dizisini verir.
- Bölümün seed'i `info["episode_seed"]` içindedir.

## Eylemler
`Discrete(4)`: `0 UP` (yukarı, satır-1), `1 RIGHT` (sağ, sütun+1), `2 DOWN` (aşağı, satır+1), `3 LEFT` (sol, sütun-1).
Duvara doğru yapılan eylemde ajan yerinde kalır ve bu bir "duvara çarpma" sayılır.
Geçersiz eylem `ValueError`; `reset()`'ten önce ya da bölüm bittikten sonra `step()` `RuntimeError` verir.

## Ödül (her adımda)
`RewardConfig` alanları ve varsayılanları:

| Alan | Varsayılan | Ne zaman |
|---|---|---|
| `step` | -0.01 | Her adımda |
| `wall` | -0.05 | Duvara çarpınca (`step`'e ek) |
| `exit` | +5.0 | Çıkış karesine girince |
| `new_tile` | 0.0 | İlk kez girilen kareye girince (çıkış dahil) |
| `timeout` | 0.0 | Adım limiti dolduğu adımda |
| `closer` | 0.0 | Ajan bir kareye geçince: çıkışa labirent içi yol mesafesi 1 azaldıysa `+closer`, 1 arttıysa `-closer`. Duvara çarpmada 0 |

Bir adımın ödülü uygun olanların toplamıdır.

`closer` (çıkışa yaklaşma) notları:
- Mesafe kuş uçuşu değil, duvarlar dolaşılarak gidilen en kısa yoldur (BFS). Manhattan mesafesi labirentte yanıltır: çıkışa kuş uçuşu yaklaşan yol çıkmaz sokak olabilir.
- Simetrik olduğu için ileri-geri giderek ödül toplanamaz: bir bölümde toplanan `closer` ödülü tam olarak `closer x (başlangıç mesafesi - son mesafe)`.
- Mesafe ajanın gözlemine girmez; ajan çıkışın yerini yine bilmez, yalnız ödülden dolaylı bir işaret alır.
- Döngüsüz labirentte her geçiş mesafeyi tam 1 değiştirir; bu yüzden işaret güçlüdür. Deneyde "bir şeyi değiştir" varyasyonu olarak kullanılabilir.

## Bölüm sonu
- `terminated = True`: ajan çıkış karesine girdi.
- `truncated = True`: çıkışa girilmeden adım sayısı `max_steps`'e ulaştı. İkisi aynı anda `True` olmaz.
- `max_steps` varsayılanı `3 x (2n² - 1)`; `EnvConfig.max_steps` ile değiştirilebilir. `env.max_steps` ile okunur.

## Gözlem
Ajan **kendi koordinatını, çıkışın koordinatını ve labirentin tamamını görmez.** Görüş penceresi `k x k`, `k = 2*view_radius + 1`, ajan ortada. Pencerenin ızgara dışına taşan kısmı duvar sayılır.

### `obs_mode="full"`: `Box(0, 1, (3k² + 5,), float32)`
Sırayla:
1. `k²` duvar haritası (1 = duvar)
2. `k²` çıkış haritası (1 = çıkış bu karede)
3. `k²` ziyaret haritası (1 = bu kareye bu bölümde daha önce girildi; ajanın bulunduğu kare her zaman 1)
4. `5` son eylem one-hot (indis 0-3 eylem, 4 = bölüm başı, henüz eylem yok)

Haritalar satır satır düzleştirilir (sol üstten başlayarak).

### `obs_mode="simple"`: `Discrete(11520)`, tablo yöntemleri için tek tam sayı
- 4 komşu (yukarı, sağ, aşağı, sol), her biri: `0` boş ve girilmemiş, `1` boş ve girilmiş, `2` duvar, `3` çıkış
- Çıkış yönü: `0` pencerede görünmüyor, `1..8` = K, KD, D, GD, G, GB, B, KB (ajana göre; satır yukarı = K). Ajan çıkışın üstündeyken (yalnız `terminated` adımının gözlemi) `0`.
- Son eylem: `0-4`

Kodlama: `((((up*4 + right)*4 + down)*4 + left)*9 + exit_dir)*5 + last_action`.
`decode_simple_state(s)` bunu geri çözer. Çıkış yönü pencerenin tamamına (`view_radius`) bakar; komşular her zaman yalnız 4 bitişik karedir.

### `env.observe_dict()`
Aynı bilginin yapılandırılmış hali: `tiles` (k×k, 0 boş, 1 duvar, 2 çıkış), `visited` (k×k), `last_action`. Ajan kendi gözlem biçimini kurmak isterse bunu kullanabilir; ek bilgi içermez.

## info
`success` (çıkışa ulaşıldı mı), `steps`, `wall_hits`, `visited_tiles` (girilmiş farklı kare sayısı, başlangıç dahil), `shortest_path` (başlangıçtan çıkışa en kısa yol, ajan kullanmamalı), `max_steps`, `episode_seed`.

## Arayüz ve hata ayıklama (ajana verilmez)
- `env.get_render_state()`: `grid`, `agent`, `exit`, `start`, `visited` (tam ızgara), `view_radius`, `steps`, `last_action`.
- `env.render_text()` / `render_mode="ansi"`: `#` duvar, `.` girilmiş, boşluk girilmemiş, `A` ajan, `E` çıkış.
