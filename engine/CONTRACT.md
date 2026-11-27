# Motor sözleşmesi (Faz 3-5)

Bu belge ve `engine/` içindeki imza taslakları sözleşmedir. **İmzaları değiştirme**; değişiklik gerekiyorsa raporuna `SÖZLEŞME:` diye yaz.
Sonuç klasörü biçimi ve okuma/yazma: `engine/storage.py` (hazır, kullan). Ortam: `maze_env/SPEC.md`. Ajan arayüzü: `agents/base.py`. Bölüm koşturma: `engine/episode.py` (`run_episode`). Ortak ayarlar: `engine/settings.py`.

## Genel kurallar
- Python 3.14, `.venv`. Bağımlılıklar: numpy, gymnasium, matplotlib, PySide6-Essentials, pytest. **pandas yok**, yeni paket ekleme.
- Ajan bulma: `engine.registry.discover()`. Baseline'lar `cls.is_baseline == True` olanlar.
- Bütün yazılar Türkçe (sütun adları, dosya adları storage'daki gibi).

## trainer: `train(cfg, progress=None, stop=None) -> Path`
1. `storage.new_run_dir(cfg.agent_name, cfg.label, cfg.results_root)` ile klasör aç.
2. `ayarlar.json` yaz: `run_id` (klasör adı), `created` (ISO zaman), `durum` ("egitiliyor"), `agent` {`name`, `class` ("modül.Sınıf"), `obs_mode`, `is_baseline`}, `train_config` (TrainConfig alanları, results_root hariç), `env` (eğitim ortamının `EnvConfig.to_dict()`'i), `shared` (o anki `load_shared_config()`), `hyperparameters` (ilk tekrardaki `agent.hyperparameters()`).
3. Her tekrar `r in range(repeats)`:
   - Ortam: `MazeEnv(env_config_for("egitim", cls.obs_mode, shared, cfg.reward_overrides, **cfg.env_overrides))`.
   - Ajan: `cls(EnvSpec.from_env(env), seed=cfg.seed + r, **cfg.hparams)`.
   - Labirent dizisini sabitle: `env.reset(seed=shared["egitim"]["seed_baslangic"] + cfg.seed + r)`; sonra bölümler `run_episode(env, agent, train=True, record=...)` (seed'siz).
   - Her bölümü `egitim_log.csv`'ye yaz (`TRAIN_FIELDS`, `repeat`=r, `episode` 1'den başlar). Dosyayı periyodik flush et (arayüz okuyabilsin).
   - **Eşit aralıklı kayıt (yalnız r == 0):** bölüm numaraları `sorted(set(round(x) for x in numpy.linspace(1, episodes, snapshots)))` (snapshots ≥ 1; episodes < snapshots ise her bölüm). Bu bölümler `record=True` ile koşulur, `storage.snapshot_path(run_dir, 0, e)`'ye yazılır.
   - **Canlı kayıt (her tekrar):** her `live_every` bölümde bir (`None` → `max(1, episodes // 100)`) `record=True`, `storage.live_path(run_dir)`'e yazılır (üzerine).
   - Tekrar sonunda `agent.save(storage.model_path(run_dir, r))`.
4. Bitince `ayarlar.json`: `durum` = "tamamlandi" (durdurulduysa "durduruldu"), `sure_sn`, `tamamlanan_tekrar`.
5. **stop:** `stop.is_set()` ise mevcut bölüm biter, o tekrarın modeli kaydedilir, eğitim biter (`durduruldu`).
6. **progress olayları** (progress verildiyse; hepsi JSON'a çevrilebilir sade tipler, yol `str`):
   - `{"type": "start", "run_dir", "episodes", "repeats", "agent"}`
   - `{"type": "episode", "repeat", "episode", "total_reward", "steps", "success"}` (her bölüm)
   - `{"type": "snapshot", "repeat", "episode", "path"}`
   - `{"type": "live", "repeat", "episode", "path"}`
   - `{"type": "repeat_done", "repeat", "model_path"}`
   - `{"type": "done", "run_dir", "stopped": bool}`
7. Baseline (`is_baseline`) için `train` → `ValueError` (eğitilmez).

## tester: `test_run(run_dir, kinds=("normal","buyuk"), progress=None, stop=None) -> {kind: Path}`
- Ayarları `ayarlar.json`'dan okur; ancak test labirentleri (`shared["test"]`: seed'ler ve boyutlar) her zaman GÜNCEL ortak ayardan gelir (`tester.test_shared`), ödüller ve ortam ayarları çalışmanın kendi kaydından. Test ortamı: `env_config_for(kind, obs_mode, shared, reward_overrides, **env_overrides)` (ayarlar.json'daki `train_config` ve `shared` ile; boyut test türünden gelir). Seed'ler `seeds_for(kind, shared)`.
- Ajan: her `model_rep<r>.pkl` için `cls(spec, seed=train_seed + r, **hparams)` + `load`, greedy (`run_episode(..., train=False)`), her seed bir satır, `method` = ajan adı, `is_baseline`=False, `repeat`=r.
- Baseline'lar: registry'deki bütün baseline'lar, kendi `obs_mode`'larıyla, `seed=0`, tek sefer (`repeat`=0), `is_baseline`=True.
- `efficiency` = `EpisodeResult.efficiency` (başarısızsa boş).
- `test_<kind>.csv` baştan yazılır (varsa silinip yeniden). Ajan sınıfı bulunamazsa anlaşılır bir `KeyError`.
- Olaylar: `{"type": "start", "run_dir", "kinds", "total"}` (toplam bölüm), `{"type": "test_episode", "kind", "method", "repeat", "seed", "success", "steps"}`, `{"type": "kind_done", "kind", "path"}`, `{"type": "done", "run_dir", "stopped"}`.
- `stop`: mevcut bölüm biter, o ana kadarki CSV kalır, `done` ile `stopped=True`.
- `replay_test_episode(run_dir, kind, method, repeat, seed)`: aynı kurulumla tek bölümü `record=True` koşturup trajectory döndürür (baseline ise `method` baseline adı). Aynı girdiler → aynı kayıt.

## background: `Job`
- `multiprocessing` (Windows'ta spawn) ile ayrı süreç; olaylar bir `Queue` ile gelir. Süreç içindeki hedef fonksiyon modül düzeyinde olmalı (picklable).
- `Job("train", cfg=TrainConfig(...))` → `trainer.train`; `Job("test", run_dir=..., kinds=[...])` → `tester.test_run`.
- `poll()` kuyruktaki olayları sırayla döndürür, bloklamaz. Süreç hata ile biterse son olay `{"type": "error", "message", "traceback"}`.
- `stop()` süreçler arası bir `Event` set eder (trainer/tester'a `stop` olarak iletilir). `is_alive()`.
- Süreç olağan dışı ölürse (exitcode ≠ 0 ve `done`/`error` gelmediyse) `poll()` bir `error` olayı üretir.

## analysis
- `load_run`: `storage.read_settings`, `read_csv` (egitim_log, test_normal, test_buyuk). Olmayan dosya → boş liste.
- `summarize(run_dir)` → `ozet.json` yazar, döndürür:
  ```json
  {"run_id", "agent", "label",
   "egitim": {"bolum": N, "tekrar": R, "son_basari": ..., "son_ortalama_odul": ..., "son_ortalama_adim": ...},
   "test": {"normal": {"<method>": {"basari": ort, "basari_std": tekrarlar arası std, "adim": ort (başarılı), "verim": ort (başarılı), "odul": ort, "carpma": ort, "n": bölüm}}, "buyuk": {...}}}
  ```
  `son_*` = her tekrarın son %10 bölümünün ortalaması, sonra tekrarların ortalaması. Test yoksa ilgili anahtar boş sözlük.
- Grafikler `matplotlib.figure.Figure` (pyplot yok; Agg ve Qt'de çalışır), Türkçe başlık/eksen. Öğrenme eğrisi: x bölüm, y metrik; her tekrar için hareketli ortalama (`window` None → `max(1, episodes // 50)`), tekrarların ortalaması çizgi + std gölge; birden çok çalışma aynı eksende, etiket `"<ajan> <label>"`.
- `save_all_plots(run_dir)`: `grafikler/ogrenme_basari.png`, `ogrenme_odul.png`, `ogrenme_adim.png`, varsa `test_normal.png`, `test_buyuk.png`, `genelleme.png`; yazılan yolları döndürür.

## report
- `export_run_zip`: klasörü tek zip; zip içinde kök klasör adı run_id.
- `import_runs`: her kaynak klasör ya da zip; içinde `ayarlar.json` olan klasör(ler)i `root` (None → `RESULTS_ROOT`) altına kopyalar; aynı ad varsa `_2` ekler; kopyalanan yolları döndürür. Geçersiz kaynak → `ValueError`.
- `export_report(run_dirs, out_dir)`: `out_dir` altında `rapor.html` (tek dosya, görseller `grafikler/` göreli), `rapor.md`, `grafikler/*.png`, `tablolar/*.csv`. İçerik: her çalışmanın ayar özeti (ajan, etiket, bölüm, tekrar, ödüller, hiperparametreler), öğrenme eğrileri, normal ve büyük testte baseline tablosu + grafik, genelleme; birden çok çalışma varsa karşılaştırma grafiği/tablosu. Eksik test verisi olan bölüm atlanır, hata vermez.
