# Arayüz sözleşmesi (Faz 4)

PySide6 (yalnız **PySide6-Essentials**: QtCore, QtGui, QtWidgets; QtCharts/QtWebEngine YOK) + matplotlib (`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`). Yeni paket kurma. Bütün metinler Türkçe.

Motor ve analiz API'si: `engine/CONTRACT.md` (trainer, tester, background.Job, analysis, report), sonuç klasörü: `engine/storage.py`. Ortam: `maze_env/SPEC.md`. Ajan bulma: `engine.registry.discover()`. Ortak ayarlar: `engine.settings.load_shared_config()`.

## Dosya sahipliği
| Dosya | Sahip |
|---|---|
| `gui/maze_view.py` (`MazeView`, `ReplayPlayer`), `tests/test_gui_widgets.py` | Parça A |
| `gui/race_view.py` (`RacePlayer`), `tests/test_race_view.py` | Yarış A (2026-11-28) |
| `gui/final_tab.py` (`FinalTab`), Test sekmesindeki yarış izleme | Yarış B (2026-11-28) |
| `app.py`, `gui/main_window.py`, `gui/settings_tab.py`, `gui/train_tab.py`, `gui/test_tab.py`, `gui/results_tab.py`, `gui/plot_canvas.py`, `gui/notify.py`, `tests/test_gui_app.py` | Parça B |

Parça B, A'nın sınıflarını yalnız imzalarıyla kullanır (`gui/maze_view.py` içindeki docstring'ler). A bitene kadar bu sınıflar `NotImplementedError` verebilir; B kendi testlerinde gerekirse bunları basit bir sahte widget ile yamalayabilir.

## Ana pencere (Parça B)
`app.py` → `QApplication`, `gui.main_window.MainWindow` (başlık "RL Labirent", ~1400x850), sekmeler sırayla: **Ayarlar, Eğitim, Test, Sonuçlar, Final Test** (İngilizce: Settings, Training, Test, Results, Final Test). Uzun iş sürerken pencere donmaz: eğitim/test `engine.background.Job` ile, `QTimer` (~100 ms) ile `poll()`. Pencere kapanırken çalışan Job'a `stop()` + kısa bekleme.

- **Ayarlar:** ajan seçimi (registry; baseline'lar listede yok; obs_mode bilgisi gösterilir; registry hataları uyarı olarak), etiket, bölüm sayısı, tekrar sayısı (varsayılan ortak ayardaki `egitim.tekrar_sayisi`), seed, ödüller (`step`, `wall`, `exit`, `new_tile`, `timeout`; varsayılan ortak ayardan, "varsayılana dön" butonu), ortam (eğitim labirent boyutu, görüş yarıçapı), hiperparametreler (`anahtar=değer` satırları ya da JSON; sayı/bool/str çözümlenir, hatalı girişte uyarı), kayıt sayısı (snapshots). Bu sekme bir `TrainConfig` üretir (`reward_overrides` yalnız varsayılandan farklı olanlar, `env_overrides` yalnız değişenler).
- **Eğitim:** Eğit / Durdur, ilerleme çubuğu (tekrar × bölüm), geçen süre. Solda `ReplayPlayer` + kayıt seçici (eşit aralıklı kayıtlar "Bölüm 1, 500, ..." ve "En son bölüm"; yeni kayıt olayıyla liste güncellenir; "en sonu otomatik göster" kutusu). Sağda **canlı öğrenme eğrisi** (matplotlib, episode olaylarından; başarı oranı ve ortalama ödül, hareketli ortalama, tekrarlar ayrı renk). Bitince bildirim, ardından `analysis.summarize` ve `save_all_plots`; "Teste geç" butonu çalışmayı Test sekmesinde seçer.
- **Test:** çalışma seçimi (`storage.list_runs`, açıklayıcı ad: ajan, etiket, tarih, durum), Normal / Büyük kutuları, Test et / Durdur, ilerleme. Bitince bildirim + `summarize`; sonuç tablosu (yöntem × başarı, ort. adım, verim, ödül, çarpma; her tür ayrı). Yarış izleme: tür, tekrar, seed seçip "İzle" → `engine.race.race_test_episode` → `gui.race_view.RacePlayer` (eğitilmiş ajan ve bütün baseline'lar aynı labirentte, farklı renklerle).
- **Sonuçlar:** çalışma listesi (işaretlenebilir; ajan, etiket, durum, eğitim son başarı, test başarı normal/büyük), grafik alanı (seçim: öğrenme eğrisi başarı/ödül/adım, baseline karşılaştırma normal/büyük, genelleme, çalışma karşılaştırma normal/büyük) işaretli çalışmalara göre. Butonlar: Yenile, İçe aktar (klasör ya da zip; `report.import_runs`), Seçileni zip olarak dışa aktar (`report.export_run_zip`), Rapor dışa aktar (klasör seç → `report.export_report` → bitince rapor.html'i varsayılan tarayıcıda aç), Klasörü aç, Sil (işaretliler ya da seçili satır; onay sorulur, Geri Dönüşüm Kutusu'na taşınır; o an eğitilen/test edilen çalışma silinemez).
- **Final Test:** `gui.final_tab.FinalTab`. Demo için: çalışma (yalnız modeli olan) ve tekrar seçimi, labirent boyutu (QSpinBox, `race.MIN_MAZE_SIZE`..`race.MAX_MAZE_SIZE`, varsayılan ortak ayardaki normal test boyutu; yanında 8x8 / 12x12 hızlı düğmeleri), seed (QLineEdit, sadece rakam; "Rastgele" düğmesi `race.RANDOM_SEED_RANGE`'den seed yazar; boşsa Başlat da rastgele seçer), "Başlat" → `engine.race.race_custom` → `RacePlayer`. Seçilen çalışma test sekmesindeki çalışmalardan bağımsızdır.

## Yarış oynatıcı (`gui/race_view.py`)
Girdi `engine/race.py` belgesindeki yarış kaydı. Genel API dosyadaki imzalardır (`RacePlayer.load/clear/play/pause/rewind/seek/set_speed/is_playing/current_step/n_steps/race`, `finished` sinyali, `racer_label`, `racer_colors`, `RACER_COLORS`).
- **Zaman çizelgesi ortak:** adım i'de oyuncu `positions[min(i, kendi_adım)]`'da; bitiren yerinde kalır. `n_steps()` en uzun oyuncu.
- **Çizim:** labirent (`MazeView` renkleri: duvar, boş, çıkış, başlangıç), her oyuncunun izi (kendi renginde yarı saydam çizgi, oyuncular arası küçük kaydırma ki izler üst üste binmesin) ve işareti (kendi renginde daire, beyaz çerçeve; aynı karedeki oyuncular kareyi paylaşacak şekilde küçük kaydırma). Görüş penceresi gösterilmez. "İzler" kutusu izleri açıp kapar.
- **Lejant:** her oyuncu için renk karesi + `racer_label`.
- **Durum tablosu** (QTableWidget, satır = oyuncu, o anki adıma göre): Oyuncu (renk karesi + ad), Durum (Yolda / Çıkışa ulaştı / Süre doldu), Adım, Duvar çarpması (o ana kadar; konum değişmediyse çarpma), Gezilen kare (o ana kadar farklı kare), Toplam ödül (o ana kadar), Sıra (bitirenler adım sayısına göre 1., 2., ...; bitirmeyen "-"). Yarış sonunda en iyi satır vurgulanır.
- Kontroller `ReplayPlayer` ile aynı his: Oynat/Duraklat, Başa sar, adım kaydırıcısı, hız 1-200 adım/sn (logaritmik). Bilgi satırı: başlık, "Adım i/N", labirent boyutu ve seed.
- **Bildirim (`gui/notify.py`):** `QSystemTrayIcon.showMessage` (sistem tepsisi yoksa durum çubuğu mesajı). Uygulama simgesi `QStyle` standart simgelerinden.

## Testler
`QT_QPA_PLATFORM=offscreen` ile, pytest-qt yok: `QApplication.instance() or QApplication([])`, olay döngüsü için `QCoreApplication.processEvents()` / `QTest.qWait`. Testler sonuç klasörünü `tmp_path` altına yazmalı (gerçek `results/` kirlenmesin): motorun `results_root` alanını ve `storage.RESULTS_ROOT`'u kullanan yerleri yamalayarak.

## Dil
Kullanıcıya görünen her metin `engine.i18n.tx("Türkçe", "English")` ile yazılır; kurallar [engine/I18N.md](../engine/I18N.md). Ana pencere sağ üstteki düğmeyle dili değiştirir ve sekmeleri baştan kurar (eğitim/test sürerken değiştirmez). Denetim: `tests/test_i18n.py`.
