# Arayüz sözleşmesi (Faz 4)

PySide6 (yalnız **PySide6-Essentials**: QtCore, QtGui, QtWidgets; QtCharts/QtWebEngine YOK) + matplotlib (`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`). Yeni paket kurma. Bütün metinler Türkçe.

Motor ve analiz API'si: `engine/CONTRACT.md` (trainer, tester, background.Job, analysis, report), sonuç klasörü: `engine/storage.py`. Ortam: `maze_env/SPEC.md`. Ajan bulma: `engine.registry.discover()`. Ortak ayarlar: `engine.settings.load_shared_config()`.

## Dosya sahipliği
| Dosya | Sahip |
|---|---|
| `gui/maze_view.py` (`MazeView`, `ReplayPlayer`), `gui/play_tab.py` (`PlayTab`), `tests/test_gui_widgets.py` | Parça A |
| `app.py`, `gui/main_window.py`, `gui/settings_tab.py`, `gui/train_tab.py`, `gui/test_tab.py`, `gui/results_tab.py`, `gui/plot_canvas.py`, `gui/notify.py`, `tests/test_gui_app.py` | Parça B |

Parça B, A'nın sınıflarını yalnız imzalarıyla kullanır (`gui/maze_view.py`, `gui/play_tab.py` içindeki docstring'ler). A bitene kadar bu sınıflar `NotImplementedError` verebilir; B kendi testlerinde gerekirse bunları basit bir sahte widget ile yamalayabilir.

## Ana pencere (Parça B)
`app.py` → `QApplication`, `gui.main_window.MainWindow` (başlık "RL Labirent", ~1400x850), sekmeler sırayla: **Ayarlar, Eğitim, Test, Sonuçlar, Oyna**. Uzun iş sürerken pencere donmaz: eğitim/test `engine.background.Job` ile, `QTimer` (~100 ms) ile `poll()`. Pencere kapanırken çalışan Job'a `stop()` + kısa bekleme.

- **Ayarlar:** ajan seçimi (registry; baseline'lar listede yok; obs_mode bilgisi gösterilir; registry hataları uyarı olarak), etiket, bölüm sayısı, tekrar sayısı (varsayılan ortak ayardaki `egitim.tekrar_sayisi`), seed, ödüller (`step`, `wall`, `exit`, `new_tile`, `timeout`; varsayılan ortak ayardan, "varsayılana dön" butonu), ortam (eğitim labirent boyutu, görüş yarıçapı), hiperparametreler (`anahtar=değer` satırları ya da JSON; sayı/bool/str çözümlenir, hatalı girişte uyarı), kayıt sayısı (snapshots). Bu sekme bir `TrainConfig` üretir (`reward_overrides` yalnız varsayılandan farklı olanlar, `env_overrides` yalnız değişenler).
- **Eğitim:** Eğit / Durdur, ilerleme çubuğu (tekrar × bölüm), geçen süre. Solda `ReplayPlayer` + kayıt seçici (eşit aralıklı kayıtlar "Bölüm 1, 500, ..." ve "En son bölüm"; yeni kayıt olayıyla liste güncellenir; "en sonu otomatik göster" kutusu). Sağda **canlı öğrenme eğrisi** (matplotlib, episode olaylarından; başarı oranı ve ortalama ödül, hareketli ortalama, tekrarlar ayrı renk). Bitince bildirim, ardından `analysis.summarize` ve `save_all_plots`; "Teste geç" butonu çalışmayı Test sekmesinde seçer.
- **Test:** çalışma seçimi (`storage.list_runs`, açıklayıcı ad: ajan, etiket, tarih, durum), Normal / Büyük kutuları, Test et / Durdur, ilerleme. Bitince bildirim + `summarize`; sonuç tablosu (yöntem × başarı, ort. adım, verim, ödül, çarpma; her tür ayrı). Bölüm izleme: tür, yöntem, tekrar, seed seçip "İzle" → `tester.replay_test_episode` → `ReplayPlayer`.
- **Sonuçlar:** çalışma listesi (işaretlenebilir; ajan, etiket, durum, eğitim son başarı, test başarı normal/büyük), grafik alanı (seçim: öğrenme eğrisi başarı/ödül/adım, baseline karşılaştırma normal/büyük, genelleme, çalışma karşılaştırma normal/büyük) işaretli çalışmalara göre. Butonlar: Yenile, İçe aktar (klasör ya da zip; `report.import_runs`), Seçileni zip olarak dışa aktar (`report.export_run_zip`), Rapor dışa aktar (klasör seç → `report.export_report` → bitince rapor.html'i varsayılan tarayıcıda aç), Klasörü aç. Silme butonu YOK.
- **Oyna:** `gui.play_tab.PlayTab`.
- **Bildirim (`gui/notify.py`):** `QSystemTrayIcon.showMessage` (sistem tepsisi yoksa durum çubuğu mesajı). Uygulama simgesi `QStyle` standart simgelerinden.

## Testler
`QT_QPA_PLATFORM=offscreen` ile, pytest-qt yok: `QApplication.instance() or QApplication([])`, olay döngüsü için `QCoreApplication.processEvents()` / `QTest.qWait`. Testler sonuç klasörünü `tmp_path` altına yazmalı (gerçek `results/` kirlenmesin): motorun `results_root` alanını ve `storage.RESULTS_ROOT`'u kullanan yerleri yamalayarak.
