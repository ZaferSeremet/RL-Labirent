# Ortam + Uygulama Planı (taslak)

Durum: **Faz 0-6 bitti (2026-11-22).** Ekip kılavuzu: [README.md](README.md). Sıradaki: kodu ekibe ulaştırma (GitHub ya da zip) ve herkesin kendi ajanı. Deney akışı için bkz. [DENEY_AKISI.md](DENEY_AKISI.md).

## Faz 6 sonucu
- Kök [README.md](README.md): kurulum, uygulamayı açma, ajan yazma, deney akışı, sonuç gönderme, kurallar, sorun giderme.
- `kurulum.bat` (sanal ortam + paketler) ve `baslat.bat` (uygulamayı açar); Windows'ta çift tıklamayla.
- Temiz kopyada denendi: `kurulum.bat` (Python 3.14) ve Python 3.11.5 ile 98 test geçiyor. En düşük sürüm 3.11.
- Dil desteği: sağ üstte English/Türkçe düğmesi; arayüz, grafikler, tablolar ve rapor iki dilli (`engine/i18n.py`, kurallar [engine/I18N.md](engine/I18N.md)). Seçim `kullanici_ayarlari.json`'da (git'e girmez).
- Rapor dışa aktarma artık seçilen yerde tek bir `rapor_<tarih>` / `report_<tarih>` klasörü açar.
- Faz 5'ten sonra eklenenler: çıkışa yaklaşma ödülü `closer` (yol mesafesiyle, simetrik, varsayılan 0; [maze_env/SPEC.md](maze_env/SPEC.md)), Sonuçlar sekmesinde silme (onaylı, Geri Dönüşüm Kutusu'na; o an çalışan çalışma silinemez), Windows karanlık mod düzeltmesi.

## Faz 3-5 sonucu
- Uygulamayı başlat: `.venv\Scripts\python app.py` (sekmeler: Ayarlar, Eğitim, Test, Sonuçlar, Oyna).
- Motor: `engine/trainer.py` (eğitim, 5 eşit aralıklı + canlı kayıt, tekrar sayısı, durdurma), `engine/tester.py` (normal + büyük test, baseline'lar dahil, test bölümünü tekrar izleme), `engine/background.py` (ayrı süreç). Sözleşme: [engine/CONTRACT.md](engine/CONTRACT.md).
- Analiz ve rapor: `engine/analysis.py` (özet, öğrenme eğrisi, baseline, genelleme, karşılaştırma grafikleri), `engine/report.py` (zip paylaşma, içe aktarma, HTML + Markdown rapor).
- Arayüz: `gui/` (sözleşme [gui/CONTRACT.md](gui/CONTRACT.md)).
- 94 test geçiyor (~15 sn). Uçtan uca denendi: arka planda eğitim → test → özet → grafikler → rapor → zip → içe aktarma.
- Düzeltilen hata: Windows'ta arayüz `kayitlar/son.json`'u okurken eğitim süreci aynı dosyayı yazamayıp çöküyordu; yazma artık yeniden deniyor, canlı kayıt yazılamazsa eğitim düşmüyor.

## Faz 2 sonucu
- Ajan arayüzü `agents/base.py` (`Agent`, `EnvSpec`), şablon `agents/ornek_ajan.py`, kılavuz [agents/README.md](agents/README.md).
- Baseline'lar `baselines/`: `random`, `sag_el` (hafızasız; bakış yönünü gözlemdeki son eylemden okur).
- Motorun başlangıcı `engine/`: ortak ayar okuma (`settings.py`), ajanları otomatik bulma (`registry.py`), tek bölüm koşturma + izleme kaydı (`episode.py`).
- Arkadaşların deneme aracı: `.venv\Scripts\python tools\ajan_dene.py --agent <ad> --episodes 2000 [--buyuk]`
- 45 test geçiyor. Faz 2 sırasında ortamda bir hata bulundu ve düzeltildi: `simple` modda çıkışa ulaşılan adımda çöküyordu (regresyon testi eklendi).

## Faz 1 sonucu
- Ortam: `maze_env/`, sözleşme: [maze_env/SPEC.md](maze_env/SPEC.md), 29 birim testi (`tests/`) geçiyor, Gymnasium `check_env` iki modda da uyarısız.
- Ortak ayarlar: `configs/ortak_ayarlar.json` (normal 8x8, büyük 12x12, test seed'leri 10000-10099 / 20000-20099).
- Adım limiti ölçümü (varsayılan `3 x boş kare`, 100 test labirenti):

| Boyut | Limit | Sağ el başarı | Sağ el ort. / en kötü adım | Random başarı |
|---|---|---|---|---|
| 8x8 | 381 | 100/100 | 102 / 228 | 2/100 |
| 12x12 | 861 | 100/100 | 222 / 488 | 1/100 |

- Klavyeyle deneme: `.venv\Scripts\python tools\oyna_terminal.py --seed 42`

## Alınan kararlar (2026-11-22)
- GUI: **PySide6 (Qt) + matplotlib**, sekmeli masaüstü uygulaması.
- Eğitim tam hızda arka planda çalışır. İzleme, kaydedilmiş bölümlerin seçilen hızda oynatılmasıdır; hız eğitimi etkilemez.
- Eğitim boyunca **~5 eşit aralıklı bölüm kaydı + en son bölüm** izlenebilir (ör. 5000 bölümlük eğitimde 1, 1250, 2500, 3750, 5000).
- Eğitim sekmesinde solda labirent izleme, **sağda canlı öğrenme eğrisi**.
- Aynı ayarla **tekrar sayısı** seçilebilir (varsayılan 3); grafikler ortalama + sapma gösterir.
- Herkes kendi bilgisayarında çalışır. Sonuçlar klasör halinde toplanır, uygulama bunları **içe aktarıp** birleşik karşılaştırma ve rapor üretir. GitHub tercih edilir ama şart değil.
- İş bitince Windows köşe bildirimi (şart değil, kolaylık).
- Klavyeyle oynama modu olacak (ortam testi + demo).
- Model seçimi her üyenin kendisinde; uygulama algoritmadan bağımsız.

## Klasör yapısı
```
RL-Project/
  maze_env/        ortam: labirent üretimi, MazeEnv, en kısa yol
  agents/          her üyenin ajanı (+ ornek_ajan.py şablonu)
  baselines/       random.py, sag_el.py
  engine/          eğitim, test, kayıt, grafik, rapor
  gui/             PySide6 arayüz
  configs/         ortak_ayarlar.json (test seed'leri, varsayılan ödüller)
  results/         her eğitim/test bir klasör
  tests/           ortam birim testleri
  app.py           uygulamayı başlatır
  requirements.txt
  README.md
```

## Fazlar

### Faz 0: İskelet
- Klasör yapısı, `requirements.txt` (numpy, matplotlib, PySide6, gymnasium), git deposu.
- Ortak Python sürümü (ekip aynı sürümü kullanmalı).

### Faz 1: Ortam (`maze_env/`)
- Seed'li rastgele labirent üretimi (DFS / recursive backtracker). **Aynı seed her bilgisayarda aynı labirenti üretmeli**; test labirentleri bu sayede herkeste aynı.
- Ajan ve çıkış rastgele konumda başlar.
- `reset(seed)` / `step(action)` (Gymnasium arayüzü), 4 eylem.
- Ödüller config'den: adım cezası, duvar cezası, çıkış ödülü (süre dolunca ek ceza yok).
- Adım limiti, `terminated` (çıkış) / `truncated` (süre).
- Gözlem: yerel pencere, ziyaret bilgisi, son eylem. `obs_mode="full"` ve `"simple"`.
- `info`: success, steps, wall_hits, shortest_path (verimlilik metriği için).
- Çizim için labirent durumu (GUI kullanacak).
- Birim testleri + klavye modu ile elle doğrulama.

### Faz 2: Ajan arayüzü ve baseline'lar
- Ajan arayüzü: `act(obs, greedy)`, `learn(...)`, `end_episode()`, `save()`, `load()`.
- `agents/ornek_ajan.py`: arkadaşlara gönderilecek şablon.
- Baseline'lar: random, sağ el takibi.
- **Bu fazdan sonra arkadaşlar ajan yazmaya başlayabilir**, GUI'yi beklemeden.

### Faz 3: Motor (`engine/`)
- Eğitim döngüsü, ayrı süreçte (GUI donmasın).
- Her bölümün dönüş / adım / başarı / çarpma kaydı (CSV).
- 5 + 1 bölüm kaydı (izleme için).
- Tekrar sayısı (3 seed) desteği.
- Test: 100 normal + 100 büyük labirent; random ve sağ el otomatik dahil.
- Her çalışma `results/<tarih>_<ajan>_<etiket>/` klasörüne: `ayarlar.json`, `egitim_log.csv`, `test_normal.csv`, `test_buyuk.csv`, `grafikler/`, `model`.

### Faz 4: GUI (`gui/`)
| Sekme | İçerik |
|---|---|
| Ayarlar | Ajan seçimi, ödül ayarları, ortam ayarları, eğitim ayarları (bölüm sayısı, tekrar sayısı) |
| Eğitim | Eğit butonu, ilerleme çubuğu; solda labirent + hız çubuğu + kayıt seçici, sağda canlı öğrenme eğrisi |
| Test | Normal / büyük, Test et butonu, ilerleme, bölüm tekrar izleme, bitince bildirim |
| Sonuçlar | Çalışma listesi, karşılaştırma grafikleri, içe aktar, rapor dışa aktar |
| Oyna | Klavyeyle oynama, ajanın gördüğü pencere gösterilir |

### Faz 5: Sonuçlar ve rapor
- Grafikler: öğrenme eğrisi (madde 5), baseline çubuk grafiği (6), varyasyon eğrileri üst üste (7), normal vs büyük (8), verimlilik (adım / en kısa yol).
- Arkadaşların sonuç klasörlerini içe aktarma.
- **Rapor dışa aktarma:** tüm grafikler + tablolar + ayarlar tek klasörde, rapora hazır bir belgeyle.

### Faz 6: Ekibe dağıtım
- README: kurulum, ajan yazma, eğitim/test, sonuç gönderme. **Bitti.**

## Açık sorular
- Dışa aktarılan raporun biçimi (Word, PDF, HTML?). Şu an HTML + Markdown, Türkçe ya da İngilizce.
- Labirent boyutları, pencere boyutu, adım limiti, varsayılan ödül değerleri: Faz 1'de deneyerek seçilecek.
- ~~Ortak Python sürümü.~~ En az 3.11 (3.11 ve 3.14'te denendi).
- Kodun ekibe ulaştırılması: GitHub deposu mu, zip mi?
